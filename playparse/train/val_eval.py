"""Generation-based validation during training, scored by the P1 eval metrics.

Val loss says how well the model predicts the gold tokens; it does not say whether
greedy decoding produces the right JSON. This module provides the `val_callback`
the training loop calls every `gen_every` steps: greedy-decode a fixed val subset
with `playparse.train.generate` (the exact training prompt), score each output with
`playparse.eval.metrics.score_example` (the same parsing and matching as the eval
harness), and return flat metrics that land in metrics.jsonl and can drive early
stopping and best-checkpoint selection (`early_stop_metric`).

The subset is stratified by bucket so the rare, hard buckets (lateral, fumble,
challenge) are present at all: a uniform 500-play val sample would hold about 0.2
laterals. Because that changes the mix, three overall numbers are logged:

    val_exact_match          micro average over the subset as drawn
    val_exact_match_natural  per-bucket exact match reweighted to the bucket shares
                             of the full val file (an estimate of plain val EM)
    val_exact_match_macro    unweighted mean over buckets (every bucket counts once)

plus `val_exact_match/<bucket>`, `val_n/<bucket>`, valid rate, and credit F1.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from torch import nn

from playparse.eval.metrics import score_example
from playparse.ffscore.schema import PlayLabel


def stratified_subset(
    records: Sequence[Mapping[str, Any]],
    n: int,
    seed: int = 0,
    strategy: str = "balanced",
) -> list[dict]:
    """A seeded sample of `n` records, stratified by `bucket`.

    balanced      equal share per bucket; a bucket with too few records gives all
                  it has and the remainder is spread over the others.
    proportional  each bucket's share follows its frequency (largest remainder),
                  with at least one record per bucket when n allows.

    Deterministic for a given (records, n, seed, strategy). Returned in a stable
    order (bucket name, then original index) so batches are reproducible.
    """
    if n <= 0:
        return []
    by_bucket: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_bucket[str(r["bucket"])].append(i)
    if n >= len(records):
        return [dict(r) for r in records]
    buckets = sorted(by_bucket)
    alloc: dict[str, int] = {b: 0 for b in buckets}
    if strategy == "balanced":
        remaining, open_b = n, list(buckets)
        while remaining > 0 and open_b:
            share = max(1, remaining // len(open_b))
            for b in list(open_b):
                take = min(share, len(by_bucket[b]) - alloc[b], remaining)
                alloc[b] += take
                remaining -= take
                if alloc[b] >= len(by_bucket[b]):
                    open_b.remove(b)
                if remaining == 0:
                    break
    elif strategy == "proportional":
        total = len(records)
        exact = {b: n * len(by_bucket[b]) / total for b in buckets}
        alloc = {b: min(len(by_bucket[b]), max(1, int(exact[b]))) for b in buckets}
        while sum(alloc.values()) > n:  # the max(1, ...) floor overshot
            b = max(alloc, key=lambda k: alloc[k])
            alloc[b] -= 1
        for b in sorted(buckets, key=lambda k: exact[k] - int(exact[k]), reverse=True):
            if sum(alloc.values()) >= n:
                break
            if alloc[b] < len(by_bucket[b]):
                alloc[b] += 1
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    rng = random.Random(seed)
    chosen: list[int] = []
    for b in buckets:
        idx = by_bucket[b]
        chosen.extend(sorted(rng.sample(idx, alloc[b])))
    return [dict(records[i]) for i in chosen]


def score_generations(
    records: Sequence[Mapping[str, Any]],
    texts: Sequence[str],
    population_shares: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Score raw generations against gold labels; return flat val_* metrics."""
    per: dict[str, Counter] = defaultdict(Counter)
    tot: Counter = Counter()
    for rec, text in zip(records, texts, strict=True):
        s = score_example(PlayLabel.from_json(rec["label"]), text)
        row = Counter(n=1, valid=int(s.valid), exact=int(s.exact), tp=s.tp, fp=s.fp, fn=s.fn)
        per[str(rec["bucket"])].update(row)
        tot.update(row)
    n = max(1, tot["n"])
    f1_den = 2 * tot["tp"] + tot["fp"] + tot["fn"]
    out: dict[str, Any] = {
        "val_gen_n": tot["n"],
        "val_exact_match": tot["exact"] / n,
        "val_valid_rate": tot["valid"] / n,
        "val_credit_f1": (2 * tot["tp"] / f1_den) if f1_den else 1.0,
    }
    bucket_em = {b: c["exact"] / c["n"] for b, c in sorted(per.items())}
    out["val_exact_match_macro"] = sum(bucket_em.values()) / len(bucket_em) if bucket_em else 0.0
    if population_shares:
        covered = {b: w for b, w in population_shares.items() if b in bucket_em}
        z = sum(covered.values())
        out["val_exact_match_natural"] = sum(w * bucket_em[b] for b, w in covered.items()) / z if z else 0.0
    for b, em in bucket_em.items():
        out[f"val_exact_match/{b}"] = em
        out[f"val_n/{b}"] = per[b]["n"]
    return out


def harness_val_callback(
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    population: Sequence[Mapping[str, Any]] | None = None,
    max_new_tokens: int = 160,
    batch_size: int = 16,
    autocast: str = "auto",
    predictions_dir: str | Path | None = None,
    prompt_style: str = "full",
) -> Callable[[nn.Module, int], dict]:
    """A `val_callback` for `playparse.train.loop.train`.

    records       the fixed val subset (see `stratified_subset`)
    population    the full val records, used only for bucket shares
                  (val_exact_match_natural); defaults to `records`
    predictions_dir  if set, each call writes step_<N>.jsonl with raw outputs
    prompt_style  the prompt style the model is trained with (RunSpec.data.prompt_style);
                  validation must decode with the training prompt
    """
    from playparse.train.generate import generate_for_records

    records = [dict(r) for r in records]
    pop = population if population is not None else records
    counts = Counter(str(r["bucket"]) for r in pop)
    shares = {b: c / sum(counts.values()) for b, c in counts.items()}

    def cb(model: nn.Module, step: int) -> dict:
        texts = generate_for_records(model, tokenizer, records, max_new_tokens=max_new_tokens,
                                     batch_size=batch_size, autocast=autocast, prompt_style=prompt_style)
        metrics = score_generations(records, texts, shares)
        metrics["val_gen_sample"] = texts[0] if texts else ""
        if predictions_dir is not None:
            d = Path(predictions_dir)
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"step_{step:07d}.jsonl", "w", encoding="utf-8") as f:
                for r, t in zip(records, texts):
                    f.write(json.dumps({"game_id": r.get("game_id"), "play_id": r.get("play_id"),
                                        "bucket": r["bucket"], "raw": t, "gold": r["label"]}) + "\n")
        return metrics

    return cb


__all__ = ["stratified_subset", "score_generations", "harness_val_callback"]
