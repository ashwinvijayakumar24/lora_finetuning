#!/usr/bin/env python
"""P4 pricing pilot: label ~1k training plays with the teacher and measure what it costs.

Blocked on B2 (needs an API credential) for the real run; ``--dry-run`` works today.

Dry run (no API calls, no key)::

    python scripts/p4_pilot.py --dry-run
    python scripts/p4_pilot.py --dry-run --data data/processed/train.jsonl --few-shot-file prompts/r4_fewshot.txt

    Estimates tokens with the rough rule tokens ~= characters / 4 and prices the pilot
    and the full 100k x k run for a few candidate teacher models.

Real run (spends money; capped in code)::

    export ANTHROPIC_API_KEY=...
    python scripts/p4_pilot.py --model claude-opus-5 --budget-usd 25 \\
        --data data/processed/train.jsonl --few-shot-file prompts/r4_fewshot.txt

    1. Picks N plays from the train split (seeded, same order as the distillation curve).
    2. Collects k samples per play into runs/p4_pilot/<model>/cache (resumable: rerun
       the same command after a crash or a budget stop).
    3. Builds the R8 and R9 files and scores both against ground truth.
    4. Prints measured $ per 1k plays, filter keep rate, and precision, and writes
       runs/p4_pilot/<model>/pilot_report.json. Copy the numbers into BENCHMARKS.md.

The teacher client defaults to a minimal Anthropic SDK client defined below. Once the
project's frontier client exists, pass it with ``--client package.module:factory``;
the factory is called as ``factory(model=..., system=..., max_tokens=...)`` and must
return a ``playparse.distill.TeacherClient``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from playparse.distill import (  # noqa: E402
    Budget,
    TeacherBatch,
    Usage,
    build_training_sets,
    label_dataset,
    precision_report,
    subset_order,
)
from playparse.paths import DATA_PROCESSED  # noqa: E402
from playparse.prompt import SYSTEM_PROMPT, render_user  # noqa: E402

# USD per million tokens: (input, output). Anthropic first-party list prices as cached
# in the claude-api reference on 2026-06-24; verify on the pricing page before a real
# run. Cache reads are billed at ~0.1x input, cache writes at ~1.25x input.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
CACHE_READ_MULT = 0.10
CACHE_WRITE_MULT = 1.25
TYPICAL_DESC_CHARS = 180      # used only when no data file is available
TYPICAL_LABEL_CHARS = 170
DEFAULT_FEW_SHOT_TOKENS = 1500  # 8 examples x (desc + label) when no few-shot file is given


def est_tokens(text: str) -> int:
    """Rough token count: about 4 characters per token for English and JSON."""
    return max(1, round(len(text) / 4))


def load_records(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def per_call_cost(model: str, prefix_tok: float, play_tok: float, out_tok: float, cached: bool) -> float:
    p_in, p_out = PRICES[model]
    prefix_mult = CACHE_READ_MULT if cached else 1.0
    return (prefix_tok * prefix_mult * p_in + play_tok * p_in + out_tok * p_out) / 1e6


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    data = Path(args.data)
    system = SYSTEM_PROMPT
    few_shot_tok = (est_tokens(Path(args.few_shot_file).read_text(encoding="utf-8"))
                    if args.few_shot_file else DEFAULT_FEW_SHOT_TOKENS)
    if data.exists():
        recs = subset_order(load_records(data), args.seed)[: args.n]
        play_tok = sum(est_tokens(render_user(r.get("posteam"), r["desc"])) for r in recs) / len(recs)
        # Label length is used only to size the expected output; nothing is filtered.
        out_tok = sum(est_tokens(r["label"]) for r in recs if "label" in r) / max(1, sum("label" in r for r in recs))
        source = f"{len(recs)} plays from {data}"
    else:
        play_tok = est_tokens("posteam: PHI\ndesc: " + "x" * TYPICAL_DESC_CHARS)
        out_tok = est_tokens("x" * TYPICAL_LABEL_CHARS)
        source = f"typical lengths ({data} not found)"
    out_tok += args.thinking_tokens
    prefix_tok = est_tokens(system) + few_shot_tok
    calls_pilot = args.n * args.k
    calls_full = args.full_plays * args.k

    rows = []
    for model in args.models:
        c_cached = per_call_cost(model, prefix_tok, play_tok, out_tok, cached=True)
        c_uncached = per_call_cost(model, prefix_tok, play_tok, out_tok, cached=False)
        rows.append({
            "model": model,
            "usd_per_call_cached": c_cached,
            "usd_per_call_uncached": c_uncached,
            "pilot_usd_cached": c_cached * calls_pilot,
            "pilot_usd_uncached": c_uncached * calls_pilot,
            "full_usd_cached": c_cached * calls_full,
            "usd_per_1k_plays_cached": c_cached * 1000 * args.k,
        })
    est = {
        "mode": "dry_run",
        "token_rule": "chars/4",
        "source": source,
        "prefix_tokens": prefix_tok,
        "play_tokens_avg": play_tok,
        "output_tokens_avg": out_tok,
        "k": args.k,
        "pilot_plays": args.n,
        "full_plays": args.full_plays,
        "rows": rows,
    }
    print(f"Token estimate (chars/4) from {source}:")
    print(f"  shared prefix (system + few-shot): {prefix_tok:.0f} tokens (prompt-cached after the first call)")
    print(f"  per play input: {play_tok:.0f}   per sample output: {out_tok:.0f}"
          f"{f' (incl. {args.thinking_tokens} thinking)' if args.thinking_tokens else ''}")
    print(f"\n{'model':<18} {'$/call':>9} {'$/1k plays':>11} {'pilot $':>9} {'pilot $ no-cache':>17} "
          f"{'full ' + str(args.full_plays // 1000) + 'k $':>11}")
    for r in rows:
        print(f"{r['model']:<18} {r['usd_per_call_cached']:>9.5f} {r['usd_per_1k_plays_cached']:>11.2f} "
              f"{r['pilot_usd_cached']:>9.2f} {r['pilot_usd_uncached']:>17.2f} {r['full_usd_cached']:>11.2f}")
    print(f"\n(k = {args.k} samples per play; $/1k plays includes all k samples. Prices are cached list "
          "prices; verify before spending. Adaptive thinking bills as output: pass --thinking-tokens.)")
    return est


class AnthropicTeacher:
    """Minimal TeacherClient over the Anthropic Messages API.

    One request per sample, with the shared system prompt + few-shot prefix marked for
    prompt caching. Samples differ because sampling is stochastic by default. Effort is
    set low: this is a short extraction task, and thinking tokens bill as output.
    ``sdk_client`` is injectable for tests.
    """

    def __init__(self, model: str, system: str, max_tokens: int = 1024, effort: str | None = "low",
                 sdk_client: Any = None):
        if model not in PRICES:
            raise SystemExit(f"no price for {model}; add it to PRICES so the budget cap can work")
        if sdk_client is None:
            import anthropic  # optional dependency: pip install -e '.[teacher]'
            sdk_client = anthropic.Anthropic()
        self.client = sdk_client
        self.model = model
        self.system = system
        self.max_tokens = max_tokens
        # Haiku 4.5 rejects the effort parameter.
        self.effort = None if model.startswith("claude-haiku") else effort

    def _one(self, record: Mapping[str, Any]) -> tuple[str, Usage]:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": render_user(record.get("posteam"), record["desc"])}],
        )
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        resp = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        u = resp.usage
        p_in, p_out = PRICES[self.model]
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        cost = (u.input_tokens * p_in + cache_read * CACHE_READ_MULT * p_in
                + cache_write * CACHE_WRITE_MULT * p_in + u.output_tokens * p_out) / 1e6
        return text, Usage(calls=1, input_tokens=u.input_tokens + cache_read + cache_write,
                           output_tokens=u.output_tokens, cache_read_tokens=cache_read, cost_usd=cost)

    def __call__(self, records: Sequence[Mapping[str, Any]], n_samples: int) -> TeacherBatch:
        outputs, total = [], Usage()
        for r in records:
            samples = []
            for _ in range(n_samples):
                text, u = self._one(r)
                samples.append(text)
                total = total + u
            outputs.append(samples)
        return TeacherBatch(outputs, total)


def make_client(args: argparse.Namespace, system: str):
    if args.client:
        mod, _, attr = args.client.partition(":")
        factory = getattr(importlib.import_module(mod), attr)
        return factory(model=args.model, system=system, max_tokens=args.max_tokens)
    return AnthropicTeacher(args.model, system, max_tokens=args.max_tokens)


def real_run(args: argparse.Namespace) -> dict[str, Any]:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or args.client):
        raise SystemExit("blocked on B2: set ANTHROPIC_API_KEY (see docs/BLOCKERS.md), or use --dry-run")
    data = Path(args.data)
    if not data.exists():
        raise SystemExit(f"{data} not found; build the dataset first")
    system = SYSTEM_PROMPT
    if args.few_shot_file:
        system = SYSTEM_PROMPT + "\n\n" + Path(args.few_shot_file).read_text(encoding="utf-8")
    else:
        print("warning: no --few-shot-file; the teacher gets the zero-shot prompt, not R4's", file=sys.stderr)

    records = subset_order(load_records(data), args.seed)[: args.n]
    run_dir = Path(args.out) / args.model
    teacher_id = f"{args.model}|sha256:{hashlib.sha256(system.encode()).hexdigest()[:16]}"
    client = make_client(args, system)
    rep = label_dataset(
        [{k: v for k, v in r.items() if k != "label"} for r in records],
        client, run_dir / "cache", n_samples=args.k, chunk_size=args.chunk_size, teacher_id=teacher_id,
        budget=Budget(max_calls=args.n * args.k, max_usd=args.budget_usd, est_usd_per_call=args.est_usd_per_call),
        on_chunk=lambda r: print(f"  labeled {r.n_already_cached + r.n_labeled}/{r.n_records}  "
                                 f"spent ${r.usage_total.cost_usd:.2f}", file=sys.stderr),
    )
    build = build_training_sets(records, run_dir / "cache", run_dir / "sets", k=args.k, sizes=(),
                                teacher_id=teacher_id)
    p8 = precision_report(build.r8_path, records)
    p9 = precision_report(build.r9_path, records)
    done = rep.n_already_cached + rep.n_labeled
    report = {
        "mode": "real",
        "model": args.model,
        "teacher_id": teacher_id,
        "plays_requested": args.n,
        "plays_labeled": done,
        "stopped_reason": rep.stopped_reason,
        "usage_total": rep.usage_total.__dict__,
        "usd_per_1k_plays": (rep.usage_total.cost_usd / done * 1000) if done else None,
        "projected_full_usd": (rep.usage_total.cost_usd / done * args.full_plays) if done else None,
        "r8": {"n": build.n_r8, "precision": p8},
        "r9": {"n": build.n_r9, "keep_rate": build.r9_stats["keep_rate"], "precision": p9,
               "drop_stats": build.r9_stats},
    }
    (run_dir / "pilot_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("plays_labeled", "stopped_reason", "usd_per_1k_plays",
                                             "projected_full_usd")}, indent=2))
    print(f"R8 precision {p8['precision']}, R9 keep rate {report['r9']['keep_rate']:.3f}, "
          f"R9 precision {p9['precision']}")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="P4 teacher pricing pilot")
    p.add_argument("--dry-run", action="store_true", help="estimate tokens and cost; no API calls")
    p.add_argument("--data", default=str(DATA_PROCESSED / "train.jsonl"))
    p.add_argument("--few-shot-file", help="R4 few-shot prefix text (appended to the system prompt)")
    p.add_argument("--n", type=int, default=1000, help="pilot plays")
    p.add_argument("--k", type=int, default=3, help="samples per play")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--full-plays", type=int, default=100_000, help="for the full-run projection")
    p.add_argument("--models", nargs="+", default=list(PRICES), help="dry run: models to price")
    p.add_argument("--thinking-tokens", type=int, default=0, help="dry run: extra output tokens per sample")
    p.add_argument("--model", default="claude-opus-5", help="real run: teacher model")
    p.add_argument("--client", help="real run: 'module:factory' returning a TeacherClient")
    p.add_argument("--budget-usd", type=float, help="real run: hard cap in dollars (required)")
    p.add_argument("--est-usd-per-call", type=float,
                   help="real run: prior per-call cost for the first chunk (default: dry-run estimate)")
    p.add_argument("--chunk-size", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--out", default=str(REPO / "runs" / "p4_pilot"))
    p.add_argument("--json-out", help="also write the dry-run estimate here")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        est = dry_run(args)
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(est, indent=2))
        return 0
    if args.budget_usd is None:
        raise SystemExit("--budget-usd is required for a real run (the cap is enforced in code)")
    if args.est_usd_per_call is None:
        dry = dry_run(argparse.Namespace(**{**vars(args), "models": [args.model]}))
        # Assume no cache hits for the prior: the first chunk writes the cache.
        args.est_usd_per_call = dry["rows"][0]["usd_per_call_uncached"]
    real_run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
