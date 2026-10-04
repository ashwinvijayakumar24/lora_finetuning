#!/usr/bin/env python
"""P3 local pilot: a reduced-scale LoRA run on the M4, scored against R0 and R1.

Stages (each is resumable; rerun the same command after an interruption):

    python scripts/p3_pilot.py data        # build seeded subsets + manifest + token stats
    python scripts/train.py --config configs/p3_local_pilot.yaml [--resume latest]
    python -m playparse.eval.run --rung lora --adapter runs/p3_local_pilot/train/checkpoints/step_0000300/adapter \
        --data runs/p3_local_pilot/data/eval_pilot.jsonl --out runs/p3_local_pilot/eval/lora --dtype bfloat16
    python -m playparse.eval.run --rung r0 --data runs/p3_local_pilot/data/eval_pilot.jsonl --out runs/p3_local_pilot/eval/r0
    python -m playparse.eval.run --rung r1 --data runs/p3_local_pilot/data/eval_pilot.jsonl --out runs/p3_local_pilot/eval/r1 --dtype bfloat16
    python scripts/p3_pilot.py report      # comparison table, paired CIs, copies artifacts to results/

Data design (written to runs/p3_local_pilot/data/, which is git-ignored; the
manifest with every key and file hash goes to results/p3_local_pilot/):

* train_pilot.jsonl: from train.jsonl (seasons 2015-2022). ALL laterals, then an
  equal share of the remaining hard budget for each of the other seven non-normal
  buckets, and the same number of normal plays as hard plays. This deliberately
  over-samples rare buckets (fumble is 1.4% of train but ~6% here), so the pilot's
  training distribution is NOT the natural one.
* Val subsets are drawn inside scripts/train.py from the full val.jsonl (2023):
  a proportional sample for val loss (data.val_loss_examples) and a bucket-balanced
  sample for generation eval (gen_eval_examples). Their keys are in run_meta.json.
* eval_pilot.jsonl: from eval_lite.jsonl (2024, the designated local eval subset):
  every non-normal play plus a seeded sample of normal plays.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from playparse.paths import DATA_PROCESSED, WEIGHTS  # noqa: E402

RUN = REPO / "runs" / "p3_local_pilot"
DATA = RUN / "data"
RESULTS = REPO / "results" / "p3_local_pilot"
HARD = ["fumble", "lateral", "challenge", "penalty_stands", "two_point", "interception", "td", "penalty_nullified"]
BUCKETS = HARD + ["normal"]


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def by_bucket(recs: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        out[r["bucket"]].append(r)
    return out


def enriched_train(train: list[dict], n_hard: int, seed: int) -> list[dict]:
    """All laterals, equal shares for the other hard buckets, then as many normals."""
    rng = random.Random(seed)
    bb = by_bucket(train)
    chosen = list(bb["lateral"])
    others = [b for b in HARD if b != "lateral"]
    remaining = n_hard - len(chosen)
    # Water-filling: a bucket smaller than its fair share gives everything it has,
    # and the shortfall is shared by the rest; then equal shares in HARD order.
    alloc: dict[str, int] = {}
    open_b = list(others)
    while True:
        small = [b for b in open_b if len(bb[b]) < remaining / len(open_b)]
        if not small:
            break
        for b in small:
            alloc[b] = len(bb[b])
            remaining -= alloc[b]
            open_b.remove(b)
    for i, b in enumerate(open_b):
        alloc[b] = remaining // (len(open_b) - i)
        remaining -= alloc[b]
    for b in others:
        chosen += rng.sample(bb[b], alloc[b])
    chosen += rng.sample(bb["normal"], len(chosen))
    rng.shuffle(chosen)
    return chosen


def token_stats(recs: list[dict]) -> dict:
    """Prompt vs completion tokens per example (the system prompt is a fixed cost)."""
    from transformers import AutoTokenizer

    from playparse.prompt import CHAT_DATE_STRING, SYSTEM_PROMPT
    from playparse.train.collate import encode_record

    tok = AutoTokenizer.from_pretrained(str(WEIGHTS))
    prompt, comp, total, padded = [], [], [], []
    for r in recs:
        ex = encode_record(r, tok)
        prompt.append(ex.n_prompt)
        comp.append(ex.n_completion)
        total.append(len(ex))
        padded.append(-(-len(ex) // 64) * 64)
    # Fixed part: everything before the play-specific user turn content.
    empty = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ""}],
        tokenize=False, add_generation_prompt=True, date_string=CHAT_DATE_STRING)
    fixed = len(tok(empty, add_special_tokens=False)["input_ids"])
    system_only = len(tok(SYSTEM_PROMPT, add_special_tokens=False)["input_ids"])

    def summ(xs: list[int]) -> dict:
        s = sorted(xs)
        return {"mean": round(statistics.mean(s), 1), "p50": s[len(s) // 2], "p99": s[int(0.99 * (len(s) - 1))],
                "max": s[-1], "sum": sum(s)}

    return {
        "n": len(recs),
        "prompt_tokens": summ(prompt),
        "completion_tokens": summ(comp),
        "total_tokens": summ(total),
        "padded_to_64_tokens": summ(padded),
        "fixed_template_tokens": fixed,
        "system_prompt_tokens": system_only,
        "prompt_share": round(sum(prompt) / sum(total), 4),
        "completion_share": round(sum(comp) / sum(total), 4),
        "padding_overhead": round(sum(padded) / sum(total) - 1, 4),
    }


def cmd_data(args: argparse.Namespace) -> None:
    src = Path(args.processed)
    train = read_jsonl(src / "train.jsonl")
    lite = read_jsonl(src / "eval_lite.jsonl")

    tr = enriched_train(train, args.n_hard, args.seed)
    rng = random.Random(args.seed + 2)
    lite_bb = by_bucket(lite)
    ev = [r for r in lite if r["bucket"] != "normal"] + sorted(
        rng.sample(lite_bb["normal"], args.n_eval_normal), key=lambda r: (r["game_id"], r["play_id"]))
    ev.sort(key=lambda r: (r["game_id"], r["play_id"]))

    files = {"train_pilot.jsonl": tr, "eval_pilot.jsonl": ev}
    for name, recs in files.items():
        write_jsonl(DATA / name, recs)

    manifest = {
        "purpose": "P3 local pilot subsets (reduced scale, Apple M4)",
        "seed": args.seed,
        "sources": {n: {"path": str(src / n), "sha256": sha256(src / n)}
                    for n in ("train.jsonl", "eval_lite.jsonl")},
        "params": {"n_hard": args.n_hard, "n_eval_normal": args.n_eval_normal},
        "design": {
            "train_pilot": "all train laterals + equal share of the remaining hard budget per other non-normal "
                           "bucket + an equal number of normal plays (enriched; NOT the natural distribution)",
            "eval_pilot": "all non-normal eval_lite plays + seeded sample of normal eval_lite plays (2024)",
        },
        "files": {},
    }
    natural = Counter(r["bucket"] for r in train)
    for name, recs in files.items():
        c = Counter(r["bucket"] for r in recs)
        manifest["files"][name] = {
            "n": len(recs),
            "sha256": sha256(DATA / name),
            "by_bucket": {b: c.get(b, 0) for b in BUCKETS},
            "keys": [f"{r['game_id']}#{r['play_id']}" for r in recs],
        }
        if name == "train_pilot.jsonl":
            manifest["files"][name]["share_vs_natural_train"] = {
                b: {"pilot": round(c.get(b, 0) / len(recs), 4), "train": round(natural[b] / len(train), 4)}
                for b in BUCKETS}
            manifest["files"][name]["seasons"] = dict(sorted(Counter(r["season"] for r in recs).items()))
        if name == "eval_pilot.jsonl":
            manifest["files"][name]["n_games"] = len({r["game_id"] for r in recs})
            manifest["files"][name]["by_part"] = dict(Counter(r.get("eval_lite_part") for r in recs))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "data_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    stats = {"train_pilot": token_stats(tr), "eval_pilot": token_stats(ev)}
    (RESULTS / "token_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    for name, m in manifest["files"].items():
        print(name, m["n"], m["by_bucket"])
    print(json.dumps(stats, indent=2))


# --------------------------------------------------------------------------- report


def _load_preds(d: Path) -> dict[str, dict]:
    return {r["key"]: r for r in read_jsonl(d / "predictions.jsonl")}


def paired_bucket_diffs(a_dir: Path, b_dir: Path, n_boot: int = 2000, seed: int = 0) -> dict:
    """Exact-match difference A - B per bucket, with a paired game-cluster bootstrap CI."""
    import numpy as np

    pa, pb = _load_preds(a_dir), _load_preds(b_dir)
    keys = sorted(set(pa) & set(pb))
    rows = defaultdict(list)  # bucket -> [(game, diff)]
    for k in keys:
        ra, rb = pa[k], pb[k]
        d = int(ra["exact"]) - int(rb["exact"])
        rows[ra["bucket"]].append((ra["game_id"], d))
        rows["OVERALL"].append((ra["game_id"], d))
    out = {}
    rng = np.random.default_rng(seed)
    for b, items in rows.items():
        games = sorted({g for g, _ in items})
        gi = {g: i for i, g in enumerate(games)}
        sums = np.zeros(len(games))
        cnts = np.zeros(len(games))
        for g, d in items:
            sums[gi[g]] += d
            cnts[gi[g]] += 1
        w = rng.multinomial(len(games), [1 / len(games)] * len(games), size=n_boot)
        boots = (w @ sums) / np.maximum(w @ cnts, 1)
        out[b] = {"n": len(items), "n_games": len(games), "diff": float(sums.sum() / cnts.sum()),
                  "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5))}
    return out


def cmd_report(args: argparse.Namespace) -> None:
    eval_root = RUN / "eval"
    rungs = [d.name for d in sorted(eval_root.iterdir()) if (d / "result.json").exists()]
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary: dict = {"rungs": {}, "paired": {}}
    for r in rungs:
        res = json.loads((eval_root / r / "result.json").read_text())
        summary["rungs"][r] = {
            "overall": res["overall"], "buckets": res["buckets"], "latency": res["latency"],
            "config_hash": res["meta"]["config_hash"], "config": res["meta"]["config"],
        }
        dst = RESULTS / "eval" / r
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(eval_root / r / "result.json", dst / "result.json")
        src_pred = eval_root / r / "predictions.jsonl"
        if src_pred.stat().st_size > 5 * 2**20:
            with open(src_pred, "rb") as fi, gzip.open(dst / "predictions.jsonl.gz", "wb") as fo:
                shutil.copyfileobj(fi, fo)
        else:
            shutil.copy2(src_pred, dst / "predictions.jsonl")
    for a in [r for r in rungs if r.startswith("lora")]:
        for b in [r for r in rungs if not r.startswith("lora")]:
            summary["paired"][f"{a}-minus-{b}"] = paired_bucket_diffs(eval_root / a, eval_root / b)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")

    def cell(ci: dict) -> str:
        if ci["point"] != ci["point"]:
            return "-"
        return f"{100 * ci['point']:.1f} [{100 * ci['lo']:.1f}, {100 * ci['hi']:.1f}]"

    names = ["OVERALL"] + BUCKETS
    print("| bucket | n | " + " | ".join(rungs) + " |")
    print("|---|---|" + "---|" * len(rungs))
    for b in names:
        blocks = [summary["rungs"][r]["overall" if b == "OVERALL" else "buckets"] for r in rungs]
        blocks = [x if b == "OVERALL" else x.get(b) for x in blocks]
        if any(x is None for x in blocks):
            continue
        print(f"| {b} | {blocks[0]['n']} | " + " | ".join(cell(x["exact_match"]) for x in blocks) + " |")
    for k, v in summary["paired"].items():
        print(f"\npaired exact-match difference {k} (pp, 95% game-cluster bootstrap)")
        for b in names:
            if b in v:
                d = v[b]
                print(f"  {b:<18} n={d['n']:>4}  {100 * d['diff']:+6.1f} [{100 * d['lo']:+6.1f}, {100 * d['hi']:+6.1f}]")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--processed", default=str(DATA_PROCESSED))
    d.add_argument("--seed", type=int, default=20263)
    d.add_argument("--n-hard", type=int, default=1200)
    d.add_argument("--n-eval-normal", type=int, default=200)
    d.set_defaults(fn=cmd_data)
    r = sub.add_parser("report")
    r.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
