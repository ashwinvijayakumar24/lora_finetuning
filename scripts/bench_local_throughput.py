"""Measure local R1/R2 throughput (plays/sec) on MPS/CPU, plus gold-label token lengths.

Usage:
    python scripts/bench_local_throughput.py --data data/cache/dev_2021_sample.jsonl \
        --n 48 --batch-sizes 1 8 16 --rungs r1 r2 --out results/bench_p1_local_throughput.json
    python scripts/bench_local_throughput.py --label-lengths data/cache/dev_train_all.jsonl

The data must be train-season plays (dev), never the test season.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playparse.eval.harness import hardware_info, load_records, run_eval  # noqa: E402
from playparse.paths import WEIGHTS  # noqa: E402


def label_lengths(path: str) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(WEIGHTS))
    labels = [json.loads(l)["label"] for l in open(path) if l.strip()]
    lens = np.array([len(x) for x in tok(labels, add_special_tokens=False)["input_ids"]])
    return {
        "n": int(lens.size),
        "mean": float(lens.mean()),
        "p50": float(np.percentile(lens, 50)),
        "p99": float(np.percentile(lens, 99)),
        "max": int(lens.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16])
    ap.add_argument("--rungs", nargs="+", default=["r1", "r2"])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--label-lengths", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report: dict = {"hardware": None, "runs": []}
    if args.label_lengths:
        report["label_token_lengths"] = label_lengths(args.label_lengths)
        print("gold label tokens:", report["label_token_lengths"])
    if args.data:
        from playparse.eval.baselines.hf_predictor import make_r1, make_r2

        recs = load_records(args.data)
        # Spread the sample across games and buckets rather than the first N plays.
        rng = np.random.default_rng(0)
        recs = [recs[i] for i in sorted(rng.choice(len(recs), size=min(args.n, len(recs)), replace=False))]
        for rung in args.rungs:
            factory = {"r1": make_r1, "r2": make_r2}[rung]
            pred = factory(WEIGHTS, batch_size=args.batch_sizes[0], max_new_tokens=args.max_new_tokens, device=args.device)
            pred.predict_batch(recs[:2])  # warm-up (MPS kernel compilation)
            for bs in args.batch_sizes:
                pred.batch_size = bs
                t0 = time.perf_counter()
                res = run_eval(pred, recs, rung=rung, n_boot=50)
                wall = time.perf_counter() - t0
                u = res["cost"]["usage_totals"]
                row = {
                    "rung": rung,
                    "batch_size": bs,
                    "n": len(recs),
                    "wall_s": wall,
                    "plays_per_sec": res["latency"]["plays_per_sec"],
                    "p50_batch_latency_s": res["latency"]["p50_s"],
                    "mean_input_tokens": u.get("input_tokens", 0) / len(recs),
                    "mean_output_tokens": u.get("output_tokens", 0) / len(recs),
                    "hit_max_new_tokens": u.get("hit_max_new_tokens", 0),
                    "valid_rate": res["overall"]["valid_rate"]["point"],
                    "exact_match": res["overall"]["exact_match"]["point"],
                    "device": pred.device,
                    "dtype": pred.dtype,
                }
                report["runs"].append(row)
                print(json.dumps(row), flush=True)
                if args.out:  # save after every row so a long run is never all-or-nothing
                    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
            del pred
        report["hardware"] = hardware_info()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
