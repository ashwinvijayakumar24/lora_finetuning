"""CLI: run one ladder rung over an eval JSONL and write a result artifact.

    python -m playparse.eval.run --rung r0 --data data/processed/eval_lite.jsonl --out results/r0_eval_lite
    python -m playparse.eval.run --rung r1 --data ... --out ... [--limit 200] [--batch-size 8]
    python -m playparse.eval.run --rung r2 --data ... --out ...
    python -m playparse.eval.run --rung r3 --data ... --train data/processed/train.jsonl --out ...
    python -m playparse.eval.run --rung r4 --model <claude model id> --data ... --out ...

Re-running the same command resumes from `<out>/predictions.jsonl`.
"""
from __future__ import annotations

import argparse
import sys
import time

from playparse.eval.harness import format_summary, load_records, run_eval
from playparse.paths import WEIGHTS


def build_predictor(args: argparse.Namespace):
    rung = args.rung
    if rung == "r0":
        from playparse.eval.baselines.regex_parser import RegexPredictor

        return RegexPredictor()
    hf_kw = dict(batch_size=args.batch_size, max_new_tokens=args.max_new_tokens, device=args.device)
    if rung == "r1":
        from playparse.eval.baselines.hf_predictor import make_r1

        return make_r1(args.weights, **hf_kw)
    if rung == "r2":
        from playparse.eval.baselines.hf_predictor import make_r2

        return make_r2(args.weights, **hf_kw)
    if rung == "r3":
        if not args.train:
            sys.exit("--train <train jsonl> is required for r3")
        from playparse.eval.baselines.hf_predictor import make_r3
        from playparse.eval.baselines.retrieval import TfidfRetriever

        retriever = TfidfRetriever.from_jsonl(args.train, max_records=args.index_size, k=args.k)
        return make_r3(args.weights, retriever, **hf_kw)
    if rung == "r4":
        if not args.model:
            sys.exit("--model <claude model id> is required for r4 (model choice is deferred)")
        from playparse.eval.baselines.frontier import make_r4

        return make_r4(args.model)
    sys.exit(f"unknown rung {rung}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rung", required=True, choices=["r0", "r1", "r2", "r3", "r4"])
    ap.add_argument("--data", required=True, help="eval JSONL (one record per play)")
    ap.add_argument("--out", required=True, help="output directory for result.json + predictions.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="only the first N records (smoke runs)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--gpu-usd-per-hour", type=float, default=None, help="price local compute time")
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--train", default=None, help="train JSONL for the r3 retrieval index")
    ap.add_argument("--index-size", type=int, default=None, help="subsample the r3 index")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--model", default=None, help="r4 model id")
    args = ap.parse_args(argv)

    records = load_records(args.data)
    if args.limit:
        records = records[: args.limit]
    predictor = build_predictor(args)
    t0 = time.time()

    def progress(done: int, total: int) -> None:
        rate = done / max(time.time() - t0, 1e-9)
        print(f"\r{done}/{total} plays ({rate:.1f}/s)", end="", file=sys.stderr, flush=True)

    result = run_eval(
        predictor,
        records,
        rung=args.rung,
        out_dir=args.out,
        eval_path=args.data if not args.limit else None,
        n_boot=args.n_boot,
        seed=args.seed,
        resume=not args.no_resume,
        gpu_usd_per_hour=args.gpu_usd_per_hour,
        extra_meta={"limit": args.limit, "data_path": args.data},
        progress=progress,
    )
    print(file=sys.stderr)
    print(format_summary(result))


if __name__ == "__main__":
    main()
