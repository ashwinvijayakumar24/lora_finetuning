"""CLI: run one ladder rung over an eval JSONL and write a result artifact.

    python -m playparse.eval.run --rung r0 --data data/processed/eval_lite.jsonl --out results/r0_eval_lite
    python -m playparse.eval.run --rung r1 --data ... --out ... [--limit 200] [--batch-size 8]
    python -m playparse.eval.run --rung r2 --data ... --out ...
    python -m playparse.eval.run --rung r3 --data ... --train data/processed/train.jsonl --out ...
    python -m playparse.eval.run --rung r4 --model <claude model id> --data ... --out ...
    python -m playparse.eval.run --rung lora --adapter runs/r5/best --data ... --out ... [--merge]
    python -m playparse.eval.run --rung lora --adapter ... --prompt-style minimal ...  # adapter trained without a system prompt
    python -m playparse.eval.run --rung lora --adapter runs/t3b/best --prompt-style minimal_v2 --schema v2 \
        --data data/processed_v2/eval_lite.jsonl --out ...   # T3b: spots out, converted to v1, scored as v1
    python -m playparse.eval.run --rung r0los --data data/processed_v2/eval_lite.jsonl --out ...  # R0 + line of scrimmage
    python -m playparse.eval.run --rung lora --adapter runs/r6/best --prompt-style minimal ...  # QLoRA: 4-bit base (as trained)
    python -m playparse.eval.run --rung lora --adapter runs/r6/best --base-quant none ...       # same adapter, bf16 base

Re-running the same command resumes from `<out>/predictions.jsonl`.
"""
from __future__ import annotations

import argparse
import sys
import time

from playparse.eval.harness import format_summary, load_records, run_eval
from playparse.paths import WEIGHTS
from playparse.prompt import PROMPT_STYLES, SCHEMAS, check_style_schema


def build_predictor(args: argparse.Namespace):
    rung = args.rung
    if rung == "r0":
        from playparse.eval.baselines.regex_parser import RegexPredictor

        return RegexPredictor()
    if rung == "r0los":
        from playparse.eval.baselines.regex_los import RegexLOSPredictor

        return RegexLOSPredictor()
    hf_kw = dict(batch_size=args.batch_size, max_new_tokens=args.max_new_tokens, device=args.device,
                 dtype=args.dtype, prompt_style=args.prompt_style,
                 # "auto" means "as the adapter was trained", which only the lora rung can know.
                 base_quant=args.base_quant if rung == "lora" else
                 (None if args.base_quant == "auto" else args.base_quant))
    if rung == "lora":
        if not args.adapter:
            sys.exit("--adapter <PEFT adapter dir> is required for --rung lora")
        from playparse.eval.baselines.lora_predictor import make_lora

        return make_lora(args.weights, args.adapter, merge=args.merge, **hf_kw)
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

        if args.prompt_style != "full":
            sys.exit("--prompt-style applies to the local HF rungs (r1-r3, lora), not r4")
        if args.provider == "openai":
            from playparse.eval.baselines.frontier_openai import make_r4_openai

            return make_r4_openai(args.model, reasoning_effort=args.reasoning_effort, concurrency=args.concurrency)
        return make_r4(args.model)
    sys.exit(f"unknown rung {rung}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rung", required=True, choices=["r0", "r0los", "r1", "r2", "r3", "r4", "lora"])
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
    ap.add_argument("--dtype", default=None, choices=["float32", "float16", "bfloat16"],
                    help="model dtype for HF rungs (default: fp32 on cpu, bf16 on cuda, fp16 on mps)")
    ap.add_argument("--adapter", default=None, help="PEFT-format adapter directory for --rung lora")
    ap.add_argument("--merge", action="store_true", help="--rung lora: merge the adapter into the base first")
    ap.add_argument("--base-quant", default="auto", choices=["auto", "none", "nf4"],
                    help="HF rungs: base model quantization. 'auto' (default) = as the adapter was trained "
                         "(its run_spec.json model.quant; none for other rungs); 'nf4' = 4-bit NF4 via "
                         "bitsandbytes (CUDA only, as QLoRA/R6 trains); 'none' = the base in --dtype")
    ap.add_argument("--prompt-style", default="full", choices=list(PROMPT_STYLES),
                    help="HF rungs: 'full' = the shared system prompt (default); 'minimal' = no system message; "
                         "'minimal_v2' = minimal plus the line of scrimmage, output schema v2 (T3b). "
                         "Must match the style a LoRA adapter was trained with (data.prompt_style)")
    ap.add_argument("--schema", default=None, choices=list(SCHEMAS),
                    help="output schema the model emits (default: the prompt style's). v2 outputs (field spots) "
                         "are converted to v1 before scoring, so every metric keeps its v1 definition; "
                         "an output that is not valid v2 counts as invalid. Needs a v2 data file (with 'los')")
    ap.add_argument("--train", default=None, help="train JSONL for the r3 retrieval index")
    ap.add_argument("--index-size", type=int, default=None, help="subsample the r3 index")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--model", default=None, help="r4 model id")
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"], help="r4 API provider")
    ap.add_argument("--reasoning-effort", default="low", help="r4 with --provider openai: reasoning effort")
    ap.add_argument("--concurrency", type=int, default=1, help="r4 with --provider openai: parallel requests")
    args = ap.parse_args(argv)

    try:
        schema = check_style_schema(args.prompt_style, args.schema)
    except ValueError as e:
        sys.exit(str(e))
    records = load_records(args.data)
    if args.limit:
        records = records[: args.limit]
    if (schema == "v2" or args.rung == "r0los") and not all("los" in r for r in records):
        sys.exit(f"{args.data} has no 'los' field; schema v2 and r0los need a v2 data file "
                 "(python -m playparse.data.build_dataset_v2 writes data/processed_v2/)")
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
        extra_meta={"limit": args.limit, "data_path": args.data,
                    **({"adapter_path": str(args.adapter)} if args.adapter else {})},
        progress=progress,
    )
    print(file=sys.stderr)
    print(format_summary(result))


if __name__ == "__main__":
    main()
