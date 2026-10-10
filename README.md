# llm_finetuning — PlayParse

The third layer of the `custom_llm` stack:

```
llm_inference_engine   → runs Llama 3.2 1B, one request at a time   (built)
llm_serving_layer      → serves it: paged KV, batching, prefix cache (built)
llm_finetuning         → specializes it, then serves many specializations at once (this)
```

**PlayParse** fine-tunes Llama 3.2 1B to read raw NFL play-by-play text and extract
fantasy stat credits as JSON. A deterministic scorer turns those credits into exact
fantasy points. nflverse data provides exact ground truth, so every number here is an
exact match, with no LLM judge involved.

> **Status (2026-10-05):** everything that can run on a laptop is built, tested
> (619 fast tests + 44 slow/GPU), and documented. The full-scale training runs and
> the authoritative serving benchmarks are written and waiting on GPU access. See
> [`docs/BLOCKERS.md`](docs/BLOCKERS.md).

---

## What we've learned so far

1. **A one-day regex scores 99.8% on the 2024 test set.** For well-formatted
   official play text, a regex is the right tool for the common case. The regex's
   errors concentrate in three rare buckets: fumbles (92.5%), laterals (62%), and
   challenges (~92%). See [`docs/benchmarks/r0-real-gt.md`](docs/benchmarks/r0-real-gt.md).
2. **Prompting a 1B model does not work at all; fine-tuning does.** Zero-shot scores
   0.0% (only 24.7% of outputs are even valid JSON). Few-shot scores 16.7% (on a 108-play slice). A LoRA
   adapter trained on 2,400 plays for 2 hours on a laptop scores 93.7% with 100%
   valid output.
3. **The small adapter loses exactly where yards must be computed, not read.** It
   ties the regex on six of nine buckets. It loses on penalties where the play stands
   (−20 points), fumbles (−16), and laterals (−14). On those plays the official yards
   come from a spot on the field (the fumble point, the foul spot), which the text's
   "for N yards" does not state. The full GPU run tests whether 100× more data
   teaches that arithmetic. See [`docs/benchmarks/p3-local-pilot.md`](docs/benchmarks/p3-local-pilot.md).
4. **The adapter does not need the system prompt.** 89% of training tokens were
   prompt, and the 186-token system prompt was most of it. An adapter trained with
   no system message matches the full-prompt adapter (93.2% vs 93.7%, paired
   difference −0.5 points [−1.5, +0.4]) with 60% fewer tokens per example, 2.8×
   faster training steps, and 1.75× faster eval. All GPU configs now use the
   minimal prompt, with one full-prompt control in the sweep. See
   [`docs/benchmarks/p3-prompt-ablation.md`](docs/benchmarks/p3-prompt-ablation.md).
5. **Multi-LoRA batching works on our own serving layer.** Mixed-adapter batches are
   token-identical to running each request alone. The gather-based kernel stays flat
   as the number of distinct adapters grows, while the simple loop grows 10×.

## Claims

Each claim's pass condition was fixed before its run. "Earned (local)" means
measured on the laptop; the authoritative GPU run is still pending.

| | Claim | Status | Evidence |
|---|---|---|---|
| T1 | Fine-tuning beats prompting at 1B | **Earned (pilot)**: 93.7% vs 0.0% (R1); R2 16.7% on a 108-play slice | [p3-local-pilot](docs/benchmarks/p3-local-pilot.md) |
| T2 | A 1B adapter matches the frontier model | Pending: needs API key (B2) | — |
| T3 | ML beats the regex on the hard buckets | **Not earned (frozen test set, 37,859 plays)**: R5 99.54% [99.47, 99.61] vs regex 99.79% [99.75, 99.84]; paired −0.25 [−0.32, −0.19] (R5 alone right on 30 plays, regex alone on 126). R5 wins only penalty_nullified (+0.43). Follow-up T3b (spots, PRD §17) pre-registered and running. | [r5_vs_r0_paired.json](results/test/r5_vs_r0_paired.json) |
| T3b | Spot decomposition beats the regex (pre-registered, PRD §17) | **Not earned**: 99.73% vs regex 99.79% and regex + LOS 99.90% on the frozen test set. But it beats R5 by +0.18 [+0.12, +0.25] and fixes penalty_stands (93.5 → 100) | [t3b-test](docs/benchmarks/t3b-test.md) |
| T4 | QLoRA costs little quality | **Earned (A100)**: 98.1% vs R5 98.5% on eval_lite (−0.4, limit 1.0); ~2% slower steps | [benchmarks](docs/BENCHMARKS.md) |
| T5 | LoRA matches full fine-tuning | **LoRA beat it**: R5 98.5% vs full fine-tune (R7) 97.35% on eval_lite, paired +1.18 [+0.43, +2.12] for LoRA. Caveats: one run each; the full fine-tune's LR (2e-5) was a standard default, not tuned | [benchmarks](docs/BENCHMARKS.md) |
| T6 | Filtering recovers distillation quality | Pending (B2); pipeline built and tested | [P4 pipeline](docs/phases/P4-distill-pipeline.md) |
| T7 | Fine-tuning causes little forgetting | **Earned (A100)**: MMLU base 31.1% vs R5 32.3% (no drop; limit was −2.0), threshold committed before the code | [T7](docs/phases/T7.md) |
| L1 | Merged adapter adds no latency | **Earned (H100)**: merged r=16 −0.07% vs base, inside base-vs-base noise | [p5a CUDA bench](results/p5a/bench_cuda_20261008_224151.json) |
| L2 | Merged equals unmerged | **Earned (local + H100)**: identical greedy tokens on all paths | [p5a-local](docs/benchmarks/p5a-local.md) |
| L3 | Unmerged overhead is small at low rank | **Batch 1: not earned (H100)**, ~30% decode overhead, flat across ranks (launch-bound). **Batch 32: earned for v1** (+11.3% with a shared adapter); v2 +18.5% | [p5a CUDA](results/p5a/bench_cuda_20261008_224151.json), [p5b-cuda](docs/benchmarks/p5b-cuda.md) |
| L4 | Multi-LoRA scales with N | **Not measurable on the pre-registered goodput metric**: our serving layer misses the latency target in every cell (≈60 ms/token vs a 37 ms target; vLLM meets it). Relative scaling holds: v2 keeps 96–98% of throughput from 1 → 256 adapters (vLLM keeps 70–80%) | [p5b-cuda](docs/benchmarks/p5b-cuda.md), [L7](docs/phases/L7.md) |
| L5 | Adapters beat one merged model per tenant | **Measured on the H100**: 31 merged copies vs ~3,300 resident r=16 adapters per 80 GB (pool allocation matched the formula exactly) | [p5b-cuda](docs/benchmarks/p5b-cuda.md) |
| L6 | Prefix cache is adapter-safe | **Earned (local)**, with a fault-injection test that catches the bug | [p5b-local](docs/benchmarks/p5b-local.md) |
| L7 | Competitive with vLLM multi-LoRA | **Not earned (H100)**: ours reaches 9–13% of vLLM 0.31's throughput (threshold 50%); ≈60 ms vs 2.3 ms per token. vLLM loads our PEFT-format adapters unchanged | [L7](docs/phases/L7.md) |

## Docs

| Doc | What it is |
|---|---|
| [`PRD.md`](PRD.md) | The full spec: task, data, eval, ladder, training, serving, claims, phases |
| [`LEARNING_MAP.md`](LEARNING_MAP.md) | Every item on the LoRA learning list → the phase and artifact that teaches it |
| [`docs/phases/`](docs/phases/README.md) | One write-up per phase, written for a learner |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Every benchmark run, with links to the committed artifacts |
| [`docs/ISSUES.md`](docs/ISSUES.md) | Every bug and surprise, with root cause, fix, and guarding test |
| [`docs/BLOCKERS.md`](docs/BLOCKERS.md) | What needs the owner's input, and the GPU jobs ready to run |

## Quick start

```bash
python3 -m venv --system-site-packages .venv && .venv/bin/pip install -e '.[dev]'
scripts/fetch_data.sh                                   # nflverse raw data → data/raw
.venv/bin/python -m playparse.data.build_dataset        # frozen splits → data/processed
.venv/bin/python -m pytest -q                           # fast suite (slow/GPU tests are opt-in)
.venv/bin/python -m playparse.eval.run --rung r0 --data data/processed/test.jsonl --out results/r0_test
.venv/bin/python scripts/train.py --config configs/train_default.yaml --set train.output_dir=runs/r5
.venv/bin/python -m playparse.eval.run --rung lora --adapter runs/r5/best --prompt-style minimal --data data/processed/eval_lite.jsonl --out results/r5
```

`PLAYPARSE_WEIGHTS` points at the HF-format Llama 3.2 1B Instruct weights (default:
`../llm_inference_engine/weights`).
