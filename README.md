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

> **Status (2026-10-10): complete.** Every experiment in the PRD has run on PACE GPUs
> (A100 / L40S / H100) except the two that need an Anthropic API key (T2, T6). 751 fast
> tests + 52 slow/GPU tests. Every claim below had its pass condition written down
> before its run, and every number links to a committed artifact.

---

## What we learned

1. **Fine-tuning is what makes a 1B model usable.** Prompted, Llama 3.2 1B scores 0.0%
   (zero-shot) and 16.7% (few-shot, 108-play slice). LoRA-tuned on 294k plays it scores
   **99.54%** on the frozen 37,859-play test set, with 100% valid JSON.
2. **For clean, fixed-format text, a careful regex still wins, narrowly.** A one-day
   regex scores 99.79% (99.90% when given the line of scrimmage). The best adapter
   (spot outputs, T3b) reaches 99.73%. Both systems are perfect on ~96% of plays; the
   regex's edge is on fumbles and laterals, where official yardage rules are subtle.
   See [`t3b-test`](docs/benchmarks/t3b-test.md).
3. **Splitting reading from arithmetic helps, as designed.** Letting the model output
   field spots and code compute yards (pre-registered T3b) beat plain LoRA by +0.18
   points and took penalties-where-the-play-stands from 93.5% to 100%. What remains is
   choosing *which* printed spot counts, a reading problem.
4. **LoRA's knobs barely matter here; data does.** 23 one-knob variants at 50k plays
   land within 0.4 points of each other, and rank 2 (≈1.4M parameters) equals rank 64
   (≈45M). Data size moves accuracy from 84.6% (1k plays) to 98.5% (294k). LoRA beat
   a full fine-tune (98.5% vs 97.35%), QLoRA's 4-bit base cost 0.4 points, and
   fine-tuning did not reduce MMLU accuracy. See [`p3-sweep`](docs/benchmarks/p3-sweep.md).
5. **The adapter does not need its system prompt.** Dropping it matched quality and
   halved training compute (60% fewer tokens). See
   [`p3-prompt-ablation`](docs/benchmarks/p3-prompt-ablation.md).
6. **Serving: correct and scalable, but not fast.** On our own stack, merged adapters
   are free (L1), merged = unmerged (L2), mixed-adapter batches equal solo runs, and the
   prefix cache never crosses adapters (L6). The gather kernel holds 96–98% of
   throughput from 1 to 256 adapters. But vLLM is ~11× faster in absolute throughput
   (≈2.3 vs ≈60 ms per token): Python-loop attention, no CUDA graphs, and unfused LoRA
   kernels. See [`L7`](docs/phases/L7.md) and [`p5b-cuda`](docs/benchmarks/p5b-cuda.md).
7. **Gate releases per bucket, not on the average.** A deliberately broken adapter
   cleared a 95% overall floor and was still refused for 8–12-point drops on hard
   buckets. See [`P6-gate-demo`](docs/phases/P6-gate-demo.md).

## Claims

Each claim's pass condition was fixed before its run. Not-earned results are published
with the same detail as earned ones.

| | Claim | Status | Evidence |
|---|---|---|---|
| T1 | Fine-tuning beats prompting at 1B | **Earned**: 99.54% on the frozen test set (R5) vs 0.0% zero-shot; few-shot 16.7% on a 108-play slice | [T3-test](docs/BENCHMARKS.md), [p3-local-pilot](docs/benchmarks/p3-local-pilot.md) |
| T2 | A 1B adapter matches the frontier model | Pending: needs API key (B2) | — |
| T3 | ML beats the regex on the hard buckets | **Not earned (frozen test set, 37,859 plays)**: R5 99.54% [99.47, 99.61] vs regex 99.79% [99.75, 99.84]; paired −0.25 [−0.32, −0.19] (R5 alone right on 30 plays, regex alone on 126). R5 wins only penalty_nullified (+0.43). Follow-up: T3b. | [r5_vs_r0_paired.json](results/test/r5_vs_r0_paired.json) |
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
