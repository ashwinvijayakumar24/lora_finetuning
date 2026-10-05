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
| T3 | ML beats the regex on the hard buckets | **Not earned at pilot scale** (−3.9 points overall, paired CI [−5.5, −2.6]). Full-scale run pending (B3). | [p3-local-pilot](docs/benchmarks/p3-local-pilot.md) |
| T4 | QLoRA costs little quality | Pending (B3; needs CUDA) | — |
| T5 | LoRA matches full fine-tuning | Pending (B3); `configs/r7_full_ft.yaml` ready | — |
| T6 | Filtering recovers distillation quality | Pending (B2); pipeline built and tested | [P4 pipeline](docs/phases/P4-distill-pipeline.md) |
| T7 | Fine-tuning causes little forgetting | Pending (after R5) | — |
| L1 | Merged adapter adds no latency | Unresolved locally (below the ±10% noise floor); H100 job ready | [p5a-local](docs/benchmarks/p5a-local.md) |
| L2 | Merged equals unmerged | **Earned (local)**: identical greedy tokens on all five paths | [p5a-local](docs/benchmarks/p5a-local.md) |
| L3 | Unmerged overhead is small at low rank | Unresolved locally; H100 job ready | [p5a-local](docs/benchmarks/p5a-local.md), [p5b-local](docs/benchmarks/p5b-local.md) |
| L4 | Multi-LoRA scales with N | Indicative locally (v2 met the latency target for 100% of requests at N=256 where v1 met it for 0%, single sample); H100 job ready | [p5b-local](docs/benchmarks/p5b-local.md) |
| L5 | Adapters beat one merged model per tenant | Computed from measured bytes: 31 merged copies vs 3,319 r=16 adapters on 80 GB | [p5b-local](docs/benchmarks/p5b-local.md) |
| L6 | Prefix cache is adapter-safe | **Earned (local)**, with a fault-injection test that catches the bug | [p5b-local](docs/benchmarks/p5b-local.md) |
| L7 | Competitive with vLLM multi-LoRA | Pending (B3); script ready | — |

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
