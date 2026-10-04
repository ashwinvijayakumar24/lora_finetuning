# P5a — LoRA adapters in `llm_inference_engine` (merged and unmerged)

**Status:** built and verified locally (Apple M4). The authoritative latency
numbers for claims L1 and L3 still need one H100 run; the Slurm script is ready.

| Claim | Local result | Status |
|---|---|---|
| L1 merged adds no latency | merged inside the base-vs-base noise band (±10% on a contended laptop); not resolvable locally | pending H100 |
| L2 merged equals unmerged | identical 16 greedy tokens merged vs unmerged (fp32 and MPS fp16); fp32 logits within 2.2e-4 of HF+PEFT | earned locally (re-checked on CUDA by the same tests) |
| L3 unmerged overhead small at low rank | batch 1: unmerged 0–15% slower end to end, same size as noise; linear-only microbench +2–6% | pending H100; batch-32 arm moves to P5b |

## 1. What was built

All code lives in `playparse/serving/`. Neither `llm_inference_engine` nor
`llm_serving_layer` was edited.

| File | Role |
|---|---|
| `_engine_path.py` | Finds the engine checkout (`PLAYPARSE_ENGINE_DIR`, else the nearest sibling `llm_inference_engine`) and puts it on `sys.path`. |
| `adapter.py` | Loads a PEFT adapter into `LoRAAdapter`: per engine weight name, `A`, `B` and `scale`. Name mapping and validation. |
| `merge.py` | Merged mode: `merge_adapter`, plus `merge_then_quantize` for int8/int4 bases. |
| `lora_engine.py` | Unmerged mode: `LoRALinear`, the `linear()` replacement, `AdapterSelection`, `LoRAModelGPU`, `LoRALlamaModel`, `generate_with_adapter`. |
| `_testing.py` | Builds tiny HF checkpoints and PEFT adapters (with random non-zero `B`) for tests. |

### Public API

```python
from playparse.serving.adapter import load_peft_adapter
from playparse.serving.merge import merge_adapter, merge_then_quantize
from playparse.serving.lora_engine import LoRAModelGPU, AdapterSelection, generate_with_adapter
from engine import LlamaModelGPU, load_config, load_weights_gpu, get_sampler

cfg     = load_config(weights_dir)
weights = load_weights_gpu(weights_dir, cfg, device="cuda:0")
adapter = load_peft_adapter("runs/playparse-r16/adapter", base_config=cfg)

# Merged: the stock engine, different numbers in the same tensors.
merged = LlamaModelGPU(merge_adapter(weights, adapter), cfg)

# Unmerged: one base model, adapters chosen per call.
model = LoRAModelGPU(weights, cfg)
model.load_adapter(adapter, "playparse-v1")
logits = model.prefill(ids, cache, adapter="playparse-v1")   # or adapter=None for base
for tok in generate_with_adapter(model, ids, get_sampler(temp=0.0), "playparse-v1"):
    ...
```

## 2. Loading a PEFT adapter

PEFT saves two files. `adapter_config.json` holds `r`, `lora_alpha` and option
flags. `adapter_model.safetensors` holds the matrices under keys like
`base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`.

**Name mapping is a string strip, with no transpose.** The engine keeps the
Hugging Face weight names verbatim (`model.layers.0.self_attn.q_proj.weight`)
and stores every projection as `(out, in)`, applied as `x @ W.T`. PEFT's `A` is
`(r, in)` and `B` is `(out, r)`, so `B @ A` already has the engine's layout. The
only layout trap is PEFT's `fan_in_fan_out` flag (for GPT-2-style `(in, out)`
weights), and it is rejected rather than transposed.

**Validation rejects what the engine cannot reproduce exactly**, by name: DoRA,
LoRA biases, `bias != "none"`, `modules_to_save`, per-module `rank_pattern` and
`alpha_pattern`, initialisations that rewrite the base weights (PiSSA, OLoRA,
LoftQ, CorDA, LoRA-GA), `lm_head` and `embed_tokens` targets, and any option
the loader does not recognise that is set to a non-default value. Silently
ignoring any of these gives a model that loads, generates fluent text, and is
wrong. rsLoRA is supported because it only changes the scale to
`lora_alpha / sqrt(r)`. Every matrix is also checked against the base model's
shapes (`validate_against(config)`).

## 3. Merged mode, and why it is free at inference

Merging computes, once at load time, for each targeted projection:

```
W' = cast_to_engine_dtype( float32(W) + scale · B @ A )
```

The sum is formed in fp32 and cast once. Doing it in fp16 would round the
update to fp16 before adding it to a much larger `W`, losing its low bits twice.

After merging, the weight dict has the same keys, shapes and dtypes as before.
The unmodified `LlamaModelGPU` runs it with the same kernels and the same FLOP
count as the base model. **Nothing about the adapter exists at runtime**, which
is why claim L1 should hold by construction: the benchmark checks that nothing
else (memory layout, caching) breaks that.

Two details:

- **Tied embeddings.** Llama 3.2 1B ties `lm_head.weight` to
  `model.embed_tokens.weight` (the loader stores one object under both names).
  Adapters on either are rejected, and merging only replaces the seven
  projections, so the alias survives. A test checks the identity.
- **Quantized bases: merge first, then quantize.** Merging into an existing
  `QuantWeight` would mean dequantize, add, re-quantize. That rounds the weight
  twice, and the second rounding can erase any update smaller than one
  quantization step (for int8 per-channel, `max|row| / 127`). `merge_adapter`
  therefore refuses a `QuantWeight`, and `merge_then_quantize` merges into the
  fp16 weights and quantizes once with the engine's own quantizers. That is also
  what a deployment does: a merged adapter is just a new checkpoint.

## 4. Unmerged mode: the math and its cost

For a projection with base `W` and adapter `(A, B, s)`:

```
y = x @ W.T  +  (x @ A.T) @ (s·B).T
```

`s` is folded into `B` when the adapter is registered, so the hot path does two
small matmuls and one add. For one token the extra arithmetic is
`2·r·(in + out)` against `2·in·out` for the base. For q_proj (2048 × 2048) at
r = 16 that is 1.6% more FLOPs; at r = 64, 6%.

**FLOPs are not the cost at batch 1.** Decode at batch 1 is bound by memory
bandwidth and by kernel launches. The low-rank path adds two matmul launches
and one add per projection: 3 × 7 projections × 16 layers = 336 extra launches
per token, plus Python dispatch for each. That fixed cost is roughly the same at
r = 8 and r = 64, which is why claim L3 expects overhead to depend more on the
number of adapted projections than on rank. At larger batch the launches are
amortised and the arithmetic share grows; that is the P5b question.

## 5. How it hooks into the engine without editing it

The engine routes every projection on its torch path through one function,
`engine.components_gpu.linear(x, w)`, the chokepoint the int8/int4 path already
uses. It gets the weight object but not the layer or projection name
([issue](../issues/p5a-linear-no-module-identity.md)). So the adapter rides on
the weight object:

1. **`LoRALinear`** replaces each targeted entry in the model's weight dict. It
   holds the base weight (fp16 tensor or `QuantWeight`) and every adapter
   registered for that projection.
2. **`install_lora_linear()`** swaps `components_gpu.linear` for `lora_linear`.
   `gqa_attention_gpu` and `swiglu_ffn_gpu` look `linear` up at call time, so they
   use it with no other change. For any other weight it does one type check and
   calls the original, so the base path is unchanged.
3. **Per-call selection.** The adapter for the current forward pass lives in a
   `contextvars.ContextVar`. `LoRAModelGPU.prefill/decode_step/forward_all/
   forward_varlen` accept `adapter=` and set it. A context variable is private to
   each thread and asyncio task, so concurrent requests cannot see each other's
   choice. Because the engine's `generate()` is a generator, the selection must be
   active whenever tokens are pulled; `generate_with_adapter` guarantees that
   ([issue](../issues/p5a-generator-context.md)).
4. **NumPy reference path.** `engine/components.py` writes `x @ w.T` inline, with
   no chokepoint. `LoRALinear.T` returns an object with `__array_ufunc__ = None`,
   which makes NumPy hand the matmul to that object's `__rmatmul__`, where the
   low-rank term is added. This keeps the fp32 engine usable as the oracle.

Quantized bases work in unmerged mode for free: `LoRALinear.base` can be a
`QuantWeight`, and the original `linear()` dequantizes it as before.

### The seam for P5b (per-row adapters)

On the serving layer's packed batch, each token row can belong to a different
request. `AdapterSelection.per_row(names, row_slots)` carries one slot index per
row (`-1` means base only). `LoRALinear._delta_rows` implements it as the simple
loop over the distinct adapters in the batch. It is the correctness reference;
P5b replaces that one method with gather + batched matmul (BGMV) and derives
`row_slots = seq_slots[meta.batch_indices]` from the engine's `BatchMeta`.
`forward_varlen(..., adapter=AdapterSelection.per_row(...))` already accepts it.

## 6. Tests

| Suite | Count | What it proves |
|---|---|---|
| `test_p5a_adapter.py` (fast) | 41 | Name mapping both ways; every rejected option and key; shape, rank and layer checks; loading real PEFT output; bf16 adapters; save/load round trip. |
| `test_p5a_lora_math.py` (fast) | 19 | Merge formula, dtype, single rounding, tied alias, zero-B identity, quantized refusal, merge-then-quantize; unmerged math on NumPy and torch, int8 base, per-row == per-call, install/uninstall, per-thread selection. |
| `test_p5a_tiny_oracle.py` (fast) | 20 | A 2-layer random Llama through HF+PEFT vs the engine, both modes, both engine paths: logits, greedy tokens, base == zero adapter exactly, adapter switching per call, generator pitfall. |
| `test_p5a_real_oracle.py` (slow; 1 `gpu` arm) | 12 (9 slow + 3 `gpu`) | The real 1B model: see §7. |

Run them with:

```bash
pytest                                              # default: 80 fast P5a tests + the rest, ~15 s
pytest -m slow tests/test_p5a_real_oracle.py -v     # real 1B, needs PLAYPARSE_WEIGHTS
PLAYPARSE_ORACLE_LAYERS=4 pytest -m slow tests/test_p5a_real_oracle.py   # low-memory mode
pytest -m "slow or gpu" tests/test_p5a_real_oracle.py                    # on a CUDA node
```

## 7. Oracle results (real Llama 3.2 1B Instruct)

Adapter: PEFT LoRA on all seven projections, r = 8, alpha = 16, `B` drawn from
N(0, 0.01²) so that it changes the greedy text. Prompt: one chat-formatted play
description (63 tokens); logits compared at every prompt position;
16 greedy tokens.

Artifacts: `results/p5a/oracle_local_L16.json` (full model) and
`results/p5a/oracle_local_L4.json` (first 4 real layers, the low-memory mode).

| Comparison (full 16 layers) | max abs logit difference | Bar |
|---|---|---|
| engine fp32 base vs HF fp32 | 2.7e-4 | < 1e-3 |
| engine fp32 **unmerged** vs HF + PEFT | 2.1e-4 | < 1e-3 |
| engine fp32 **merged** vs HF + PEFT | 2.2e-4 | < 1e-3 |
| engine fp32 merged vs unmerged | 4.9e-5 | — |
| engine fp16 (MPS) merged vs unmerged | 0.19 | < 0.25 |
| engine fp16 (MPS) unmerged vs HF + PEFT fp32 | 0.088 | < 0.5, argmax ≥ 95% of positions, top-10 overlap ≥ 8 |
| engine fp16 (MPS) merged vs HF + PEFT fp32 | 0.12 | same |
| engine fp16 (MPS) base vs HF fp32 | 0.086 | (reference for fp16 noise) |
| zero adapter (B = 0) vs base, both modes, both paths | 0 (bit-identical) | exact |

Greedy tokens (16 new): HF + PEFT, engine fp32 merged, engine fp32 unmerged,
engine fp16 merged and engine fp16 unmerged all produce the same sequence,
`"To parse this play, I'll break it down into its components:\n\n1."`. The
base model says "let's break it down" instead, so the adapter is doing
something.

The merged-vs-unmerged fp16 difference (0.19) is about twice the
fp16-vs-fp32 difference of the base model itself (0.086). That is expected:
merged mode rounds `W + sBA` to fp16 once per weight, unmerged rounds `W` and
the low-rank product separately, and each path accumulates its own rounding
over 16 layers. The 4-layer run is tighter (fp32 at most 6e-5, fp16 merged vs
unmerged 0.035).

The full-depth run took 31 minutes, almost all of it in the setup fixture,
because the laptop was swapping under other jobs (see
[issue](../issues/p5a-oracle-memory.md)). On an idle machine it is a few
minutes.

## 8. Local benchmark (indicative only)

Full write-up: [docs/benchmarks/p5a-local.md](../benchmarks/p5a-local.md).
Artifact: `results/p5a/bench_mps_20261004_151553.json`.

The laptop was shared with other agents' jobs, so the same model measured twice
(`base` vs `base_repeat`) differed by up to 10%. Within that noise:

- **Merged r=16 is indistinguishable from base** on all three prompts (L1
  consistent, not resolved).
- **Unmerged r = 8/16/64 is 0–15% slower than base** on most prompts; no rank
  trend is visible above the noise (L3 not resolved).
- **Linear-only microbenchmark:** the low-rank path adds 2–6% to the 112
  projections at batch 1, and 20–56% at batch 32 (noisy, not monotone in rank).
  Batch 32 is a preview for P5b, not the claim.
- Merged and unmerged produced identical greedy tokens on every benchmark
  prompt.

## 9. What is pending GPU

- `sbatch scripts/slurm/p5a_bench.sbatch` on one H100 (fill in account, QOS,
  partition and storage paths first). It runs the oracle tests including the
  CUDA arm, then the benchmark with 128 new tokens and 5 runs. Results land in
  `results/p5a/` stamped with job id, GPU and git SHA.
- L3's threshold is fixed in `scripts/p5a_bench.py` before that run:
  **≤ 15% decode tok/s overhead at r = 16, batch 1.**
- L3's batch-32 arm moves to P5b. The engine is batch-1 by construction (its
  README states every number is batch 1), and batching lives in the serving
  layer. The microbenchmark here measures the 112 projections at batch 32 as a
  preview, not as the claim.

## 10. Issues found

- [linear() has no module identity](../issues/p5a-linear-no-module-identity.md) — the main engine limitation for P5b.
- [Generator + context variable pitfall](../issues/p5a-generator-context.md)
- [transformers 5 config.json lacks `rope_theta`/`rope_scaling`](../issues/p5a-transformers5-rope-config.md)
- [Engine quantizes only from disk](../issues/p5a-quant-loader-needs-disk.md)
- [First real-model oracle run thrashed swap](../issues/p5a-oracle-memory.md)
