# P0 benchmark: trainable parameters and training memory (Llama 3.2 1B)

- **Date:** 2026-10-04
- **Code:** git `efe88fe` (`scripts/p0_param_table.py --measure`)
- **Hardware:** Apple M4, 16 GB unified memory, MPS backend, torch 2.12.0,
  transformers 5.10.1
- **Artifacts:** [`results/p0/param_table.json`](../../results/p0/param_table.json),
  [`results/p0/param_table.md`](../../results/p0/param_table.md)

## Headline

LoRA r=16 on all seven projections trains 11.3M parameters, which is 0.91% of the
model. Its weights, gradients, and optimizer state together need about 2.47 GiB,
and 2.30 GiB of that is the frozen bf16 base. Full fine-tuning needs 18.4 GiB for
the same items, before any activations. That is more than this machine has, so
full fine-tuning was computed but not measured.

## How the numbers are computed

The analytic model lives in `playparse/lora/memory.py`. It reads `config.json`
only, so no weights are loaded.

**Parameter counts.** A LoRA adapter on a linear layer of shape (in → out) adds
`r × (in + out)` parameters, because A is `r × in` and B is `out × r`. For Llama
3.2 1B the per-layer shapes are:

| Projection | in → out | LoRA params at r=16 |
|---|---|---|
| q_proj | 2048 → 2048 | 65,536 |
| k_proj | 2048 → 512 | 40,960 |
| v_proj | 2048 → 512 | 40,960 |
| o_proj | 2048 → 2048 | 65,536 |
| gate_proj | 2048 → 8192 | 163,840 |
| up_proj | 2048 → 8192 | 163,840 |
| down_proj | 8192 → 2048 | 163,840 |

k and v are small because of grouped-query attention (GQA): the model has 32
query heads but only 8 key/value heads, so k and v project to 8 × 64 = 512
dimensions instead of 2048.

**Verification.** Every analytic count was checked against `count_trainable` on a
real `LlamaForCausalLM` built on the meta device (a PyTorch device that records
shapes but allocates no memory) with adapters actually injected. All 12 LoRA rows
and the full-model total (1,235,814,400) matched exactly. The same check runs in
the default test suite (`tests/test_lora_memory.py`).

**Memory assumptions.**

- *Full fine-tuning* uses mixed-precision AdamW. Each parameter has a bf16
  working copy (2 B), a bf16 gradient (2 B), an fp32 master copy (4 B), and fp32
  AdamW moments m and v (4 B each). That is 16 bytes per parameter.
- *LoRA* keeps the base frozen in bf16 (2 B per parameter, with no gradient and no
  optimizer state). Each adapter parameter is fp32 with an fp32 gradient and fp32
  m and v (16 B). It needs no master copy because it is already fp32.
- Activations are excluded from the table. They depend on batch size and sequence
  length rather than on the method, and they are measured below.

## Table

GiB, excluding activations.

| Method | Targets | r | Trainable | % | Optimizer state | Total |
|---|---|---|---|---|---|---|
| full | — | — | 1,235,814,400 | 100 | 13.81 | 18.42 |
| LoRA | q,v | 4 | 425,984 | 0.034 | 0.00 | 2.31 |
| LoRA | q,v | 8 | 851,968 | 0.069 | 0.01 | 2.31 |
| LoRA | q,v | 16 | 1,703,936 | 0.138 | 0.01 | 2.33 |
| LoRA | q,v | 64 | 6,815,744 | 0.552 | 0.05 | 2.40 |
| LoRA | q,k,v,o | 4 | 851,968 | 0.069 | 0.01 | 2.31 |
| LoRA | q,k,v,o | 8 | 1,703,936 | 0.138 | 0.01 | 2.33 |
| LoRA | q,k,v,o | 16 | 3,407,872 | 0.276 | 0.03 | 2.35 |
| LoRA | q,k,v,o | 64 | 13,631,488 | 1.103 | 0.10 | 2.51 |
| LoRA | all-linear | 4 | 2,818,048 | 0.228 | 0.02 | 2.34 |
| LoRA | all-linear | 8 | 5,636,096 | 0.456 | 0.04 | 2.39 |
| LoRA | all-linear | 16 | 11,272,192 | 0.912 | 0.08 | 2.47 |
| LoRA | all-linear | 64 | 45,088,768 | 3.649 | 0.34 | 2.97 |

"Optimizer state" is m + v (+ the master copy for full fine-tuning). The full
breakdown, including weights and gradients, is in `results/p0/param_table.md`.

## Measured on MPS

One LoRA training step on the real model: bf16 base on MPS, fp32 adapters on all
seven projections, r=16, AdamW, batch 1, sequence length 256, random tokens. Two
steps were run, and the second is reported, because AdamW allocates m and v
lazily on the first step.

| Dropout | Steady state | Peak live tensors | Saved activations | Peak driver allocation | Step time |
|---|---|---|---|---|---|
| 0.05 | 2.43 GiB | 4.09 GiB | 1.66 GiB | 5.25 GiB | 0.65 s |
| 0.0 | 2.43 GiB | 3.72 GiB | 1.29 GiB | 4.84 GiB | 0.63 s |

What the columns mean:

- **Steady state** is live tensors between steps. Because gradients are freed by
  `zero_grad(set_to_none=True)`, this should equal weights + m + v from the
  analytic model: 2.30 + 0.04 + 0.08 = 2.43 GiB. It matched to within 0.003 GiB,
  which confirms the analytic model on real hardware.
- **Saved activations** are live tensors right after the forward pass, minus the
  steady state. They are the tensors autograd keeps for the backward pass.
- **Peak driver allocation** includes the caching allocator's pool. MPS has no
  `max_memory_allocated()`, so this is the best available high-water mark: the
  pool only grows until `empty_cache()` is called.

### Observations

1. **At this size, activations rival the adapter's whole training state.** The
   adapter's weights, gradients, and moments total about 0.17 GiB. A single
   256-token sequence saves 1.3–1.7 GiB of activations. Once LoRA has removed the
   optimizer state, the activation memory is what decides batch size and sequence
   length. Gradient checkpointing (recomputing activations in the backward pass
   instead of storing them) is the lever for P2 if memory becomes the limit.
2. **Dropout adds 0.37 GiB (+29% of activations).** Each adapter applies dropout
   to its own input. Dropout has to remember which elements it zeroed, so it
   saves an extra full-size tensor for every adapter input. The inputs to the
   seven adapters sum to 20,480 features per token, and 20,480 × 256 tokens ×
   16 layers × 4 bytes is 0.31 GiB, which is close to the measured 0.37. q, k
   and v all read the same input, yet each makes its own fp32 copy and its own
   dropout mask. PEFT does the same. A fused q/k/v adapter could share one copy.
   This is a possible optimization, not something P0 needs.
3. **Matched budgets fall out of GQA.** q,v at r=8 and q,k,v,o at r=4 have exactly
   the same parameter count (851,968). P3's "matched trainable-parameter budget"
   ablation can use pairs like this directly.
4. **The MLP holds most of the all-linear budget.** gate, up, and down account
   for 70% of all-linear adapter parameters, because their 8192-wide side is 4×
   the hidden size.

## Why full fine-tuning was not measured

The weights, gradients, and AdamW state alone need 18.4 GiB, against 16 GiB of
unified memory. MPS also reports a recommended working set of 10.7 GiB on this
machine (`torch.mps.recommended_max_memory()`). Other agents share the machine,
so pushing it into swap would also disturb their runs. A pure-bf16 AdamW variant
(8 B/param, about 9.2 GiB plus activations) might just fit. However, it is not
the recipe P3 would use for R7, so measuring it would not answer the real
question. The R7 measurement belongs on the PACE GPUs.

## Reproduce

```
export PLAYPARSE_WEIGHTS=/path/to/Llama-3.2-1B-Instruct   # HF-format dir
python scripts/p0_param_table.py            # analytic table + meta-device check
python scripts/p0_param_table.py --measure  # also the MPS step measurement
```
