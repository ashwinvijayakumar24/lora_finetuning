# P2: micro-batch 4 pushed the 16 GB laptop into swap

## What happened

The first smoke run on the real 1B model (micro-batch 4 × accumulation 2, about
300 tokens per example) did not finish 4 optimizer steps in 10 minutes. The
process had grown to about 13 GB. The machine's swap was nearly full, and the
process sat in uninterruptible wait inside `backward()`.

## How it was found

`top` showed the training process at 13 GB, and `sample` showed it stuck in the
autograd engine. Other agents were also using about 5 to 14 GB at the time, so
the machine was oversubscribed.

## Root cause

Activation memory, not a leak. A stage-by-stage probe on MPS, with one 247-token
example, measured:

| Stage | Live tensor memory |
|---|---|
| bf16 base loaded | 2.30 GiB |
| Base forward with autograd graph | +0.66 GiB |
| LoRA forward with autograd graph | +1.65 GiB |

So LoRA costs about 6.7 MB of saved activations per token, compared with about
2.7 MB for the bare base model. The reason: PEFT keeps adapter weights in fp32
and casts each LoRA layer's input to fp32. With dropout on, it also saves the
dropout output. That happens in 7 projections × 16 layers, and the MLP's
`down_proj` input is 8192 values wide. Activations grow linearly with tokens per
micro-batch, so 4 × 300 tokens needs about 8 GB of activations, plus the logits
(tokens × 128,256 vocabulary) and their gradient.

## Fix

- The smoke script defaults to micro-batch 1 × accumulation 8. The effective
  batch is the same, and because the loss normalization is correct (see
  `docs/phases/P2.md` §4), the objective is the same too.
- The smoke script calls `torch.mps.set_per_process_memory_fraction(0.7)` so an
  oversized configuration fails fast with an out-of-memory error instead of
  swapping the whole machine.
- The smoke script pads every batch to a multiple of 64 tokens
  (`pad_to_multiple_of`). Even at micro-batch 1, a later run hit the memory cap
  at step 8 although live tensors peaked at only about 5.6 GB. Each new sequence
  length leaves differently sized blocks in the MPS allocator cache, and the
  cache fragments. With fixed shapes, memory stayed flat at 6.33 to 6.36 GiB from
  step 8 to step 30. Padding costs some compute on pad tokens, but they carry no
  loss and no attention from real tokens.
- `loop.release_cached_memory` empties the MPS cache after evaluation and
  generation, whose batch shapes differ from training's.
- `ModelSpec.gradient_checkpointing` is available (recompute activations during
  backward instead of storing them) for when larger micro-batches are needed.

## Why it matters for GPU runs

On an H100 with 80 GB, micro-batch 16 at about 300 tokens is about 32 GB of
activations by the same arithmetic. That fits, but it is the main memory cost,
not the model. If memory gets tight, the options in order are gradient
checkpointing, LoRA dropout 0 (saves one fp32 copy per layer), and a smaller
micro-batch with more accumulation.

## Guarding test

None automated; the guard is the memory cap in `scripts/p2_smoke_train.py`. The
smoke record (`results/p2/smoke.json`) logs peak memory for every run.
