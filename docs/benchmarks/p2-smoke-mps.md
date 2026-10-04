# P2 smoke run: Llama 3.2 1B + LoRA on an Apple M4 (MPS)

**Artifact:** [`results/p2/smoke.json`](../../results/p2/smoke.json)
**Script:** `python scripts/p2_smoke_train.py --steps 30`
**Code:** git `383c3a3` (the commit with the MPS copy fix; smoke-script flags as recorded in the JSON)
**Date:** 2026-10-04

## Bottom line

The hand-written loop trains the real model correctly on MPS. On template plays,
loss fell from 0.459 to about 0.005 in 30 steps, and exact match on 16 held-out
template plays rose from 0% to 94%. But local throughput is about **144 tokens per
second**, so a real training run belongs on a GPU. One epoch over 100,000 examples
(about 26 million tokens) would take about 50 hours on this laptop.

## Hardware and software

| | |
|---|---|
| Machine | Apple M4, 16 GB unified memory, macOS 15.6.1 |
| Software | Python 3.13.7, torch 2.12.0, transformers 5.10.1, peft 0.21.2 |
| Contention | Shared with other agents running model jobs (5 to 14 GB each at times). Throughput is a lower bound. |

## Settings

| Setting | Value |
|---|---|
| Model | Llama 3.2 1B Instruct, base weights in bf16, no autocast |
| Adapter | PEFT LoRA, r=16, alpha=32, dropout 0.05, all 7 linear projections; adapter weights fp32 |
| Trainable parameters | 11.27M of 1.247B (0.90%) |
| Batch | micro-batch 1 × accumulation 8 = 8 examples per step |
| Optimizer | AdamW, lr 2e-4, 3 warmup steps, cosine to 10% |
| Steps | 30 (64 training examples, so about 3.75 epochs) |
| Sequence length | mean 262 tokens, max 313; completion mean 39 tokens; padded to a multiple of 64 |
| Data | template plays from `playparse.train.synthetic`, not real play-by-play |
| MPS memory cap | 75% of the recommended working set (8.0 GB) |

## Results

| Metric | Value |
|---|---|
| Median step time | 14.6 s |
| Throughput (all tokens) | 144 tokens/s (median) |
| Throughput (completion tokens only) | 20 tokens/s |
| Peak memory, Metal driver total | 6.36 GiB |
| Peak memory, live tensors | 5.35 GiB |
| Adapter checkpoint size | 45 MB |
| Model load time | 13.5 s |
| Training wall time | 565 s |

### Loss

| Step | 1 | 7 | 14 | 21 | 28 | 30 |
|---|---|---|---|---|---|---|
| Train loss | 0.459 | 0.102 | 0.010 | 0.005 | 0.004 | 0.005 |
| Val loss | | 0.062 | 0.017 | 0.024 | 0.013 | 0.011 |

The first-step loss is already low (0.46) because most completion tokens are
predictable JSON punctuation and key names once the system prompt has spelled
out the format.

### Generation on 16 held-out template plays (greedy, at most 96 new tokens)

| | Before training | After 30 steps |
|---|---|---|
| Exact match | 0% | 94% (15 of 16) |
| Parses as valid JSON label | 38% | 100% |
| Generation time for 16 plays | 40 s | 26 s |

A typical untrained output keeps jersey numbers in names and invents stats:
`{"player": "27-J.Gibbs", "stat": "pass_yds", ...}` for a run. After training:
`{"nullified":false,"credits":[{"player":"J.Gibbs","stat":"rush_yds","value":25}]}`.
Template plays are far easier than real ones, so this shows only that the loop
teaches the output format and simple rules.

## Memory notes

- Micro-batch 4 did not fit. It needed about 13 GB and pushed the machine into
  swap. See [`p2-mps-activation-memory`](../issues/p2-mps-activation-memory.md).
  LoRA through PEFT costs about 6.7 MB of saved activations per token on this
  model.
- Padding every batch to a multiple of 64 tokens kept the memory flat
  (6.33 to 6.36 GiB from step 8 onwards). Without it, varying shapes fragmented
  the MPS allocator cache, and the run hit the memory cap at step 8.

## What this means for planning

- **Local:** fine for tests, smoke runs, and debugging with a few hundred
  examples. Not for sweeps.
- **GPU (H100):** expect throughput to be roughly two orders of magnitude higher
  (micro-batch 16, bf16 autocast, fused kernels). This needs measuring, and it is
  the first job to run once PACE access is set up.
