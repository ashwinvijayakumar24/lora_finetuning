# Learning map — every list item → where you learn it

Each row is one item from the LoRA learning list. "Artifact" is the thing that
proves you learned it. It should be something you can point to in an interview, not
just something you read.

Phases refer to the build plan in [`PRD.md`](PRD.md) §11 (P0–P6).

## Core mechanics

| Learning item | Phase | Artifact that proves it |
|---|---|---|
| What LoRA is: `W + (α/r)·BA`, why it cuts trainable parameters and optimizer memory | P0 | Hand-written `LoRALinear`. A test shows it matches PEFT outputs to <1e-5. A table shows trainable parameters and optimizer-state bytes for full fine-tuning vs r=4/8/16/64. |
| The intuition: adaptation lives in a low-dimensional subspace | P3 | Rank sweep. If r=8 is about as good as r=64 on your task, that *is* the intuition, measured on your own data. |
| Knobs: rank, alpha, target modules, dropout, learning rate | P3 | Sweep table with one knob varied at a time. Train/val loss curves showing what overfitting looks like per knob. Short notes: "r=64 with α=16 underfit because the effective scale α/r was too small." |
| Why "all-linear" became the default | P3 | Target-module ablation: q,v only vs attention-only vs all-linear (attention plus MLP), at a fixed parameter budget. |
| QLoRA: 4-bit frozen base + bf16 adapter | P4 | Same task on bf16 LoRA vs QLoRA: quality delta, step time (the dequantization cost), and peak memory. Then QLoRA on 3B/8B, where bf16 would not fit as comfortably. |
| Dataset formatting and completion-only loss | P1 | A loss-masking unit test: prompt tokens get label `-100`, and the loss is computed only on completion tokens. A plot of the eval delta with masking vs without. |
| Tokenization gotchas | P1 | A written list of the ones you actually hit: chat template, BOS/EOS duplication, padding side, the label shift by one, truncation silently dropping the completion. Each has a regression test. |
| Gradient accumulation | P2 | A test that 4 micro-batches × accum 4 gives the same gradient as one batch of 16, within tolerance. This catches the classic loss-normalization bug. |
| Checkpointing the adapter | P2 | Adapter-only checkpoints (MBs, not GBs), with a test that reload reproduces eval exactly. |
| Held-out eval with harness discipline | P1 | Eval set frozen and hashed before any training. Splits made by time or source so test data can't leak in. Results carry a config hash. |

## Serving intersection

| Learning item | Phase | Artifact |
|---|---|---|
| Merge vs dynamic adapters | P5a | Your engine serves a merged adapter with identical latency to base (claim L1) and bit-identical output to unmerged (claim L2). The unmerged overhead is measured per rank (claim L3). |
| Multi-LoRA serving: S-LoRA / vLLM pattern, effect on KV cache and batching | P5b | Heterogeneous-adapter batching in `llm_serving_layer`: N adapters, one base, one batch. Throughput vs N, and vs the "one merged model per tenant" baseline (claims L4–L5). |
| Why multi-LoRA matters economically | P5b | A cost table: GPUs needed to serve 100 tenants as merged models vs as adapters. It is computed from your measured memory numbers, not quoted from a paper. |
| Fine-tuning's effect on inference behavior | P1/P5 | Format-validity rate (does the output parse?) for base vs fine-tuned, and latency unchanged after merge. |
| Adapter versioning and rollback | P6 | An adapter registry with an eval gate: a new adapter version is promoted only if it beats the current one on the frozen eval. Includes a demo of rejecting a deliberately broken adapter. |

## Distillation and the decision layer

| Learning item | Phase | Artifact |
|---|---|---|
| Sequence-level distillation: the teacher writes outputs, the student fine-tunes on them | P4 | Students trained on teacher outputs (R8–R9), plus the rare extra: a comparison against the same student trained on ground truth (R5), with the filter's precision measured against ground truth. |
| Data quality beats quantity; rejection sampling as the filter | P4 | Curve of student quality vs teacher-data size, with and without the rejection filter. |
| Decision tree: prompting vs RAG vs few-shot vs fine-tune vs distill | P1–P4 | The decision-ladder table (PRD §7, rungs R0–R9): quality per bucket, $/1k plays, p50 latency, and engineering hours for each rung. |
| Awareness: full fine-tuning, RLHF/DPO, catastrophic forgetting, when LoRA loses | P6 | A one-paragraph "boundaries" section per topic in the final README. Two of these are measured, not just described: full fine-tuning vs LoRA (R7, claim T5) and forgetting on a general benchmark (claim T7). |

## Explicitly skipped

Intrinsic-dimensionality theory, the DoRA/AdaLoRA/etc. variant zoo, pre-training,
implementing RLHF, and multi-node training. If an interviewer raises them, answer:
"out of scope for what I built, and here is where the boundary is."
