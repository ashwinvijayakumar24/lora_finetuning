# P0 parameter and training-memory table (Llama 3.2 1B)

Generated 2026-10-04T20:42:32+00:00 from git `efe88fe005` by `scripts/p0_param_table.py`.
Total parameters: 1,235,814,400 (tied embeddings counted once).

Memory columns are GiB, excluding activations. Full FT assumes mixed-precision AdamW
(bf16 weights and grads, fp32 master, m, v = 16 B/param). LoRA assumes a frozen bf16
base plus fp32 adapters under AdamW (16 B per adapter param). Every count was checked
against `count_trainable` on a meta-device model with adapters injected.

| Method | Targets | r | Trainable | % of total | Weights | Grads | Optimizer (m+v+master) | Total |
|---|---|---|---|---|---|---|---|---|
| full | - | - | 1,235,814,400 | 100.000 | 2.30 | 2.30 | 13.81 | 18.42 |
| lora | qv | 4 | 425,984 | 0.034 | 2.30 | 0.00 | 0.00 | 2.31 |
| lora | qv | 8 | 851,968 | 0.069 | 2.31 | 0.00 | 0.01 | 2.31 |
| lora | qv | 16 | 1,703,936 | 0.138 | 2.31 | 0.01 | 0.01 | 2.33 |
| lora | qv | 64 | 6,815,744 | 0.552 | 2.33 | 0.03 | 0.05 | 2.40 |
| lora | qkvo | 4 | 851,968 | 0.069 | 2.31 | 0.00 | 0.01 | 2.31 |
| lora | qkvo | 8 | 1,703,936 | 0.138 | 2.31 | 0.01 | 0.01 | 2.33 |
| lora | qkvo | 16 | 3,407,872 | 0.276 | 2.31 | 0.01 | 0.03 | 2.35 |
| lora | qkvo | 64 | 13,631,488 | 1.103 | 2.35 | 0.05 | 0.10 | 2.51 |
| lora | all-linear | 4 | 2,818,048 | 0.228 | 2.31 | 0.01 | 0.02 | 2.34 |
| lora | all-linear | 8 | 5,636,096 | 0.456 | 2.32 | 0.02 | 0.04 | 2.39 |
| lora | all-linear | 16 | 11,272,192 | 0.912 | 2.34 | 0.04 | 0.08 | 2.47 |
| lora | all-linear | 64 | 45,088,768 | 3.649 | 2.47 | 0.17 | 0.34 | 2.97 |

## Measured on MPS

- LoRA all-linear r=16, dropout 0.05, batch 1 x seq 256: peak live tensors 4.09 GiB, peak driver allocation 5.25 GiB, saved activations 1.66 GiB, step time 0.65 s (after warm-up).
- LoRA all-linear r=16, dropout 0.0, batch 1 x seq 256: peak live tensors 3.72 GiB, peak driver allocation 4.84 GiB, saved activations 1.29 GiB, step time 0.63 s (after warm-up).
- full r=None: not measured (weights+grads+AdamW state alone need 18.4 GiB, above the 16 GiB of unified memory and MPS's recommended working set)
