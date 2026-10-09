# P3 sweep on PACE: what each LoRA knob does on this task

**Runs:** 25 LoRA runs on PACE (A100 80GB / L40S), 2026-10-08/09. Every run is the base config (`configs/p3_sweep_50k.yaml`: 50k train plays, minimal prompt, r=16, α=32, all-linear, dropout 0.05, lr 2e-4, 2 epochs, early stop) with **one** knob changed. Each run scored its best checkpoint on `eval_lite` (2,795 plays from the 2024 season: 18 whole games plus a top-up so every bucket has ≥100 plays where possible). CIs: 95%, game-clustered bootstrap. Buckets show exact match %.

## Headline

1. **No LoRA knob matters much on this task.** All 23 knob variations at 50k plays land between 97.5% and 97.9%: a 0.4-point spread, well inside each run's ±0.8-point CI. Rank 2 (≈1.4M trainable parameters) scores the same as rank 64 (≈45M). That is LoRA's core premise, measured: the task needs only a tiny adjustment to the weights.

2. **Data is the lever.** 1k plays → 84.6%, 5k → 96.3%, 20k → 97.7%, 50k → ~97.7%, all 294k (R5) → 98.5%. The gain from 50k to 294k comes almost entirely from the rare buckets (fumble 79 → 85, penalty_stands 87 → 94, lateral 21 → 45), which simply appear more often in a bigger sample.

3. **The minimal prompt holds up at 50k.** The full-prompt control scores 97.9% vs 97.7% for the identical minimal-prompt run, the same no-difference result as the laptop ablation, now at 20× the data.

4. **Every run still trails the regex (99.1%)** on fumble, penalty_stands, and lateral. That is the motivation for T3b (PRD §17).

## Full table

### Reference

| Run | Exact match % [95% CI] | Fumble | Pen. stands | Lateral | Challenge |
|---|---|---|---|---|---|
| R0 regex | 99.1 [98.6, 99.5] | 92 | 100 | 62 | 96 |
| R5 (all 294k plays) | 98.5 [97.7, 99.0] | 85 | 94 | 45 | 96 |

### Rank (α = 2r)

| Run | Exact match % [95% CI] | Fumble | Pen. stands | Lateral | Challenge |
|---|---|---|---|---|---|
| rank_r2 | 97.5 [96.3, 98.2] | 79 | 83 | 17 | 94 |
| rank_r4 | 97.6 [96.4, 98.3] | 81 | 84 | 14 | 95 |
| rank_r8 | 97.6 [96.3, 98.3] | 80 | 86 | 14 | 92 |
| rank_r16 | 97.7 [96.6, 98.4] | 79 | 87 | 21 | 94 |
| rank_r64 | 97.7 [96.6, 98.4] | 80 | 87 | 14 | 94 |

### Alpha

| Run | Exact match % [95% CI] | Fumble | Pen. stands | Lateral | Challenge |
|---|---|---|---|---|---|
| alpha_r16_a16 | 97.8 [96.5, 98.5] | 80 | 89 | 21 | 93 |
| alpha_fixed16_r2 | 97.5 [96.2, 98.3] | 79 | 87 | 14 | 92 |
| alpha_fixed16_r4 | 97.7 [96.4, 98.4] | 77 | 88 | 17 | 94 |
| alpha_fixed16_r8 | 97.8 [96.7, 98.5] | 82 | 86 | 24 | 94 |
| alpha_fixed16_r64 | 97.6 [96.4, 98.3] | 80 | 84 | 17 | 94 |

### Target modules

| Run | Exact match % [95% CI] | Fumble | Pen. stands | Lateral | Challenge |
|---|---|---|---|---|---|
| targets_qv_r16 | 97.7 [96.6, 98.4] | 82 | 86 | 21 | 92 |
| targets_qkvo_r16 | 97.7 [96.5, 98.4] | 81 | 86 | 17 | 94 |
| targets_qv_matched_r106 | 97.7 [96.6, 98.4] | 78 | 87 | 17 | 95 |
| targets_qkvo_matched_r53 | 97.9 [96.7, 98.6] | 79 | 90 | 24 | 94 |

### Dropout

| Run | Exact match % [95% CI] | Fumble | Pen. stands | Lateral | Challenge |
|---|---|---|---|---|---|
| dropout_0.0 | 97.7 [96.5, 98.4] | 81 | 88 | 17 | 93 |
| dropout_0.1 | 97.7 [96.5, 98.4] | 81 | 88 | 17 | 94 |

### Learning rate (base run: 2e-4 = rank_r16)

| Run | Exact match % [95% CI] | Fumble | Pen. stands | Lateral | Challenge |
|---|---|---|---|---|---|
| lr_5.0e-5 | 97.5 [96.1, 98.2] | 78 | 87 | 17 | 92 |
| lr_1.0e-4 | 97.5 [96.1, 98.2] | 78 | 84 | 21 | 92 |
| lr_5.0e-4 | 97.8 [96.6, 98.5] | 82 | 87 | 17 | 94 |
| lr_1.0e-3 | 97.6 [96.3, 98.3] | 77 | 86 | 17 | 93 |

### Prompt control

| Run | Exact match % [95% CI] | Fumble | Pen. stands | Lateral | Challenge |
|---|---|---|---|---|---|
| prompt_full_r16 | 97.9 [96.9, 98.5] | 82 | 89 | 21 | 95 |

### Data size (nested samples of full train)

| Run | Exact match % [95% CI] | Fumble | Pen. stands | Lateral | Challenge |
|---|---|---|---|---|---|
| datasize_1000 | 84.6 [80.7, 86.7] | 26 | 48 | 3 | 74 |
| datasize_5000 | 96.3 [94.6, 97.2] | 53 | 82 | 14 | 92 |
| datasize_20000 | 97.7 [96.5, 98.4] | 83 | 83 | 21 | 94 |

Artifacts: `results/eval_lite/<run>/` (result.json + predictions.jsonl) and `results/sweep_runs/<run>/` (run_spec.json + metrics.jsonl). The 100k data-size point and R7 (full fine-tune) were still running when this was written.
