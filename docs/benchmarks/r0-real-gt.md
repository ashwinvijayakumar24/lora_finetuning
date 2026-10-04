# R0 regex baseline on the real ground truth

**Date:** 2026-10-04 · **Hardware:** Apple M4 CPU · **Eval:** frozen v1 dataset (`results/p1/dataset_manifest.json`; test sha256 `13d0d714…`) · **CIs:** 95%, 1,000 bootstrap resamples over games

## Headline

The one-day regex parser scores **99.8% play exact match on the 2024 test set**. The remaining errors sit almost entirely in the rare, messy buckets: fumbles, laterals, and challenges. This is the PRD's risk #1 ("the regex is nearly as good as the adapter") coming true for common plays. See *What this means* below.

## Per-bucket exact match (%)

| bucket | val n | val exact | test n | test exact |
|---|---|---|---|---|
| **overall** | 38102 | **99.8 [99.7, 99.8]** | 37859 | **99.8 [99.7, 99.8]** |
| lateral | 16 | 43.8 [21.4, 69.2] | 29 | 62.1 [45.5, 78.8] |
| fumble | 523 | 91.2 [88.7, 93.4] | 530 | 92.5 [90.2, 94.5] |
| challenge | 330 | 98.5 [97.0, 99.7] | 335 | 97.9 [96.0, 99.2] |
| penalty_nullified | 2474 | 99.9 [99.8, 100.0] | 2794 | 99.5 [99.2, 99.7] |
| normal | 32145 | 100.0 [99.9, 100.0] | 31455 | 100.0 [100.0, 100.0] |
| interception | 435 | 99.8 [99.3, 100.0] | 393 | 100.0 [100.0, 100.0] |
| penalty_stands | 829 | 99.9 [99.6, 100.0] | 859 | 100.0 [100.0, 100.0] |
| td | 1205 | 99.7 [99.3, 99.9] | 1309 | 100.0 [100.0, 100.0] |
| two_point | 145 | 100.0 [100.0, 100.0] | 155 | 100.0 [100.0, 100.0] |

Game-level PPR fantasy-point MAE on test: 0.012 [0.006, 0.018] points per player-game. Throughput ≈ 50–95k plays/s on one CPU core, so R0's cost per 1k plays is effectively zero.

## How the number moved: 96.0% → 99.8%

The first run on the real ground truth scored 96.0%. Error analysis on **val only** (the test set was not inspected) showed that 1,485 of about 1,570 val errors were a single convention mismatch: the regex emitted zero-yard credits (`rush_yds: 0` for "no gain"), and the ground truth drops zero-value credits. The eval agent had flagged this convention as an open question before the real labels existed. Aligning it is a spec fix, not extra parsing effort, so it was applied. Other gaps found in the same analysis, such as substitution prefixes ("7-G.Smith back in at quarterback.") and official fumble yardage, were deliberately **not** fixed: the one-day budget is part of R0's definition.

## What this means for the project

1. **Claim T3 becomes the interesting claim.** On ~96% of plays, fine-tuning cannot beat the regex by much; the question is whether an adapter wins on fumbles (92.5%), laterals (62%), and challenges, without losing ground elsewhere.

2. **The decision-ladder answer for the common case is already in:** for well-formatted official play text, a regex is the right tool. Publishing that is the point of the ladder.

3. **Robustness is the regex's weak spot.** It is tuned to one text format. A format-shift eval (for example, the same plays rendered with different wording, or college play-by-play) would test whether an adapter generalizes where the regex breaks. This is recorded as a candidate extension in the P3 plan rather than a scope change now.

Artifacts: `results/r0_val/`, `results/r0_test/` (result.json + predictions.jsonl).
