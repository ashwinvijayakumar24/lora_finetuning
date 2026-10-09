# T3b on the frozen test set: the pre-registered verdict

**Pre-registration:** PRD §17, committed (`bed20a0`) before any T3b code or run.
**Eval:** frozen test set, all 37,859 plays from 2024 (v1 sha `13d0d714…`; v2 copy has
the same plays and labels, round trip verified 100%). Scored once. Paired, game-clustered
bootstrap, 2,000 resamples, applied mechanically by `scripts/t3b_compare.py`.
Artifact: [`results/t3b/test_comparison.json`](../../results/t3b/test_comparison.json).

## Verdict: T3b not earned

| Bucket | T3b (spots) | R0 regex | R0 + LOS | R5 (yards) |
|---|---|---|---|---|
| **Overall** | **99.73** | 99.79 | 99.90 | 99.54 |
| penalty_stands | **100.00** | 100.00 | 100.00 | 93.48 |
| penalty_nullified | **100.00** | 99.50 | 99.50 | 99.93 |
| challenge | 97.91 | 97.91 | 99.40 | 97.01 |
| fumble | 86.04 | 92.45 | 98.11 | 83.21 |
| lateral (n=29) | 20.69 | 62.07 | 82.76 | 44.83 |
| normal, td, interception, two_point | 100 | ~100 | ~100 | 100 |

| Paired difference (points) | Estimate [95% CI] |
|---|---|
| T3b − R0 | −0.07 [−0.12, −0.02] |
| T3b − R0 + LOS | −0.18 [−0.23, −0.12] |
| **T3b − R5** | **+0.18 [+0.12, +0.25]** |

The pass condition required T3b to beat both regex arms overall with a CI excluding 0
and no bucket regression. It beats neither.

## What the experiment did show

1. **Decomposition works where the error was arithmetic.** Penalty plays where the play
   stands went from 93.5% (R5) to 100%, and the overall gap to the regex shrank from
   −0.25 to −0.07 points. T3b is significantly better than R5.
2. **The remaining errors are reading errors, not arithmetic.** On fumbles the model
   must pick *which* printed spot is official (the fumble point, the recovery, the
   advance); the T3b pilot's val diagnostic found those misses are choices between
   printed spots. Laterals (29 plays) got worse: multi-lateral plays need a computed
   spot nflverse never credits, so the model must still do arithmetic there.
3. **The fairness arm mattered.** Given the line of scrimmage, the regex improved to
   99.90% in 20 minutes of work. The same field that helped the model helped the
   hand-written rules more.

## Takeaway for the decision ladder

For clean, official play-by-play text with a fixed format, a regex written with the
official stat rules in mind is the right tool: 99.9% at ~74,000 plays/s on a CPU. The
fine-tuned 1B model gets within 0.1–0.2 points, ties or wins on 7 of 9 buckets, and
needed no rules at all, which matters most when the format is not fixed (not tested
here; see the format-shift idea in the project notes).
