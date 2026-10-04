# P1 game-level cross-check: ground truth vs official weekly stats

**Date:** 2026-10-04 · **Hardware:** Apple M4 (CPU only, ~30 s for all ten seasons)
**Artifact:** [`results/p1/crosscheck.json`](../../results/p1/crosscheck.json)
**Reproduce:** `PLAYPARSE_DATA_DIR=<raw> python -m playparse.data.crosscheck`

## Headline

The per-play labels, summed into game totals, agree with the official stats on
**every** passing, touchdown, interception, and two-point number across ten
seasons (50,602 player-games). Every remaining disagreement falls into one of
four explained classes; none is unexplained.

| Measure (2015–2024, dataset scope + kneels) | Value |
|---|---|
| Player-games where all ten stats agree | 99.79% |
| PPR fantasy-point MAE vs official, per player-game | 0.0035 points |
| pass_yds, pass_td, int, rush_td, rec_td, two_pt agreement | 100.00% each |
| rush_yds / rec / rec_yds agreement | 100.00% / 99.98% / 99.88% |
| fumble_lost agreement (vs nflverse's fantasy fumbles) | 97.66% |
| fumble_lost agreement, counting lost fumbles nflverse's formula leaves out | 99.88% |
| Unexplained disagreements | 0 |

The dataset itself excludes kneels (PRD scope). Measured on exactly the dataset
plays, all-stat agreement is 94.93% and MAE is 0.0127 points; the whole
difference is QB kneels (2,462 player-game rushing totals, usually −1 yard each).
That is a scope choice, not a labeling error, so both numbers are reported.

## What is compared, and why it is a real check

* **Ground truth side.** `build_label` runs on every in-scope play. Credits are
  summed per `(game_id, posteam, name)`, where the name is the label's
  `desc`-style name. Every v1 stat belongs to the offense, so the team is the
  play's `posteam`.
* **Official side.** `stats_player_week_<season>.parquet`. nflverse builds it from
  the NFL's own per-play stat records, not from the play-by-play columns the
  builder reads, so agreement is evidence, not a tautology. For each v1 stat
  the official expression is: `passing_yards`, `passing_tds`,
  `passing_interceptions`, `rushing_yards`, `rushing_tds`, `receptions`,
  `receiving_yards`, `receiving_tds`, `sack_fumbles_lost + rushing_fumbles_lost +
  receiving_fumbles_lost`, and the sum of the three `*_2pt_conversions` columns.
* **Player matching.** Join on `(game_id, team, abbreviated name)`. This uses the
  same information a label carries, so it tests the labels as the model will see
  them. Both sides get one normalization: a space after the first initial is
  removed ("D. Thomas" → "D.Thomas"). Official rows with no v1 stats (defenders,
  kickers, linemen) are not counted as players. A GSIS-id join was also run as a
  diagnostic; its agreement was within 0.05 percentage points of the name join in
  every season checked, so name matching is not hiding errors.
* **Unmatched rate.** 36 of 50,602 player-games (0.07%) appear on only one side.
  All of them belong to the explained classes below (mostly a lineman charged with a
  fumble, or a multi-lateral player whose yards net to zero on one side).
* **Fantasy points.** Both sides are scored with `ffscore.scorer`'s `ppr` config.
  The official side is checked against nflverse's own `fantasy_points_ppr` minus
  6 × `special_teams_tds` (return touchdowns are out of v1 scope). The scorer
  reproduces nflverse's points on 100% of official player-games.

## Per-season results (dataset scope + kneels and spikes)

Agreement is the share of player-games, among those where either side is
non-zero for that stat, where the two sides are equal.

| Season | Plays | Player-games | Unmatched | All stats agree | PPR MAE | pass_yds | pass_td | int | rush_yds | rush_td | rec | rec_yds | rec_td | fumble_lost | two_pt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2015 | 37,228 | 4,923 | 6 | 99.67% | 0.0048 | 100 | 100 | 100 | 100 | 100 | 100 | 99.82 | 100 | 96.55 | 100 |
| 2016 | 36,911 | 4,916 | 1 | 99.88% | 0.0017 | 100 | 100 | 100 | 100 | 100 | 100 | 99.92 | 100 | 98.76 | 100 |
| 2017 | 36,489 | 4,911 | 4 | 99.69% | 0.0053 | 100 | 100 | 100 | 100 | 100 | 99.95 | 99.82 | 100 | 96.49 | 100 |
| 2018 | 36,381 | 4,898 | 5 | 99.73% | 0.0038 | 100 | 100 | 100 | 100 | 100 | 99.95 | 99.82 | 100 | 97.49 | 100 |
| 2019 | 36,717 | 4,886 | 4 | 99.77% | 0.0054 | 100 | 100 | 100 | 100 | 100 | 99.95 | 99.95 | 100 | 96.43 | 100 |
| 2020 | 37,054 | 5,089 | 6 | 99.86% | 0.0022 | 100 | 100 | 100 | 100 | 100 | 100 | 99.95 | 100 | 97.87 | 100 |
| 2021 | 38,823 | 5,290 | 3 | 99.85% | 0.0023 | 100 | 100 | 100 | 100 | 100 | 100 | 99.93 | 100 | 97.85 | 100 |
| 2022 | 38,364 | 5,210 | 0 | 99.83% | 0.0027 | 100 | 100 | 100 | 100 | 100 | 100 | 99.93 | 100 | 97.60 | 100 |
| 2023 | 38,625 | 5,267 | 1 | 99.91% | 0.0012 | 100 | 100 | 100 | 100 | 100 | 100 | 99.95 | 100 | 98.82 | 100 |
| 2024 | 38,371 | 5,212 | 6 | 99.73% | 0.0059 | 100 | 100 | 100 | 99.96 | 100 | 99.95 | 99.75 | 100 | 98.76 | 100 |
| **all** | 374,963 | 50,602 | 36 | **99.79%** | **0.0035** | 100 | 100 | 100 | 100.00 | 100 | 99.98 | 99.88 | 100 | 97.66 | 100 |

The largest single player-game error is 8.2 PPR points (2024, a receiver the
text spells two ways in one game; see class 3). Otherwise it never exceeds 4.2.

### Dataset scope only (kneels excluded)

| Season | All stats agree | PPR MAE | rush_yds agreement |
|---|---|---|---|
| 2015 | 94.54% | 0.0145 | 87.68% |
| 2016 | 95.18% | 0.0107 | 88.63% |
| 2017 | 94.87% | 0.0148 | 88.59% |
| 2018 | 94.96% | 0.0126 | 89.02% |
| 2019 | 94.80% | 0.0146 | 88.36% |
| 2020 | 95.19% | 0.0116 | 89.24% |
| 2021 | 95.35% | 0.0106 | 89.73% |
| 2022 | 94.88% | 0.0117 | 88.87% |
| 2023 | 94.82% | 0.0109 | 88.42% |
| 2024 | 94.70% | 0.0155 | 88.59% |
| **all** | 94.93% | 0.0127 | 88.73% |

Every other column is identical to the table above.

## The remaining disagreements, explained

Counts are (game, player, stat) cells over all ten seasons, from
`mismatch_classes` in the JSON. Every cell is listed under `mismatches`.

| Class | Cells | Stats | Builder wrong? |
|---|---|---|---|
| 1. Uncategorized lost fumble | 54 | fumble_lost | No |
| 2. Multi-lateral play | 41 | rec_yds 38, fumble_lost 3 | Yes, unfixable from the columns |
| 3. `desc` spells one player two ways | 16 | rec 8, rec_yds 8 | No |
| 4. Play outside v1 scope | 1 (+2,462 kneels) | rush_yds | No (scope) |
| Unexplained | 0 | | |

1. **Uncategorized lost fumbles (54).** The builder charges every lost fumble to
   the player who fumbled: a center's botched snap ("K.Cousins Aborted.
   67-D.Dalman FUMBLES ... RECOVERED by PIT"), a lateral receiver, or a lineman
   who recovered a fumble and then fumbled it away. nflverse's fantasy formula only
   counts sack, rushing, and receiving fumbles, so these are missing from the
   official fantasy-fumble columns. They are present in the official
   `fumbles_lost_total`, and in all 54 cells the label equals that total. The
   label says what the text says. It costs 2 fantasy points against nflverse's
   formula, almost always for a lineman nobody rosters.
2. **Multi-lateral plays (41).** nflverse records only the last lateral, with a
   lumped yardage. A desperation play like "R.Bell ... for 12 yards. Lateral to
   E.Saubert ... for -3 yards. Lateral to B.Aiyuk ... for 5 yards ..." (six
   laterals) cannot be labeled correctly from the columns. These plays are in the
   `lateral` bucket, flagged noisy, and should be read as a known GT error rate
   there. Fixing them would mean parsing the text, which would make the labels
   depend on the thing being tested.
3. **The text spells one player two ways in one game (16).** Examples:
   "5-Di.Johnson" and "5-Dio.Johnson" (CAR, 2024 week 3), "11-M.Jones" and
   "11-M.Jones Jr." (DET 2019), "N.Williams" and "N.Williamss" (a typo, LA 2018).
   Each label uses the spelling printed in its own play, which is the contract
   ("name as written"). Game totals for that player then split across two names.
4. **Out of scope.** 2,462 QB rushing totals differ only because kneels are not
   in the dataset. One cell is a fake punt run (D.Ogunbowale, 35 yards, 2024 week
   15, `play_type == 'punt'`).

## Builder bugs this check found and fixed

| Bug | Cells fixed | Issue |
|---|---|---|
| Two-point tries with a dead-ball foul were labeled nullified | 42 two_pt cells (23 tries) | [p1-two-point-dead-ball-penalty](../issues/p1-two-point-dead-ball-penalty.md) |
| A player who fumbled twice on one play was not charged | 8 fumble_lost cells | [p1-same-player-fumbles-twice](../issues/p1-same-player-fumbles-twice.md) |
| "D. Thomas" credited although the text says "D.Thomas" | 58 credits whose name was not in `desc` | [p1-name-spacing](../issues/p1-name-spacing.md) |

Before these fixes, two_pt agreement ranged from 89% to 100% by season. Afterwards
it is 100% in every season.

The check also caught a bug in itself: the first version of the name
normalization rewrote "A.St. Brown" to "A.St.Brown" and broke matching for
multi-part surnames. See [p1-crosscheck-name-normalization](../issues/p1-crosscheck-name-normalization.md).

## What this does *not* show

Agreement of game totals does not prove every play is right. Two errors on the
same player in one game can cancel (for example +3 on one play and −3 on another).
The per-play audit ([p1-audit.md](p1-audit.md)) covers that gap. The game totals
also cannot see whether a yardage label matches the play's *text*, only whether it
matches the official books. Those two differ by design on some plays; see
"Official yards vs the text" in [docs/phases/P1-data.md](../phases/P1-data.md).
