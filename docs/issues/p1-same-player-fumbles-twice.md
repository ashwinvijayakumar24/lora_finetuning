# A player who fumbled twice on one play was not charged with the lost fumble

**Phase:** P1 · **Severity:** label bug (fixed) · **Guarding test:** the
`same_player_fumbles_twice` and `recoverer_fumbles_lost` fixtures in
`tests/test_ground_truth.py`.

## What happened

```
(2:21) 16-J.Goff to LA 29 for -5 yards. FUMBLES, and recovers at LA 28.
16-J.Goff sacked at LA 28 for -6 yards (54-K.Grugier-Hill). FUMBLES ...,
RECOVERED by PHI-24-C.Graham at LA 28.
```

Goff fumbled, picked the ball up, was sacked, and fumbled again; Philadelphia
recovered the second one. The label gave Goff no `fumble_lost`. The official stats
charge him one.

## How it was found

The game-level cross-check flagged `fumble_lost` cells with ground truth 0 and
official 1. All eight had the same structure: one player, two fumbles, with an
aborted snap or a bobble before the real fumble.

## Root cause

nflverse has two fumble slots: `fumbled_1_*` / `fumbled_2_*` for *who* fumbled,
and `fumble_recovery_1_*` / `fumble_recovery_2_*` for *who recovered*. The builder
paired slot k with recovery k. When the same player fumbles twice, nflverse fills
`fumbled_1` once and leaves `fumbled_2` empty, but still records the second
recovery:

| Column | Value |
|---|---|
| `fumbled_1_player_name` | J.Goff |
| `fumble_recovery_1_team` | LA (own team, not lost) |
| `fumbled_2_player_name` | *empty* |
| `fumble_recovery_2_team` | PHI (lost) |

The builder skipped slot 2 because nobody was listed as fumbling there.

## Fix

In `_fumble_loser`, an empty `fumbled_2` with a filled `fumble_recovery_2_team`
is read as the slot-1 player's second fumble. The rule is documented as rule 8 in
`ground_truth.py`.

## Result

8 of the 65 remaining `fumble_lost` disagreements disappeared. The other 57 are
explained in [p1-crosscheck.md](../benchmarks/p1-crosscheck.md) (54 uncategorized
fumbles that the official `fumbles_lost_total` confirms, 3 on multi-lateral plays).

## Lesson

Parallel "slot" columns are not always aligned by index. When two lists describe
one sequence of events, check what happens when an entity appears twice.
