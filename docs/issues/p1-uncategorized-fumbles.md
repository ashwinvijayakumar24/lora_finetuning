# nflverse's fantasy points leave out some lost fumbles

**Phase:** P1 · **Severity:** surprise in the reference data (documented, labels
unchanged) · **Guarding test:**
`tests/test_crosscheck.py::test_uncategorized_fumble_is_explained_by_total`, and
the `aborted_snap_center_lost` / `recoverer_fumbles_lost` fixtures.

## What happened

After the real fumble bugs were fixed, 54 `fumble_lost` cells still disagreed
with the official weekly stats, all in the same direction: the label charges a
lost fumble, the official fantasy columns do not. Examples:

* `18-K.Cousins Aborted. 67-D.Dalman FUMBLES at PIT 35, RECOVERED by PIT-90-T.Watt`
  (the center's snap; Dalman is charged).
* `... recovered by DAL-60-T.Guyton at DAL 33. 60-T.Guyton to DAL 32 for -1 yards.
  FUMBLES (5-J.Pitre), RECOVERED by HOU-95-D.Barnett` (a lineman recovers a sack
  fumble, then fumbles it away).
* A lateral receiver fumbling on a last-play desperation lateral.

## How it was found

Cross-check class analysis: for each disagreeing cell, the official
`fumbles_lost_total` column was compared to the label. It matched in all 54.

## Root cause

nflverse's `fantasy_points` only subtracts `sack_fumbles_lost +
rushing_fumbles_lost + receiving_fumbles_lost`. A fumble that is none of those
(a snap, a lateral, a fumble after a recovery) still counts in
`fumbles_lost_total`, but not in the fantasy formula.

## Decision

The labels keep charging these fumbles. The task is "read the play and say what
happened", and the text says that player lost the ball. ESPN-style scoring counts
total fumbles lost. The cost is at most 2 points against nflverse's formula,
almost always for an offensive lineman.

The cross-check reports both numbers: `fumble_lost` agreement against nflverse's
fantasy fumbles (97.66%) and against "fantasy fumbles or `fumbles_lost_total`"
(99.88%).

## If the owner wants the nflverse convention instead

Change `_fumble_loser` to charge only the ball carrier, receiver, or sacked QB.
The fixtures named above will fail and point at exactly the cases that change.
