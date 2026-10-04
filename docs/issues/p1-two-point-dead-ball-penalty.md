# Two-point tries followed by a dead-ball foul were labeled "nullified"

**Phase:** P1 · **Severity:** label bug (fixed) · **Guarding test:**
`tests/test_ground_truth.py::test_two_point_dead_ball_penalty_is_not_nullified`
and the `two_point_success_deadball_penalty` / `two_point_nullified` fixtures.

## What happened

The first version of the builder treated every `play_type == 'no_play'` row as
wiped out. That is right for ordinary plays. It was wrong for 23 successful
two-point tries, which therefore got `nullified: true` and no credits.

An example from 2015 (GB at CAR):

```
(Pass formation) TWO-POINT CONVERSION ATTEMPT. 12-A.Rodgers pass to 17-D.Adams is
complete. ATTEMPT SUCCEEDS. PENALTY on CAR-41-R.Harper, Defensive Holding, 5 yards,
enforced between downs.
```

The try counted (the score went up by two) and the official stats credit Rodgers
and Adams with a two-point conversion each.

## How it was found

The game-level cross-check. two_pt agreement with the official weekly stats was
89–97% in seven of ten seasons, and every disagreement had the same shape:
ground truth 0, official 1. Printing the plays behind those cells showed the same
pattern every time: "ATTEMPT SUCCEEDS" followed by a foul "enforced between downs"
(or on the kickoff).

## Root cause

A foul after the try is over (taunting, unsportsmanlike conduct, a late hit) is a
*dead-ball* foul. It does not undo the try; its yardage is applied to the next
kickoff, which the text calls "enforced between downs". nflverse still sets
`play_type = 'no_play'` on these rows. The only structured signal that separates
them from a really wiped-out try is `two_point_conv_result`:

| Situation | `play_type` | `two_point_conv_result` | Text ends with |
|---|---|---|---|
| Try stands, dead-ball foul | `no_play` | `success` / `failure` | "enforced between downs." |
| Try wiped out | `no_play` | empty | "- No Play." |

Across 2015–2024 all 29 `no_play` rows with a filled `two_point_conv_result` are
the first kind, and none say "No Play".

## Fix

`ground_truth.is_two_point_that_stands(row)` is true when `two_point_attempt == 1`
and `two_point_conv_result` is filled. `is_nullified(row)` is now
`play_type == 'no_play' and not is_two_point_that_stands(row)`. The two-point
branch also stopped relying on `play_type == 'pass'` to tell a pass try from a run
try, because these rows say `no_play`; it now checks whether a receiver is
recorded.

## Result

two_pt agreement is 100% in every season (999 player-games). The bucket for these
plays is `two_point` either way, so per-bucket metrics now see the correct label.

## Lesson

A structured "play type" column can encode a *bookkeeping* outcome (no normal
play happened) rather than the *scoring* outcome (the try counted). When a column
looks authoritative, check it against an independent total before trusting it.
