# Official yardage often differs from the "for N yards" in the text

**Phase:** P1 · **Severity:** surprise; a property of the task, documented and
measured, labels unchanged · **Guarding tests:** the `penalty_stands_spot_foul`
and `reception_fumble_lost` fixtures in `tests/test_ground_truth.py`; the rate is
recomputed on every build (`text_yards_check` in the manifest).

## What happened

Labels take yards from nflverse's structured columns, which follow the official
scoring rules. On some plays those rules give a different number from the phrase
printed in `desc`:

```
(15:00) (Shotgun) 4-J.Cook left end to BUF 45 for 5 yards (...). PENALTY on
BUF-73-D.Dawkins, Offensive Holding, 10 yards, enforced at BUF 41.
Rush credited to BUF 41 (spot of foul).
```

Label: `J.Cook rush_yds 1`. The text says "for 5 yards".

```
10-B.Nix pass short left to 38-J.McLaughlin to DEN 48 for 2 yards (...). FUMBLES
(24-K.Wallace), RECOVERED by SEA-17-J.Baker at DEN 47.
```

Label: `rec_yds 1`, `pass_yds 1`. The text says "for 2 yards".

## How it was found

The build compares the first "for N yards" in the final ruling of each
single-ball-carrier play with the label's rushing or receiving yards.

| Bucket | Plays checked (test) | Text disagrees (test) | Train rate |
|---|---|---|---|
| normal | 24,558 | 0.0% | 0.0% |
| td | 1,309 | 0.0% | 0.0% |
| challenge | 216 | 3.7% | 3.5% |
| penalty_stands | 680 | 16.9% | 12.6% |
| fumble | 284 | 41.5% | 32.4% |

## Root cause

Two official stat rules:

1. **Spot fouls by the offense.** If the offense commits a foul beyond the line of
   scrimmage (holding, an illegal block) and the penalty is enforced from the spot
   of the foul, the runner or receiver is credited only with the yards up to that
   spot. Sometimes the text says so ("Rush credited to BUF 41", "Officially a 2
   yard rush"); usually it does not.
2. **Fumbles that go backward.** The gain is measured to where the ball was
   recovered, not to where the runner was hit.

## Why the labels were not changed

The project's ground truth is the official stat line. The cross-check shows the
labels match the official game totals exactly, and fantasy leagues score the
official numbers. Re-deriving yards from the text would make the labels disagree
with the books.

## What it means for the project

* These plays are learnable in principle: the start of the play is implied by
  "to BUF 45 for 5 yards" (so the line was BUF 40), and the spot is printed
  ("enforced at BUF 41"). The model has to do field-position arithmetic, across
  midfield where the side changes ("to 50", "NO 48" vs "ATL 48").
* A regex that copies "for N yards" will be wrong on about a third of `fumble`
  plays and about one in seven `penalty_stands` plays, and right on all `normal`
  plays. That is one of the places the decision ladder should show ML earning
  its keep, or not.
* For the owner's hand audit: a label that disagrees with "for N yards" on these
  buckets is *expected*, not an error, as long as it matches the spot arithmetic.
