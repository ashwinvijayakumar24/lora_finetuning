# p1-eval: official yardage is not always the "for N yards" in the text

## What happened

The first regex dev run scored 80% exact match on `penalty_stands` and 65% on
`fumble`, while `normal` was 99.5%. Almost every miss had the right players and stats
but the wrong yardage value.

There were two cases, both in train seasons:

1. **Offensive foul enforced at a spot.** `25-L.McCoy right tackle to BUF 35 for
   6 yards. PENALTY on BUF-62-V.Ducasse, Offensive Holding, 10 yards, enforced at
   BUF 31.` The official rushing yards are **2**, not 6. The run started at BUF 29,
   the foul was at BUF 31, and yards gained past the foul spot do not count.
2. **Fumbles.** `34-A.Collins left tackle to BAL 37 for -3 yards. FUMBLES, RECOVERED by
   BUF-58-M.Milano at BAL 35.` The official rushing yards are **-5**. A fumble that
   rolls backward is charged to the carrier, measured to the recovery spot. A forward
   bounce counts only when the carrier recovers the ball himself.

## Why it matters

- "Penalty yards are ignored" (PRD decision #2) does not mean "use the stated gain".
  The ground truth follows the official stat, and that needs field-position
  arithmetic.
- The text never states the line of scrimmage. It has to be worked out from the end
  spot and the stated gain. This is real reasoning, and it is exactly the kind of case
  where a 1B model may fail even after fine-tuning. It deserves a call-out in the
  per-bucket results.

## How it was found

Running `scripts/regex_dev_estimate.py --mismatches 40` on season 2018, which compares
the parser against the structured columns.

## Fix

`regex_parser._adjust_yards` converts spots like `BUF 31` into yards from the
offense's own goal line. This needs `posteam`. It then applies two rules:

- An offensive foul enforced between the start and the end of the gain caps the
  yards at the foul spot.
- A fumble recovered behind the downed spot moves the yards back to the recovery
  spot.

On the full train set this raised `penalty_stands` from 80% to 99.9% and `fumble`
from 65% to 94%. Without `posteam`, the parser falls back to the stated gain.

## Guarding tests

In `tests/test_regex_parser.py`: the cases "offensive holding downfield", "touchdown
nullified by a spot foul", "fumble lost after reception", and "fumble recovered by
offense", plus `test_without_posteam_falls_back_to_stated_yards`.
