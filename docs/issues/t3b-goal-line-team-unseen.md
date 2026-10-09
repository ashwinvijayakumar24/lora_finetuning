# T3b: a touchdown spot that names an unseen team made the model write a safety

**Phase:** T3b · **Severity:** bug in the first v2 format, found in the local
pilot, fixed before the GPU run · **Guarding tests:** `test_yardline100_to_spot_is_the_inverse`
and `test_advance` (tests/test_t3b_spots.py), `rush_td` in tests/test_t3b_schema_v2.py

## What happened

The first v2 format wrote a touchdown's yardage spot as the defense's goal line in
desc's own spelling, `"<defteam> 0"` (for example `"CLE 0"`). The input is only
`posteam`, `los` and `desc`, and a long touchdown from the offense's own half often
never names the defense:

```
posteam: PIT
los: PIT 26
desc: (14:22) 30-J.Warren right end for 74 yards, TOUCHDOWN.
```

The ground truth said `"to":"CLE 0"`, a team the model cannot see. The pilot
adapter wrote `"to":"PIT 0"`: the only team in sight. `"PIT 0"` is PIT's *own*
goal line, so the converter (correctly) read it as a safety and credited −26
yards instead of +74.

## How it was found

The pilot's td bucket fell from 100% (v1) to 95%. To find out why without looking
at the 2024 eval plays, the pilot adapter was run on 160 seeded val (2023)
touchdown plays: 154 right, and all 6 misses were `"<posteam> 0"` on a play whose
defense is not named anywhere in the input.

How often that situation arises (ground truth, train and val):

| Split | Touchdown spots | Defense in `los` | Defense only in desc | Defense nowhere in the input |
|---|---|---|---|---|
| train | 17,334 | 16,090 | 99 | 1,145 (6.6%) |
| val | 2,094 | 1,939 | 18 | 137 (6.5%) |

## Root cause

A format choice: "write spots the way desc writes them" works for every spot desc
prints, but desc never prints the goal line a touchdown reaches. The target then
contained a token that could not be read from the input, and the model's best
guess was the one team name it could see, which means the opposite end of the
field.

## Fix

The opponent's goal line is written `"OPP 0"`: no team name, so nothing to guess.
`"OPP"` is not an NFL abbreviation, and the spot arithmetic already treats every
team other than `posteam` as the opponent, so the converter needed no change and
`"CLE 0"` still parses. A safety stays `"<posteam> 0"`, whose team is always in
the input. The v2 dataset was rebuilt (round trip still 100% on every split) and
the GPU config trains on the fixed files.

## Why it matters

* It cost the pilot 5 points on td and most of its fantasy-point error (one
  misread touchdown moves a player's yards by 70–80).
* Without the fix the pre-registered GPU run would have carried a known ~4% td
  loss, biasing the T3b test against the decomposition for a reason unrelated to
  the hypothesis.
* The pilot was not rerun with the fix (the local compute budget was spent); the
  GPU run is the first training on `"OPP 0"`. Its td bucket will show whether
  the fix worked.
