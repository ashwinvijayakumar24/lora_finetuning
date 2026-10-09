# T3b: the official yardage rules on fumbles and laterals, measured from the line of scrimmage

**Phase:** T3b (R0 + LOS fairness arm) · **Severity:** surprise; a property of the
ground truth, labels unchanged · **Guarding tests:** tests/test_t3b_regex_los.py

## What happened

Building the "R0 + LOS" arm meant writing down, as rules, how nflverse's official
yards relate to the spots in the text once the line of scrimmage is known. Several
of the rules are not what a reader would guess, and R0's one-day rules (P1) only
covered the first of them.

All examples are train-season plays; "LOS" is the line of scrimmage.

1. **A backward fumble is measured to where the ball was first touched**, not to
   where it was recovered. `to TB 44 for -1 yards. FUMBLES, touched at TB 44,
   RECOVERED by NYG at TB 24` (LOS TB 45) is −1, not −21.
2. **A forward bounce while the runner is behind the line is measured to the
   bounce, but never past the line.** `to CIN 22 for -3 yards. FUMBLES, RECOVERED
   by SEA at CIN 23` (LOS CIN 25) is −2. The same play recovered at CIN 28 is 0
   (no credit).
3. **A runner who recovers his own fumble and runs on is measured from the line to
   where he finally stopped** (`and recovers at X. P to Y for N yards`: Y − LOS).
   The forward-bounce floor at 0 applies to botched snaps: a snap that ends behind
   the line is a 0-yard rush.
4. **A handoff after a recovered fumble credits only the final runner**, from the
   line.
5. **Laterals credit only the first carrier and the last lateral taker.** The
   first carrier keeps his own leg, floored at 0 if it ended behind the line
   (unless the laterals gained nothing at all); the passer gets line of scrimmage
   to final spot; the last taker gets the rest, *net of the middle legs, which
   nobody is credited with*.
6. **A fumble out of the opponent's end zone is a touchback**: possession lost,
   so a lost fumble, even though no one "recovered" it.

## How it was found

`scripts/t3b_r0los_dev.py --regressions` on train (2015–2022) and val (2023):
every rule was added to fix a group of misses and kept only if no play R0 already
got right became wrong.

## Why it matters

* These are exactly the plays where the v1 adapter loses to R0 (fumble,
  penalty_stands, lateral). The rules are arithmetic on spots the text prints, so
  they are what schema v2 hands to code.
* Rules 2, 3 and 5 depend on the line of scrimmage; a parser without it cannot
  apply them. That is why the fairness arm exists.
* Rule 5's middle legs mean "sum the stated gains" is wrong on multi-lateral
  plays. The v2 ground truth sidesteps this: the last taker's `from` spot is where
  the previous credited leg ended, so `to − from` reproduces the official number.

## Effect

R0 → R0 + LOS on val: fumble 91.2% → 99.2%, lateral 43.8% → 68.8%, overall
99.78% → 99.94%. On train no play R0 got right became wrong.
