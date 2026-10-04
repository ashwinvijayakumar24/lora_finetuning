# Cross-check name normalization broke multi-part surnames

**Phase:** P1 · **Severity:** validation-tool bug (fixed, never reached the
dataset) · **Guarding tests:**
`tests/test_ground_truth.py::test_name_spacing_follows_desc` (asserts
`normalize_name("A.St. Brown") == "A.St. Brown"`) and
`tests/test_crosscheck.py::test_official_name_spacing_is_normalized`.

## What happened

After the [name-spacing fix](p1-name-spacing.md), the official stats also needed
"D. Thomas" turned into "D.Thomas" so the two sides would join. The first version
replaced *every* ". " with ".". On the next cross-check run, agreement fell from
about 99.8% to 98.6% in 2022 and the PPR MAE rose roughly 50-fold.

## How it was found

The per-season summary printed by the cross-check. A sudden drop in seasons that
had been clean, right after a one-line change, pointed straight at the change.
The mismatch list was full of players like "A.St. Brown" and "E.St. Brown".

## Root cause

"A.St. Brown" has a dot followed by a space *inside the surname*. The global
replace turned the official name into "A.St.Brown", while the label (correctly)
kept "A.St. Brown". Every Amon-Ra St. Brown game became two unmatched rows.

A second, smaller problem followed: normalizing only the official side meant
"Dam. Williams" (which the text really prints) no longer matched either.

## Fix

* `normalize_name` only touches a space right after the *first* initial:
  `^([A-Za-z]+)\.\s+` → `\1.`.
* The cross-check applies the same normalization to *both* sides of the join, so
  the join key is symmetric while the labels themselves keep the text's spelling.

## Lesson

A validation tool can be wrong too, and a sudden regression right after a
"harmless" cleanup is the signature. Keep the normalization narrow, apply it
symmetrically, and pin the tricky inputs in a test.
