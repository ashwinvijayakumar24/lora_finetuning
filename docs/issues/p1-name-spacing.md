# Credited name "D. Thomas" does not appear in the text "88-D.Thomas"

**Phase:** P1 · **Severity:** label bug (fixed) · **Guarding tests:**
`tests/test_ground_truth.py::test_name_spacing_follows_desc` and
`::test_name_kept_when_desc_uses_the_spaced_form`.

## What happened

The output contract says credits use the player's name *as written in the play*.
The builder takes names from nflverse's `*_player_name` columns, which normally
match the text exactly. For Denver in 2017, nflverse stored Demaryius Thomas as
`D. Thomas` (with a space), while every play prints `88-D.Thomas`. 58 training
credits therefore named a player who is not literally in the input. A model
trained on them would have to learn to invent a space.

## How it was found

The dataset build reports `name_in_desc_rate`, the share of credits whose name is
a substring of `desc`. It was 99.971% on train. Listing the misses showed one
player in one season.

## Root cause

An nflverse roster-name formatting quirk. The opposite also happens: in 2018 both
nflverse and the text use "Dam. Williams" for Kansas City's Damien Williams.

## Fix

`ground_truth.desc_name(name, desc)`: keep the name if it is in `desc`; otherwise
try `normalize_name(name)`, which removes a space after the first initial only
("D. Thomas" → "D.Thomas"), and use it if *that* is in `desc`. The name is never
changed when the original spelling is already in the text, so "Dam. Williams"
stays as printed.

`name_in_desc_rate` is now 100% on every split.

## Related: a bug this fix caused in the cross-check

See [p1-crosscheck-name-normalization](p1-crosscheck-name-normalization.md).
