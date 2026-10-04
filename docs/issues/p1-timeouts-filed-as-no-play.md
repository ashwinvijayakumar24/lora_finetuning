# Timeouts were in the dataset as "nullified plays"

**Phase:** P1 · **Severity:** dataset bug (fixed before freezing) · **Guarding
tests:** `tests/test_load_pbp.py::test_timeouts_filed_as_no_play_are_dropped`, and
the slow `test_real_season_loads` asserts no timeout text survives the filter.

## What happened

The first dataset build contained about 20,000 rows like this one, each labeled
`{"nullified": true, "credits": []}` in the `penalty_nullified` bucket:

```
posteam: UNK
desc: Timeout #2 by BAL at 00:31.
```

That was 44% of all `no_play` rows, 6% of the whole dataset, and almost half of
the `penalty_nullified` bucket.

## How it was found

The hand audit. Six of the first 22 sampled `penalty_nullified` plays were
timeouts. None of the automated checks could have flagged them: the label
("nothing happened, no credits") is technically consistent with the text.

## Root cause

nflverse gives every timeout `play_type = 'no_play'` and no `posteam`. The PRD's
scope rule, `play_type ∈ {pass, run, no_play}`, was written with "a play wiped out
by a penalty" in mind, and that is what `no_play` means for real plays.

## Fix

`load_pbp.is_timeout_row`: a `no_play` row whose text does not contain the word
"penalty" is dropped. That removes 20,007 rows: 20,006 begin with "Timeout", and
one is "E.Manuel in at QB. Timeout #1 by BUF at ...".

The slow test then found two timeouts that slipped through, because a scorer's
comment mentions a penalty: "Timeout #3 by TEN at 00:22. penalty was charged due
to an injury on the previous play." So a `no_play` row that starts with
"Timeout" and never says "No Play" is dropped as well. In total 20,009 rows are
removed across 2015–2024, and every remaining `no_play` row is a play with a
penalty in its text.

## Effect

| Split | Plays before | Plays after |
|---|---|---|
| train | 309,845 | 294,016 |
| val | 40,155 | 38,102 |
| test | 39,986 | 37,859 |

`penalty_nullified` shrank from 11.7% to 7.0% of training plays and is still the
second most common bucket, so the bucket precedence did not change. The test
file's hash changed. That is fine because nothing had been evaluated on it yet;
this is exactly why the eval is frozen at the end of P1 rather than at the start.

## Lesson

Automated checks verify that labels are *consistent*; only reading examples tells
you whether the rows are the *task*. A bucket that is mostly trivial inflates its
own accuracy and hides the hard cases it was created to measure.
