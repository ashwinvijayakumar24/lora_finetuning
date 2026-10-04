# `penalty == 0` on plays whose text contains a declined penalty

**Phase:** P1 · **Severity:** bucketing surprise (fixed) · **Guarding test:**
`tests/test_buckets.py::test_real_rows[penalty_stands_declined-penalty_stands]`
and the "declined" case in `test_precedence_on_combinations`.

## What happened

The first bucket rule was `penalty_stands = penalty == 1 and not nullified`. A
real play failed its test:

```
(12:11) (Shotgun) 4-D.Carr pass deep right to 89-A.Cooper pushed ob at TEN 12 for
41 yards (25-B.Wreh-Wilson). Penalty on TEN-25-B.Wreh-Wilson, Defensive Holding,
declined.
```

It landed in `normal`, although the text contains a penalty the model has to read
past.

## Root cause

nflverse sets `penalty = 0` (and leaves `penalty_type` empty) when every foul on
the play was declined. From a stats point of view nothing was enforced. From a
reading point of view, the play still has the penalty sentence.

## Fix

`penalty_stands` is now "a penalty is in the play (`penalty == 1` **or** the text
matches `penalty on`) and the play still counts". This moved about 2,900 training
plays from `normal` into `penalty_stands`, which changed the bucket's standalone
frequency from 1.39% to 2.37%. The rarest-first precedence was re-measured and
`fumble` now ranks ahead of `penalty_stands`.

## Lesson

Buckets exist to measure *reading difficulty*, so they should be defined by what
the text contains, with the structured columns as a helper, not the other way
around.
