# p1-eval: a double-escaped `\b` silently disabled the spot-foul rule

## What happened

After the spot-foul rule was added to the regex parser, `penalty_stands` exact match
stayed at exactly 80.5%. The rule had no effect, and nothing raised an error.

## How it was found

The dev estimate did not move after a change that should have fixed most
`penalty_stands` misses. Printing the compiled pattern showed `\\b` (a literal
backslash followed by `b`) where `\b` (a word boundary) was intended.

## Root cause

The pattern was patched in by a Python script that wrote an already-escaped string
into an `rf"..."` literal, so the source ended up with `\\b`. A regex that can never
match fails silently: `search` returns `None`, and the code falls back to the stated
gain. An earlier version had a second problem: it used `[^.]*?`, which cannot cross
the periods inside player names such as `V.Ducasse`.

## Fix

The pattern now uses `\b` in the raw f-string and matches `.*?` up to `enforced at`.
`penalty_stands` then reached 100% on season 2018.

## Lesson

A rule that only adjusts values is invisible when it never fires. Every adjustment
rule now has its own unit test, so a dead pattern fails a test instead of quietly
doing nothing.

## Guarding test

`tests/test_regex_parser.py::test_parse_desc[offensive holding downfield: yards to the foul spot]`.
