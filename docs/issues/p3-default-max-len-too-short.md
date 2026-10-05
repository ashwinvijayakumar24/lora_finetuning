# P3: the default max_len of 512 would have crashed a full training run

## What happened

`configs/train_default.yaml` set `max_len: 512` with `on_overlength: raise`.
Tokenizing every record with the real chat template shows that six train plays
are longer than 512 tokens: four laterals and two challenges. The longest is a
595-token lateral (`2022_04_CHI_NYG`, play 3895). Val's longest example is 482
tokens, so val alone would not have shown the problem.

## Why it matters

`raise` is the right policy: silently truncating cuts off the completion, and
silently dropping hides data loss. But with 512, the first full-data run on the
H100 would have stopped at startup. The obvious workaround,
`on_overlength: drop`, would have discarded 4 of the 166 train laterals (2.4%),
from the very bucket where the adapter most needs to beat the regex (claim T3).

## How it was found

Building the pilot data: the pilot's enriched sample contains all 166 laterals,
and the dry run reported `max 595`. A full scan of train and val followed
(`results/p3_local_pilot/full_dataset_token_lengths.json`).

## Root cause

The 512 limit came from P2, which estimated lengths from synthetic template
plays. Real laterals and challenge reversals describe several actions in one
play and are much longer.

## Fix

`max_len: 640` in the default config (595 rounded up to the 64-token padding
multiple). The pilot config uses 768.

## Guarding test

`tests/test_p3_dataset_lengths_slow.py` (`slow`): tokenizes all of train and val
and asserts the longest example fits the default config's `max_len`.

## Related numbers from the same scan

The full train split is 87.9 million tokens (mean 299 per example), not the
PRD's estimate of 57 million (about 150 per example). Prompt tokens are 89% of
that, completions 11%. Padding to a multiple of 64 adds 14%. See
`docs/benchmarks/p3-local-pilot.md` for what this means for H100 time.
