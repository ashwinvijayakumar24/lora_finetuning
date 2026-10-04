# The shared system prompt says reversed plays are "nullified"; the labels do not

**Phase:** P1 · **Severity:** contract mismatch (open; needs an owner decision) ·
**Files:** `playparse/prompt.py` (not edited; it is a shared contract),
`playparse/data/ground_truth.py` rule 10.

## What happened

`SYSTEM_PROMPT` ends with:

> If the play is wiped out (No Play, or reversed), set nullified to true and
> credits to [].

The ground truth treats a replay reversal as a *change of ruling*, not a
wipe-out. For example:

```
8-L.Jackson pass short middle to 80-I.Likely for 10 yards, TOUCHDOWN. The Replay
Official reviewed the pass completion ruling, and the play was REVERSED.
8-L.Jackson pass incomplete short middle to 80-I.Likely (32-N.Bolton).
```

Label: `{"nullified": false, "credits": []}`. And a reversal can *add* credits:
"26-J.Gibbs left tackle to LA 1 for no gain ... REVERSED. 26-J.Gibbs left tackle
for 1 yard, TOUCHDOWN" is labeled with Gibbs' 1 yard and rushing TD. In the 2024
test season, 207 of the 339 replay-reviewed plays were reversed. 113 of those
reversed plays have non-empty credits, and only 5 are nullified (by a penalty, not
by the review).

## Why the labels follow the final ruling

The official stats (and fantasy points) count the final ruling. nflverse's
structured columns describe the final ruling, and the game-level cross-check
agrees with the official totals exactly on these plays. Labeling a reversed
touchdown as `nullified: true` would also be wrong for a reversal *to* a
touchdown.

## Consequence if left as is

Prompt-only baselines (R1–R4) will follow the instruction and output
`nullified: true` on reversed plays, losing exact match on most of the
`challenge` bucket for a reason unrelated to reading skill. A fine-tuned model
learns the labels and will ignore the instruction.

## Suggested fix (owner decision)

Reword the last sentence of `SYSTEM_PROMPT`, for example:

> If a penalty wipes the play out ("No Play"), set nullified to true and credits
> to []. If a replay review reverses the call, use the final ruling.

Changing the prompt changes every prompt-based baseline and every training
prompt, so it should be done once, before any P1 baseline is run.

## How to check the count

Load 2024 with `load_pbp_season(2024)` and filter
`replay_or_challenge_result == 'reversed'`; 204 of the 207 land in the
`challenge` bucket (the other three are a lateral and a two-point try, which
take precedence).
