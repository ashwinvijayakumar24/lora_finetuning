# Issues log

Every bug, surprise, and wrong assumption found during the project, in the order it
was found. Each entry: what happened, how it was found, root cause, fix, and the test
that now guards it.

## P0 — LoRA layer

- Injected LoRA layers started in train mode inside an `eval()` model, so dropout was silently on. Fixed: inherit the base layer's mode. [p0-injected-layers-start-in-train-mode](issues/p0-injected-layers-start-in-train-mode.md)
- bf16 merge → unmerge is not bit-exact (~9% of weights move one bf16 step). Inherent to bf16; documented; bounded by a test. Consequence: never switch adapters by repeated merge/unmerge on a bf16 base. [p0-bf16-unmerge-drift](issues/p0-bf16-unmerge-drift.md)
- Scripts run from a git worktree imported the main checkout's `playparse` via the editable install. Fixed: `sys.path` insert plus a subprocess test. [p0-worktree-script-imports-main-checkout](issues/p0-worktree-script-imports-main-checkout.md)

## P4 — distillation pipeline (caught in tests, before real data)

- A raw substring check accepted `J.Brown` inside `A.J.Brown`. Fixed: names match only at token boundaries. [P4-name-substring-match](issues/P4-name-substring-match.md)
- pandas turned `play_id` into a float, so cache keys `55.0` and `55` missed each other and labels would be bought twice. Fixed: `play_key` normalizes ids. [P4-play-id-float-cache-key](issues/P4-play-id-float-cache-key.md)
- An in-memory spend counter reset on every resume, so the budget cap wasn't a cap. Fixed: persistent ledger written before the cache append. [P4-budget-cap-per-process](issues/P4-budget-cap-per-process.md)
- Dropping code-fenced JSON would make the "unfiltered" R8 set secretly filtered. Fixed: strip one surrounding fence before parsing. [P4-code-fence-biases-r8](issues/P4-code-fence-biases-r8.md)

## P6 — registry and eval gate

- Percent-scale metrics (91 instead of 0.91) would make the 1-point bucket tolerance 100× too loose. Fixed: the loader rejects rates above 1. [P6-percent-scale-eval-artifact](issues/P6-percent-scale-eval-artifact.md)

## P1 — data and ground truth

Fixed:
- Two-point tries followed by a dead-ball foul were labeled nullified (found by the cross-check, 42 cells). [p1-two-point-dead-ball-penalty](issues/p1-two-point-dead-ball-penalty.md)
- A player who fumbled twice on one play was not charged the lost fumble (cross-check). [p1-same-player-fumbles-twice](issues/p1-same-player-fumbles-twice.md)
- "D. Thomas" was credited where the text says "D.Thomas" (name-in-desc check). [p1-name-spacing](issues/p1-name-spacing.md)
- The cross-check's own name normalization broke "A.St. Brown" (regression in the cross-check). [p1-crosscheck-name-normalization](issues/p1-crosscheck-name-normalization.md)
- About 20,000 timeouts were in the dataset as nullified plays (found by the audit). [p1-timeouts-filed-as-no-play](issues/p1-timeouts-filed-as-no-play.md)
- Declined fouls have `penalty == 0`, which mis-bucketed them (bucket tests). [p1-declined-penalty-flag](issues/p1-declined-penalty-flag.md)
- The system prompt said reversed plays are nullified, but labels follow the final ruling. Resolved by changing the prompt before any baseline ran. [p1-prompt-reversal-wording](issues/p1-prompt-reversal-wording.md)

Documented, labels unchanged:
- Official yards differ from the text's "for N yards" on 32–42% of fumble plays and 13–17% of penalty_stands plays. Labels follow official rules; this is genuine task difficulty. [p1-official-yards-vs-text](issues/p1-official-yards-vs-text.md)
- nflverse's fantasy formula omits some lost fumbles that its official totals include. [p1-uncategorized-fumbles](issues/p1-uncategorized-fumbles.md)
