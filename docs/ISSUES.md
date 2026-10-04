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

## P1 — eval harness and baselines

- The Llama chat template stamps today's date into every prompt, so prompts changed daily. Fixed: the date is pinned; training must use the same renderer. [p1-eval-chat-template-date](issues/p1-eval-chat-template-date.md)
- The checkpoint's generation config samples (temperature 0.6) by default. Fixed: greedy decoding requested explicitly. [p1-eval-generation-config-sampling](issues/p1-eval-generation-config-sampling.md)
- Official yards differ from the text's stated gain after downfield fouls and backward fumbles; the regex computes them from field positions. [p1-eval-official-yards-vs-stated-gain](issues/p1-eval-official-yards-vs-stated-gain.md)
- A double-escaped `\b` made one regex rule silently never fire. [p1-eval-spot-foul-regex-escape](issues/p1-eval-spot-foul-regex-escape.md)
- Ground-truth convention questions raised before real labels existed (zero-yard credits, botched snaps, two-point penalties, lateral yards). The zero-yard one turned out to be worth 3.8 points of R0 exact match; resolved by following the ground truth. [p1-eval-gt-convention-questions](issues/p1-eval-gt-convention-questions.md)
- Scripts run by file path imported the main checkout's code instead of the worktree's. [p1-eval-editable-install-shadows-worktree](issues/p1-eval-editable-install-shadows-worktree.md)
- R1 greedy outputs differ between batch 16 and batch 1 (padding changes fp16 numerics). Batch size is now part of the config hash. [p1-eval-batch-size-changes-greedy-outputs](issues/p1-eval-batch-size-changes-greedy-outputs.md)

## P5a — engine LoRA

- The engine's `linear(x, w)` is never told which layer it serves, and the LM head bypasses it, so the adapter must ride on the weight object and `linear` is replaced process-wide. Open: a small engine change would remove this. [p5a-linear-no-module-identity](issues/p5a-linear-no-module-identity.md)
- An adapter selected around `generate()` silently stopped applying when tokens were pulled after the block exited, which is exactly what a streaming server does. Fixed: `generate_with_adapter`. [p5a-generator-context](issues/p5a-generator-context.md)
- transformers 5 saves `config.json` without `rope_theta`/`rope_scaling`, which the engine needs. [p5a-transformers5-rope-config](issues/p5a-transformers5-rope-config.md)
- The engine can only quantize weights loaded from disk; merge-then-quantize needed an in-memory path. [p5a-quant-loader-needs-disk](issues/p5a-quant-loader-needs-disk.md)
- The first real-model oracle run thrashed swap. Fixed: zero-copy views of HF parameters and a low-memory layer option. [p5a-oracle-memory](issues/p5a-oracle-memory.md)
