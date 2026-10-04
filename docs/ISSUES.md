# Issues log

Every bug, surprise, and wrong assumption found during the project, in the order it
was found. Each entry: what happened, how it was found, root cause, fix, and the test
that now guards it.

## P0 — LoRA layer

- Injected LoRA layers started in train mode inside an `eval()` model, so dropout was silently on. Fixed: inherit the base layer's mode. [p0-injected-layers-start-in-train-mode](issues/p0-injected-layers-start-in-train-mode.md)
- bf16 merge → unmerge is not bit-exact (~9% of weights move one bf16 step). Inherent to bf16; documented; bounded by a test. Consequence: never switch adapters by repeated merge/unmerge on a bf16 base. [p0-bf16-unmerge-drift](issues/p0-bf16-unmerge-drift.md)
- Scripts run from a git worktree imported the main checkout's `playparse` via the editable install. Fixed: `sys.path` insert plus a subprocess test. [p0-worktree-script-imports-main-checkout](issues/p0-worktree-script-imports-main-checkout.md)
