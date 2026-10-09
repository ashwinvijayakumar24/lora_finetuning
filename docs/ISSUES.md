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

## P2 — training loop

- **Serious, silent:** `non_blocking=True` CPU→MPS copies of temporary batches delivered garbage labels (eval loss 2.28 instead of 0.53 depending on batch size) and caused OOMs. Fixed: blocking copies; two MPS tests failed before the fix. Earlier smoke numbers were discarded. [p2-mps-nonblocking-copy-race](issues/p2-mps-nonblocking-copy-race.md)
- The chat template stamps today's date into the prompt. Fixed: pinned date, with a clock-faking test. [p2-chat-template-date](issues/p2-chat-template-date.md)
- Micro-batch 4 needed ~13 GB on MPS (fp32 copies of LoRA inputs), and varying shapes fragmented the allocator. Fixed: micro-batch 1 × accumulation 8, pad to a multiple of 64, MPS memory cap, cache emptied after eval. [p2-mps-activation-memory](issues/p2-mps-activation-memory.md)
- Toy models with tied embeddings at std 0.02 cap the logits, so LoRA plateaued at loss 3.57 in tests. Fixed in tests by rescaling embeddings. [p2-tiny-lora-loss-floor](issues/p2-tiny-lora-loss-floor.md)
- In transformers 5, `apply_chat_template(tokenize=True)` returns a dict, not a list of ids. [p2-apply-chat-template-returns-batchencoding](issues/p2-apply-chat-template-returns-batchencoding.md)

## P5b — multi-LoRA serving

- Re-installing P5a's `linear()` wrapper after P5b's formed a delegation cycle (RecursionError). Fixed: the installer walks the wrapper chain. [p5b-linear-wrapper-cycle](issues/p5b-linear-wrapper-cycle.md)
- Adding a sibling repo root to `sys.path` let its `tests` package shadow ours; the full suite passed only by file sort order. Fixed: `engine`, `serving`, `bench` registered by file location. [p5b-serving-root-shadows-tests](issues/p5b-serving-root-shadows-tests.md)
- The serving layer's `Scheduler.step()` has no forward hook; worked around with recorded slots checked against BatchMeta every step. Proposed upstream patch included. [p5b-scheduler-forward-hook](issues/p5b-scheduler-forward-hook.md)
- Early kernel costs were not about adapters: host syncs per projection (v1), per-row weight copies in prefill (v2), and a slow MPS strided-slice fallback. Fixed: per-forward BatchPlan. [p5b-kernel-costs-not-about-adapters](issues/p5b-kernel-costs-not-about-adapters.md)
- The first SLO calibration measured a negative TTFT; the SLO guard caught it. [p5b-calibration-negative-ttft](issues/p5b-calibration-negative-ttft.md)

## P3 — integration and local pilot

- The pinned chat-template date was defined twice (train and eval). Fixed: one constant in `playparse/prompt.py`, plus a train/eval prompt-parity test. [p3-duplicated-chat-date](issues/p3-duplicated-chat-date.md)
- Training with our own LoRA wrote checkpoints no downstream loader could read. Fixed: PEFT format everywhere. [p3-lora-checkpoints-not-peft-format](issues/p3-lora-checkpoints-not-peft-format.md)
- The old val callback scored the first N val plays, which are almost all normal. Fixed: seeded, bucket-stratified subset. [p3-val-callback-head-of-file](issues/p3-val-callback-head-of-file.md)
- 6 train plays exceed 512 tokens, so the first full H100 run would have crashed at startup. Fixed: `max_len` 640. [p3-default-max-len-too-short](issues/p3-default-max-len-too-short.md)
- The pilot's per-bucket allocation ignored caps (caught by a test; no effect on the real pilot data). [p3-pilot-allocation-ignored-caps](issues/p3-pilot-allocation-ignored-caps.md)
- Two agents' real-model jobs overlapped on the laptop GPU, distorting both timings. [p3-shared-mps-contention](issues/p3-shared-mps-contention.md)
- With no system message, the Llama template still renders a dated 35-token header, so the date pin stays necessary. [p3-minimal-prompt-keeps-template-header](issues/p3-minimal-prompt-keeps-template-header.md)
- Scoring an adapter with a prompt style it wasn't trained on would silently score lower and look like a weaker adapter. Fixed: `LoRAPredictor` refuses a mismatch. [p3-prompt-style-mismatch-is-silent](issues/p3-prompt-style-mismatch-is-silent.md)

## PACE runs

- The first gate ran the real-model serving test on the node's CPU (the device helper only knew MPS and CPU); on x86 fp16, v1 diverged from the single-request reference while v2 matched. Fixed: CUDA preferred. Open: confirm the x86 fp16 divergence is a near-tie. [pace-gate-cpu-fallback-and-x86-fp16](issues/pace-gate-cpu-fallback-and-x86-fp16.md)
- The job scripts never passed `--prompt-style minimal`, so the eval guard refused every minimal-prompt adapter. Caught by the second gate before any training run reached its eval. Fixed in the gate, training, and eval jobs; both eval paths re-checked locally. (Commit `ba7599c`.)
- `embers` caps queued jobs per user at 50, and an SSH drop cut one submission short. Fixed: eval runs inside each training job, and `submit_all.sh` runs under `nohup` on the login node.

## T3b — spot decomposition

- A touchdown spot `"<defteam> 0"` named a team the input often omits, so the pilot model wrote a safety. Fixed: `"OPP 0"`, before the GPU run. [t3b-goal-line-team-unseen](issues/t3b-goal-line-team-unseen.md)
- Official fumble and lateral yardage rules, measured from the line of scrimmage (found while building R0 + LOS). [t3b-official-fumble-yardage-conventions](issues/t3b-official-fumble-yardage-conventions.md)
