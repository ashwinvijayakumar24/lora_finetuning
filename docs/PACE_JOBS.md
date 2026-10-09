# PACE job ledger

## Round 1 (2026-10-08, commit `2d4eafd`): gate failed, nothing else ran

Gate 13901868 failed 2 real-model tests (it tested serving on the node's CPU; see `docs/issues/pace-gate-cpu-fallback-and-x86-fp16.md`). Its 49 dependents were cancelled.

## Round 2 (2026-10-08, commit `3f73c69`)

Every job depends on the gate (`afterok`). Each training job scores its best (or final) checkpoint on `eval_lite` at the end of its last allocation, writing to `results/eval_lite/<run>/`. Training and eval use the free, preemptible `embers` QOS; the two serving benchmarks use `inferno` on an H100.

Not yet submitted: the R5 test-set eval (after R5 finishes), QLoRA (R6, not built), and the vLLM comparison (L7, needs a separate `vllm` env).

| Job ID | Name | QOS | State at submit | Reason |
|---|---|---|---|---|
| 13905062 | playparse-gate | embers | RUNNING | None |
| 13905070 | pp-r5_full | embers | PENDING | Dependency |
| 13905076 | pp-rank_r2 | embers | PENDING | Dependency |
| 13905084 | pp-rank_r4 | embers | PENDING | Dependency |
| 13905606 | pp-rank_r8 | embers | PENDING | Dependency |
| 13905613 | pp-rank_r16 | embers | PENDING | Dependency |
| 13905644 | pp-rank_r64 | embers | PENDING | Dependency |
| 13905659 | pp-alpha_r16_a16 | embers | PENDING | Dependency |
| 13905663 | pp-alpha_fixed16_r2 | embers | PENDING | Dependency |
| 13905664 | pp-alpha_fixed16_r4 | embers | PENDING | Dependency |
| 13905665 | pp-alpha_fixed16_r8 | embers | PENDING | Dependency |
| 13905666 | pp-alpha_fixed16_r64 | embers | PENDING | Dependency |
| 13905667 | pp-targets_qv_r16 | embers | PENDING | Dependency |
| 13905668 | pp-targets_qkvo_r16 | embers | PENDING | Dependency |
| 13905669 | pp-targets_qv_matched_r106 | embers | PENDING | Dependency |
| 13905670 | pp-targets_qkvo_matched_r53 | embers | PENDING | Dependency |
| 13905671 | pp-dropout_0.0 | embers | PENDING | Dependency |
| 13905672 | pp-dropout_0.1 | embers | PENDING | Dependency |
| 13905673 | pp-prompt_full_r16 | embers | PENDING | Dependency |
| 13905674 | pp-lr_5.0e-5 | embers | PENDING | Dependency |
| 13905675 | pp-lr_1.0e-4 | embers | PENDING | Dependency |
| 13905676 | pp-lr_5.0e-4 | embers | PENDING | Dependency |
| 13905677 | pp-lr_1.0e-3 | embers | PENDING | Dependency |
| 13905678 | pp-datasize_1000 | embers | PENDING | Dependency |
| 13905679 | pp-datasize_5000 | embers | PENDING | Dependency |
| 13905680 | pp-datasize_20000 | embers | PENDING | Dependency |
| 13905681 | pp-datasize_100000 | embers | PENDING | Dependency |
| 13905682 | pp-r7_full_ft | embers | PENDING | Dependency |
| 13905683 | p5a_bench | inferno | PENDING | Dependency |
| 13905684 | p5b_bench | inferno | PENDING | Dependency |
