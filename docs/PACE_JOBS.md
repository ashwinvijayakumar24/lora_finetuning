# PACE job ledger

Submitted 2026-10-08 from commit `2d4eafd`. Every job depends on the gate (`afterok`); each sweep training job has its own eval job that runs on `afterok` of that training job.

Not yet submitted (embers per-user submit limit hit): the eval of the last sweep job, R7 (`configs/r7_full_ft.yaml`) and its eval, the R5 test-set eval, QLoRA (R6, not built), and the vLLM comparison (needs a `vllm` env).

| Job ID | Name | QOS | State at submit | Reason |
|---|---|---|---|---|
| 13901868 | playparse-gate | embers | PENDING | Priority |
| 13901869 | pp-r5_full | embers | PENDING | Dependency |
| 13901870 | pp-eval-r5_full | embers | PENDING | Dependency |
| 13901871 | pp-rank_r2 | embers | PENDING | Dependency |
| 13901872 | pp-eval-rank_r2 | embers | PENDING | Dependency |
| 13901873 | pp-rank_r4 | embers | PENDING | Dependency |
| 13901874 | pp-eval-rank_r4 | embers | PENDING | Dependency |
| 13901875 | pp-rank_r8 | embers | PENDING | Dependency |
| 13901876 | pp-eval-rank_r8 | embers | PENDING | Dependency |
| 13901877 | pp-rank_r16 | embers | PENDING | Dependency |
| 13901878 | pp-eval-rank_r16 | embers | PENDING | Dependency |
| 13901879 | pp-rank_r64 | embers | PENDING | Dependency |
| 13901880 | pp-eval-rank_r64 | embers | PENDING | Dependency |
| 13901881 | pp-alpha_r16_a16 | embers | PENDING | Dependency |
| 13901882 | pp-eval-alpha_r16_a16 | embers | PENDING | Dependency |
| 13901883 | pp-alpha_fixed16_r2 | embers | PENDING | Dependency |
| 13901884 | pp-eval-alpha_fixed16_r2 | embers | PENDING | Dependency |
| 13901885 | pp-alpha_fixed16_r4 | embers | PENDING | Dependency |
| 13901886 | pp-eval-alpha_fixed16_r4 | embers | PENDING | Dependency |
| 13901887 | pp-alpha_fixed16_r8 | embers | PENDING | Dependency |
| 13901888 | pp-eval-alpha_fixed16_r8 | embers | PENDING | Dependency |
| 13901889 | pp-alpha_fixed16_r64 | embers | PENDING | Dependency |
| 13901890 | pp-eval-alpha_fixed16_r64 | embers | PENDING | Dependency |
| 13901891 | pp-targets_qv_r16 | embers | PENDING | Dependency |
| 13901892 | pp-eval-targets_qv_r16 | embers | PENDING | Dependency |
| 13901893 | pp-targets_qkvo_r16 | embers | PENDING | Dependency |
| 13901894 | pp-eval-targets_qkvo_r16 | embers | PENDING | Dependency |
| 13901895 | pp-targets_qv_matched_r106 | embers | PENDING | Dependency |
| 13901896 | pp-eval-targets_qv_matched_r106 | embers | PENDING | Dependency |
| 13901897 | pp-targets_qkvo_matched_r53 | embers | PENDING | Dependency |
| 13901898 | pp-eval-targets_qkvo_matched_r53 | embers | PENDING | Dependency |
| 13901899 | pp-dropout_0.0 | embers | PENDING | Dependency |
| 13901900 | pp-eval-dropout_0.0 | embers | PENDING | Dependency |
| 13901901 | pp-dropout_0.1 | embers | PENDING | Dependency |
| 13901902 | pp-eval-dropout_0.1 | embers | PENDING | Dependency |
| 13901903 | pp-prompt_full_r16 | embers | PENDING | Dependency |
| 13901904 | pp-eval-prompt_full_r16 | embers | PENDING | Dependency |
| 13901905 | pp-lr_5.0e-5 | embers | PENDING | Dependency |
| 13901906 | pp-eval-lr_5.0e-5 | embers | PENDING | Dependency |
| 13901907 | pp-lr_1.0e-4 | embers | PENDING | Dependency |
| 13901908 | pp-eval-lr_1.0e-4 | embers | PENDING | Dependency |
| 13901909 | pp-lr_5.0e-4 | embers | PENDING | Dependency |
| 13901910 | pp-eval-lr_5.0e-4 | embers | PENDING | Dependency |
| 13901911 | pp-lr_1.0e-3 | embers | PENDING | Dependency |
| 13901912 | pp-eval-lr_1.0e-3 | embers | PENDING | Dependency |
| 13901913 | pp-datasize_1000 | embers | PENDING | Dependency |
| 13901914 | pp-eval-datasize_1000 | embers | PENDING | Dependency |
| 13901915 | pp-datasize_5000 | embers | PENDING | Dependency |
| 13901916 | pp-eval-datasize_5000 | embers | PENDING | Dependency |
| 13901917 | pp-datasize_20000 | embers | PENDING | Dependency |
| 13901918 | p5a_bench | inferno | PENDING | Dependency |
| 13901927 | p5b_bench | inferno | PENDING | Dependency |
