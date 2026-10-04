# Benchmarks index

Every benchmark run, local or GPU, gets a row here and a detail file under
`docs/benchmarks/`. Every number links to a committed artifact in `results/`.

| Date | ID | What | Hardware | Result | Artifact |
|---|---|---|---|---|---|
| 2026-10-04 | P0-params | Trainable params + train-state memory, full FT vs LoRA r∈{4,8,16,64} × {qv,qkvo,all-linear}; LoRA r=16 step measured | Apple M4 16GB, MPS | full 1.236B / 18.4 GiB; LoRA all-linear r16 11.27M (0.91%) / 2.47 GiB; measured r16 b1×256: 4.09 GiB peak live, 0.65 s/step | [results/p0/param_table.json](../results/p0/param_table.json) · [detail](benchmarks/p0-param-memory.md) |
| 2026-10-04 | P1-crosscheck | Ground-truth game totals vs official weekly stats, 2015–2024 (50,602 player-games) | M4 CPU | 99.79% of player-games agree on all stats; PPR MAE 0.0035; 0 unexplained disagreements | [results/p1/crosscheck.json](../results/p1/crosscheck.json) · [detail](benchmarks/p1-crosscheck.md) |
| 2026-10-04 | P1-audit | 200-play stratified label audit (model-audited; owner spot-check pending: `results/p1/owner_spot_check.json`) | — | 2.5% wrong overall; lateral 21.7% (multi-lateral plays); every other bucket 0% | [results/p1/audit_results.json](../results/p1/audit_results.json) · [detail](benchmarks/p1-audit.md) |
| 2026-10-04 | P1-dataset | Frozen v1 dataset | M4 CPU | train 294,016 / val 38,102 / test 37,859 / eval_lite 2,795; test sha256 `13d0d714…` | [results/p1/dataset_manifest.json](../results/p1/dataset_manifest.json) |
| 2026-10-04 | p1-r0-dev | R0 regex on train 2015–22 against the eval agent's approximate labels (dev only; superseded by r0-real-gt) | M4 CPU | 99.82% exact | [results/r0_dev_train/result.json](../results/r0_dev_train/result.json) |
| 2026-10-04 | p1-local-throughput | R1/R2 plays/sec on MPS while the machine was heavily swapping (worst case) | M4 16 GB, MPS fp16 | R1 0.20 plays/s @ batch 16; R2 0.036 plays/s @ batch 4 | [detail](benchmarks/p1-local-throughput.md) |
| 2026-10-04 | **r0-real-gt** | **R0 regex on the frozen v1 val and test sets** | M4 CPU | **test 99.8% exact [99.7, 99.8]**; fumble 92.5%, lateral 62.1%, challenge ~92%; first run 96.0% before the zero-credit convention fix | [results/r0_test/](../results/r0_test/) · [detail](benchmarks/r0-real-gt.md) |
