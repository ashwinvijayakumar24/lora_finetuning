# Benchmarks index

Every benchmark run, local or GPU, gets a row here and a detail file under
`docs/benchmarks/`. Every number links to a committed artifact in `results/`.

| Date | ID | What | Hardware | Result | Artifact |
|---|---|---|---|---|---|
| 2026-10-04 | P0-params | Trainable params + train-state memory, full FT vs LoRA r∈{4,8,16,64} × {qv,qkvo,all-linear}; LoRA r=16 step measured | Apple M4 16GB, MPS | full 1.236B / 18.4 GiB; LoRA all-linear r16 11.27M (0.91%) / 2.47 GiB; measured r16 b1×256: 4.09 GiB peak live, 0.65 s/step | [results/p0/param_table.json](../results/p0/param_table.json) · [detail](benchmarks/p0-param-memory.md) |
| 2026-10-04 | P1-crosscheck | Ground-truth game totals vs official weekly stats, 2015–2024 (50,602 player-games) | M4 CPU | 99.79% of player-games agree on all stats; PPR MAE 0.0035; 0 unexplained disagreements | [results/p1/crosscheck.json](../results/p1/crosscheck.json) · [detail](benchmarks/p1-crosscheck.md) |
| 2026-10-04 | P1-audit | 200-play stratified label audit (model-audited; owner spot-check pending: `results/p1/owner_spot_check.json`) | — | 2.5% wrong overall; lateral 21.7% (multi-lateral plays); every other bucket 0% | [results/p1/audit_results.json](../results/p1/audit_results.json) · [detail](benchmarks/p1-audit.md) |
| 2026-10-04 | P1-dataset | Frozen v1 dataset | M4 CPU | train 294,016 / val 38,102 / test 37,859 / eval_lite 2,795; test sha256 `13d0d714…` | [results/p1/dataset_manifest.json](../results/p1/dataset_manifest.json) |
