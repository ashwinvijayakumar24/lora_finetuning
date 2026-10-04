# Benchmarks index

Every benchmark run, local or GPU, gets a row here and a detail file under
`docs/benchmarks/`. Every number links to a committed artifact in `results/`.

| Date | ID | What | Hardware | Result | Artifact |
|---|---|---|---|---|---|
| 2026-10-04 | P0-params | Trainable params + train-state memory, full FT vs LoRA r∈{4,8,16,64} × {qv,qkvo,all-linear}; LoRA r=16 step measured | Apple M4 16GB, MPS | full 1.236B / 18.4 GiB; LoRA all-linear r16 11.27M (0.91%) / 2.47 GiB; measured r16 b1×256: 4.09 GiB peak live, 0.65 s/step | [results/p0/param_table.json](../results/p0/param_table.json) · [detail](benchmarks/p0-param-memory.md) |
