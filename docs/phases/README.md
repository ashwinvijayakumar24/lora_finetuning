# Phase log

One write-up per phase, written while the phase ran: what was built, what was
tested, what broke, what was measured, and what is still open. Read them in this
order to follow the project.

| Phase | Doc | Status | One-line summary |
|---|---|---|---|
| P0 | [P0.md](P0.md) | Done | Hand-written LoRA layer, exact match with PEFT; parameter and memory table |
| P1 | [P1-data.md](P1-data.md) | Done | Ground truth from nflverse, validated against official weekly stats (99.79% agreement); frozen splits |
| P1 | [P1-eval.md](P1-eval.md) | Done | Eval harness, cluster bootstrap, regex baseline (99.8% on test), prompting and frontier rungs |
| P2 | [P2.md](P2.md) | Done | Hand-written training loop: prompt masking, exact gradient accumulation, exact resume |
| P3 | [P3.md](P3.md) | Local pilot done; full runs wait on GPU | Integration; first LoRA run (93.7% vs regex 97.6%); sweep ready |
| P4 | [P4-distill-pipeline.md](P4-distill-pipeline.md) | Built; runs wait on API key | Teacher labeling with budget cap; rejection filter |
| P5a | [P5a.md](P5a.md) | Done locally; H100 timing pending | LoRA in the from-scratch engine, merged and unmerged |
| P5b | [P5b.md](P5b.md) | Done locally; H100 benchmarks pending | Multi-LoRA batching, adapter pool, adapter-keyed prefix cache |
| P6 | [P6-registry.md](P6-registry.md) | Built | Adapter registry with per-bucket eval gate and rollback |
| T3b | [T3b.md](T3b.md) | Pilot done; GPU run queued | Let the model read field spots and code do the arithmetic (PRD §17) |
