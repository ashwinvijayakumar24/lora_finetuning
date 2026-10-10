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
| P3 | [P3.md](P3.md) | Done | Integration, laptop pilot, 25-run PACE sweep, R5 (99.54% on test), R6 QLoRA, R7 full fine-tune |
| P4 | [P4-distill-pipeline.md](P4-distill-pipeline.md) | Built; runs need an API key (B2) | Teacher labeling with budget cap; rejection filter |
| P5a | [P5a.md](P5a.md) | Done (H100) | LoRA in the from-scratch engine; L1 earned, L2 earned, L3 not earned at batch 1 |
| P5b | [P5b.md](P5b.md) | Done (H100) · [L7](L7.md) | Multi-LoRA batching, adapter pool, adapter-keyed prefix cache; vLLM ~11× faster (L7 not earned) |
| P6 | [P6-registry.md](P6-registry.md) · [gate demo](P6-gate-demo.md) | Done: broken adapter refused on real runs | Adapter registry with per-bucket eval gate and rollback |
| T3b | [T3b.md](T3b.md) | Done: not earned vs regex; beats R5 | Let the model read field spots and code do the arithmetic (PRD §17) |
| T7 | [T7.md](T7.md) | Done | Forgetting check: no MMLU drop after fine-tuning |
