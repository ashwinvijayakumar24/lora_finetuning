# P6 eval-gate demo on real adapters

**Run:** 2026-10-09 on PACE. Candidate = `configs/gate_demo_broken.yaml` (R5's recipe with
the prompt mask **off**, 20k plays, 1 epoch; job 13931411, A100). Current = R5. Both scored
on the same eval_lite file (sha `78081c81…`). Artifacts:
[`results/gate_demo/demo.json`](../../results/gate_demo/demo.json),
[`results/gate_demo/registry_index.json`](../../results/gate_demo/registry_index.json).

## What happened

1. **R5 promoted** (first version; 98.53% ≥ the 95% floor, 100% valid).
2. **Broken candidate refused.** It passed the eval-file match, valid-output rate
   (99.96%), and the absolute floor (96.67% ≥ 95%), and failed two checks:
   - overall exact match 96.67% vs current 98.53% (−1.86 points);
   - per-bucket regressions beyond the 1-point tolerance: fumble −12, penalty_stands −12,
     challenge −8, two_point −3 (lateral, n=29, is reported but not gated as too small).
3. **Serving stayed on v0001**; a re-release (v0003) was then rolled back to v0001.

## Why this matters

An "overall accuracy ≥ X" release rule alone would have shipped this adapter: it is
96.7% accurate and almost always produces valid JSON. The per-bucket check is what
catches a model that quietly gets worse on the rare, important cases. That is the same
discipline as the rest of this project — report per bucket, not just the average —
applied to deployment.

## A finding on the side

Training with the prompt mask off (loss on prompt tokens too) costs only ~1.9 points
overall at 20k plays, but 8–12 points on the hard buckets: the model spends capacity
learning to reproduce the (identical) prompt template instead of the answer.
