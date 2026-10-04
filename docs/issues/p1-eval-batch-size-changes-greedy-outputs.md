# p1-eval: R1 greedy outputs change with batch size

## What happened

In the local throughput benchmark, R1 ran on the same 16 train plays twice: once at
batch size 16 and once at batch size 1. Valid-output rate was 62.5% at batch size 16
and 56.3% at batch size 1. Greedy decoding is deterministic for a *fixed* batch (the
slow smoke test checks this), but the outputs differ *between* batch sizes.

## Why it matters

- **Comparability.** Two runs of "the same rung" at different batch sizes are not the
  same system at the token level. Batch size is a serving choice, but here it leaks
  into quality numbers.
- **Resume safety.** Batch size was not part of `HFPredictor.config()`. So a run
  interrupted at batch size 16 and resumed at batch size 8 would have mixed two sets
  of outputs under one config hash, without any warning.

## How it was found

Comparing the `valid_rate` fields of the batch-16 and batch-1 rows in
`results/bench_p1_local_throughput/result.json`.

## Root cause (likely; not yet isolated)

Generation uses left padding and fp16 on MPS. Different batch shapes run different
kernel tilings and reduction orders, so the logits differ by rounding error. When the
top two tokens are close, greedy argmax flips, and the rest of the generation
diverges from there. Position ids are derived from the attention mask by
`generate`, so a padding-position bug is less likely. A CPU fp32 comparison would
confirm the explanation. That is a follow-up.

## Fix

`batch_size` is now in `HFPredictor.config()`. It is therefore part of the config
hash, recorded in every result file, and a resume with a different batch size is
refused. Report the batch size next to every R1–R3 number, and compare rungs at the
same batch size.

## Guarding test

`tests/test_eval_harness.py::test_resume_refuses_mixed_configs` covers the refusal
mechanism. `tests/test_eval_baselines.py::test_hf_config_includes_batch_size` checks
that batch size is in the config without loading the model.
