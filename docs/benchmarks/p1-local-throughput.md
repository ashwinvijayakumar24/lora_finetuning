# P1 local throughput: R1 and R2 on the M4 (MPS)

**Bottom line.** On this Mac, running other agents' jobs at the same time, R1
processed **0.20 plays/sec** at batch size 16, and R2 processed **0.036 plays/sec**
at batch size 4. At those rates, eval_lite (~3,000 plays) would take about **4 hours
for R1** and about **23 hours for R2** (R3 costs about the same as R2). Treat these
as pessimistic: the machine was swapping heavily during the run. The practical plan
is R0 locally, a few-hundred-play R1/R2 smoke locally, and full R1–R3 runs on the
H100.

## Setup

- **Hardware.** Apple M4, 16 GB unified memory, MPS backend, torch 2.12,
  transformers 5.10.1. Model: Llama 3.2 1B Instruct in fp16.
- **Contention.** Several other agents ran PyTorch jobs at the same time (one used
  about 12 GB). macOS swap usage was 15–19 GB during the run. Per-token decode at
  batch size 1 came out near 64 ms per generated token, which is several times slower
  than this model normally runs on an M4. **These numbers are an upper bound on
  time, not a clean measurement.**
- **Data.** 16 plays sampled with a fixed seed from 5 games of season 2021 (train,
  dev only). Labels come from the approximate dev labeler.
- **Decoding.** Greedy, `max_new_tokens = 256`, left padding, after one warm-up
  batch.
- **Command.**
  ```bash
  python scripts/bench_local_throughput.py --data data/cache/dev_2021_sample.jsonl \
      --n 16 --batch-sizes 16 1 --rungs r1 --out results/bench_p1_local_throughput/result.json
  python scripts/bench_local_throughput.py --data data/cache/dev_2021_sample.jsonl \
      --n 16 --batch-sizes 4 1 --rungs r2 --out results/bench_p1_local_throughput/r2.json
  ```
- **Artifacts.** `results/bench_p1_local_throughput/result.json` (R1) and
  `results/bench_p1_local_throughput/r2.json` (R2).

## Results

| Rung | Batch | Plays/sec | p50 batch latency | Input tok/play | Output tok/play | Valid | Exact |
|---|---|---|---|---|---|---|---|
| R1 | 16 | 0.204 | 78.5 s | 232 | 131 | 10/16 | 1/16 |
| R1 | 1 | 0.078 | 8.2 s | 232 | 127 | 9/16 | 1/16 |
| R2 | 4 | 0.036 | 125 s | 1,001 | 78 | 16/16 | 3/16 |
| R2 | 1 | 0.021 | 13.4 s | 1,001 | 78 | 16/16 | 3/16 |

With 16 plays, the quality columns are only a sanity check, not a measurement. They
do agree with expectations: zero-shot R1 often wanders off the schema and pretty-
prints long outputs, two of its 16 hit the 256-token cap, and eight examples make
every R2 output schema-valid.

## Estimated time for eval_lite (~3,000 plays)

| Rung | Configuration | Estimate (contended) |
|---|---|---|
| R0 | CPU | under 1 second |
| R1 | batch 16 | about 4.1 hours |
| R1 | batch 1 | about 10.7 hours |
| R2 / R3 | batch 4 | about 23 hours |
| R2 / R3 | batch 1 | about 40 hours |

## Observations

- **R2 at batch size 16 did not finish.** It ran for more than 35 minutes without
  completing one batch and was stopped. Sixteen prompts of about 1,000 tokens make
  large attention intermediates during prefill, and with the machine already in
  swap that thrashed. Keep R2/R3 at batch size 4 or below on a 16 GB Mac.
- **Batch size changed R1's outputs** (10/16 valid at batch 16, 9/16 at batch 1).
  Batch size is now part of the HF predictor config. See
  `docs/issues/p1-eval-batch-size-changes-greedy-outputs.md`.
- **Prompt length drives R2's cost.** R2's prompt is about 4.3× R1's, but R2's
  outputs are shorter (78 vs 131 tokens), because the examples teach compact JSON.
- **Gold-label length,** measured with the Llama tokenizer over 293,986 train-season
  labels: p50 26, p99 88, max 105 tokens. This is the basis for
  `max_new_tokens = 256`.

## Follow-up

Rerun on an idle machine, or on the H100, with `--n 64` before using these numbers in
the ladder's latency or cost columns.
