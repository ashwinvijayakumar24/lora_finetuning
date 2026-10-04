# P5a local benchmark — Apple M4, MPS (indicative only)

> **Read this first.** These numbers come from a 16 GB Apple M4 laptop that was
> running several other agents' training and benchmark jobs at the same time
> (macOS swap reached 17–24 GB during the session). The run-to-run noise is larger
> than every effect being measured. Nothing on this page earns or refutes L1 or
> L3. It shows that the benchmark script runs end to end and produces the
> artifact the H100 run will produce. The authoritative run is
> `scripts/slurm/p5a_bench.sbatch`.

**Artifact:** `results/p5a/bench_mps_20261004_151553.json` (git `c92f9b9`, working
tree dirty only with docs and this script; engine checkout as found).
**Command:** `python scripts/p5a_bench.py --device mps --max-tokens 64 --n-runs 3 --micro`
**Settings:** batch 1, greedy, 64 new tokens, the engine harness's three prompts
(chat-formatted), 1 warmup + 3 measured runs per prompt, all arms interleaved
round-robin. Adapters are synthetic, on all seven projections, r = 8/16/64.

## End to end, decode tokens per second (median of 3 runs per prompt)

| Arm | short | medium | long |
|---|---|---|---|
| base | 25.7 | 18.4 | 27.9 |
| base_repeat (same model again) | 27.4 | 17.4 | 25.4 |
| merged_r16 | 28.8 | 16.8 | 28.4 |
| base_patched (hook installed, no adapter) | 28.9 | 17.4 | 27.7 |
| unmerged_r8 | 27.0 | 16.3 | 26.0 |
| unmerged_r16 | 22.5 | 15.7 | 24.0 |
| unmerged_r64 | 24.5 | 15.6 | 27.5 |

How to read it:

- **The noise floor is about ±10%.** `base` and `base_repeat` are the *same model*
  measured in alternation, and they differ by up to 10% on one prompt. Across all
  nine runs per arm, every arm's range spans roughly 15–30 tok/s.
- **The "medium" column is slower for every arm**, including base. That is the
  machine, not the prompt: those runs happened while other jobs were busiest.
  Interleaving the arms is what keeps this drift from landing on one arm.
- **L1 (merged = base):** merged sits inside the base/base_repeat spread on every
  prompt. Consistent with "no added latency", which merged mode guarantees by
  construction (same tensors, same kernels), but this run cannot resolve a small
  difference.
- **L3 (unmerged overhead):** unmerged arms are 0–15% slower than base on most
  prompts, which is the same size as the noise. No rank trend is resolvable.
- **Merged and unmerged r=16 produced identical greedy tokens on all three
  prompts** under benchmark settings (`merged_unmerged_greedy_identical: true`),
  a cross-check of L2.

TTFT is in the JSON but not tabled here: with a ~70–90 ms prefill and the
contention above, single TTFT samples swung by 5× and carry no signal.

## Microbenchmark: the 112 projections of one decode step

This times only `linear()` over all seven projections in all 16 layers, base vs
base plus the low-rank path, with the device synchronised before and after 50
iterations. Batch 32 here means 32 token rows through each projection; it is a
preview of what P5b's batched decode will pay, not an end-to-end number.

| Batch | base | r=8 | r=16 | r=64 |
|---|---|---|---|---|
| 1 | 19.6 ms | 20.3 ms (+3.7%) | 20.0 ms (+2.3%) | 20.7 ms (+5.7%) |
| 32 | 31.9 ms | 38.3 ms (+20%) | 49.6 ms (+56%) | 45.2 ms (+42%) |

- At batch 1 the low-rank path adds 2–6%, below the end-to-end noise.
- At batch 32 the extra cost is much larger, and it is not monotone in rank
  (r=16 slower than r=64), which says this measurement is also dominated by
  contention. Larger overhead at batch 32 is plausible regardless: the base matmul
  gets more efficient per row as the batch grows, while the low-rank path is
  three small, poorly-utilised kernels. That is exactly the question P5b's
  batched kernels (and the H100 run) answer.

## What the H100 run will add

Same script, `--device cuda --max-tokens 128 --n-runs 5`, one allocation, plus
the CUDA oracle arm. L3's threshold is fixed in the script before that run:
**≤ 15% decode overhead at r = 16, batch 1**.
