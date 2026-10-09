# P5b on the H100: multi-LoRA serving, real Llama 3.2 1B

**Job:** 13906092 · PACE H100 80GB HBM3 (inferno) · 2026-10-09 · artifact
[`results/p5b/bench_cuda_real_L16_20261009_041143.json`](../../results/p5b/bench_cuda_real_L16_20261009_041143.json).
Thresholds were fixed in source before the run: L3 ≤ 15% at r=16 batch 32; L4 v2 at
N=256 uniform keeps ≥ 50% of its N=1 goodput; L7 within 2× of vLLM.

## Decode at batch 32 (L3's batch-32 arm)

| Arm | v1 tok/s | v1 overhead | v2 tok/s | v2 overhead |
|---|---|---|---|---|
| base | 418.6 | — | 418.6 | — |
| one shared adapter, r=8 / 16 / 64 | 375.7 / 376.2 / 376.6 | **+11.4 / +11.3 / +11.1%** | 352.5 / 353.2 / 353.2 | +18.7 / +18.5 / +18.5% |
| 32 distinct adapters, r=16 | 133.8 | **+212.9%** | 351.8 | **+19.0%** |

**L3 (batch 32) earned for v1** (11.3% ≤ 15% at r=16); v2 misses it (18.5%). Overhead
is flat in rank for both kernels, the same launch-bound pattern as P5a's batch-1 result.
With 32 distinct adapters in the batch, v1 is 3× slower than base while v2 stays at +19%.

## Linears only, batch 32, r=16 (kernel arm)

v1 grows with the number of distinct adapters (8.6 → 23.9 → 83.0 → 162.2 ms at 1 / 4 /
16 / 32); v2 is flat at ~13.1 ms. Crossover between 1 and 4 distinct adapters.

## Goodput grid (L4): inconclusive on the pre-registered metric

The SLO was anchored on unloaded single-request latency (TTFT ≤ 217 ms, TPOT ≤ 35 ms =
3× unloaded). Under batching, TPOT is ~56 ms for every LoRA arm even with one adapter,
so SLO attainment was ≤ 1.5% in every cell and goodput ≈ 0. All 512 requests completed
in every cell and v2's TTFT stayed ~92 ms, so this is an SLO-calibration flaw (TPOT
target unattainable under any batching), not overload. The pre-registered retention
ratio is degenerate (0.023 / 0.023), so **L4 is reported as inconclusive**, not earned.
The metric was not changed after the fact.

The load-insensitive measurements tell the scaling story:

| Distinct adapters (uniform) | v1 saturation tok/s | v2 saturation tok/s | v1 TTFT p50 | v2 TTFT p50 | pool hit rate |
|---|---|---|---|---|---|
| 1 | 361 | 328 | 65 ms | 92 ms | 100% |
| 4 | 303 | 325 | 113 ms | 93 ms | 99% |
| 16 | 207 | 329 | 13.8 s | 91 ms | 97% |
| 64 | 150 | 325 | 40.8 s | 93 ms | 47% |
| 256 | 133 | 321 | 55.6 s | 98 ms | 10–11% |

v2 keeps 98% of its saturation throughput from 1 to 256 adapters (with a 32-slot pool
and 10% hit rate at N=256, so adapter swapping is included); v1 keeps 37%. Zipf
popularity behaves the same with higher hit rates.

**Fix for a rerun:** anchor the SLO on the loaded base model (or measure goodput at a
TPOT target set from the base model at the same batch load), and pre-register that
before running.

## Capacity (L5)

Measured on the H100: base weights 2.47 GB; adapter slots 11.3 / 22.5 / 90.2 MB at
r = 8 / 16 / 64, and the pool's device allocation matched the formula exactly. On an
80 GB GPU that is ~31 merged copies of the model vs ~3,300 resident r=16 adapters
(weights only).
