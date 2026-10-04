# P5b local benchmark — Apple M4, MPS (indicative only)

> **Read this first.** These numbers come from a 16 GB Apple M4 laptop while
> another agent was training a model on the same GPU (`scripts/train.py`, MPS)
> for the whole session. Run-to-run noise is larger than most of the effects
> being measured, and the model is a 1B-shaped random model with **4 of 16
> layers** to fit the run in memory and time. Nothing on this page earns or
> refutes L3, L4 or L7. It shows that every arm of the benchmark runs end to end
> and produces the artifact the H100 run will produce, and it shows the few
> effects large enough to survive the noise. The authoritative runs are
> `scripts/slurm/p5b_bench.sbatch` and `scripts/slurm/p5b_vllm.sbatch`.

**Artifacts** (both carry git SHAs of this repo, the engine and the serving layer):

- `results/p5b/bench_mps_synthetic_L4_20261004_233847.json` — kernel, decode32,
  capacity (git `d6a408c`, clean tree). Command:
  `python scripts/p5b_bench.py --quick --layers 4 --arms kernel,decode32,capacity`
- `results/p5b/bench_mps_synthetic_L4_20261004_233638.json` — goodput (git
  `ca37777`, clean tree; the goodput arm is unchanged since). Its kernel and
  decode32 sections come from the earlier non-interleaved version of those arms
  and are superseded by the file above. Command:
  `python scripts/p5b_bench.py --quick --layers 4`
- `results/p5b/gate_real_mps_L16.json` — the real-1B correctness gate (not a
  timing result).

Re-render any of them with `python scripts/p5b_bench.py --render <file>`.

## What each arm measures

- **kernel** times only the 28 adapted projections of one forward pass (7 per
  layer x 4 layers) on a decode-shaped batch (one row per sequence), base vs
  base + the low-rank term. Each configuration is timed in 5 interleaved rounds
  (base, v1, v2 back to back) and the minimum per-round mean is reported:
  contention only ever adds time, so the minimum is the best estimate of the
  uncontended cost.
- **decode32** times whole decode steps through the `AdapterScheduler` with 32
  sequences decoding together, interleaving all arms step by step so that drift
  lands on every arm alike.
- **goodput** replays a seeded open-loop workload (Poisson arrivals, lognormal
  prompt and output lengths, tenant per request uniform or Zipf s=1.1) against a
  pool of 8 slots. The SLO is frozen before any loaded run at 10x the unloaded
  TTFT and 3x the unloaded time per output token of the base model (the serving
  layer's multipliers). Offered load is 0.5x the measured base capacity. Each
  cell also replays the same requests all at once (saturation) as a capacity
  measure that does not depend on where the offered rate sits.
- **capacity** measures the bytes of one base copy, one adapter slot per rank
  and one KV token, scales them to the full 16-layer model, and counts tenants
  on an 80 GB GPU (90% usable).

## Kernel arm (L4 kernel, L3 preview)

### kernel (adapted projections of one forward pass, min over rounds)

| batch | rank | distinct | base ms | v1 ms (overhead) | v2 ms (overhead) |
|---|---|---|---|---|---|
| 1 | 8 | 1 | 6.70 | 7.20 (+7%) | 7.82 (+17%) |
| 1 | 16 | 1 | 10.11 | 11.75 (+16%) | 13.22 (+31%) |
| 1 | 64 | 1 | 10.23 | 9.36 (-8%) | 12.85 (+26%) |
| 32 | 8 | 1 | 14.48 | 15.30 (+6%) | 42.43 (+193%) |
| 32 | 8 | 4 | 8.12 | 15.39 (+90%) | 23.85 (+194%) |
| 32 | 8 | 16 | 8.28 | 41.44 (+400%) | 22.61 (+173%) |
| 32 | 8 | 32 | 8.14 | 127.60 (+1467%) | 23.99 (+195%) |
| 32 | 16 | 1 | 8.59 | 11.89 (+38%) | 54.11 (+530%) |
| 32 | 16 | 4 | 8.39 | 14.90 (+78%) | 33.02 (+294%) |
| 32 | 16 | 16 | 8.09 | 71.40 (+783%) | 33.15 (+310%) |
| 32 | 16 | 32 | 8.35 | 120.20 (+1339%) | 33.29 (+299%) |
| 32 | 64 | 1 | 7.64 | 10.87 (+42%) | 95.65 (+1152%) |
| 32 | 64 | 4 | 7.72 | 13.99 (+81%) | 94.34 (+1122%) |
| 32 | 64 | 16 | 7.59 | 38.58 (+409%) | 90.79 (+1097%) |
| 32 | 64 | 32 | 7.60 | 75.40 (+892%) | 90.06 (+1085%) |

How to read it:

- **v1's cost grows with the number of distinct adapters; v2's does not.** At
  batch 32 and r=16, v1 goes from 11.9 ms (1 adapter) to 120 ms (32 adapters);
  v2 stays at 33 ms whatever the mix. This is the one effect in this file large
  enough to be unambiguous, and it is the mechanism L4 is about.
- **The crossover is between 4 and 16 distinct adapters** at r=8 and r=16 on
  this machine. With one or a few adapters in the batch, v1 is much cheaper,
  because v2 gathers a private copy of the weights for every row whether or not
  rows share an adapter.
- **v2's cost scales with `max_rank`** (24 / 33 / 90 ms at r = 8 / 16 / 64): the
  gather copies `rows x max_rank x (in + out)` values. At r=64 it never beats v1
  here. A fused BGMV kernel (v3) reads the stacks in place and removes exactly
  this cost; the MPS gather is also likely to be relatively slower than on CUDA.
- **Batch 1 overhead** is 7–31%, inside the noise of a shared laptop (the base
  column itself moves between 6.7 and 10.2 ms across rows that should be equal).

## decode32 arm (L3 batch-32 arm, L4 end to end)

### decode32 (32 sequences decoding, interleaved rounds)

| arm | kernel | ms/step p50 | ms/step min | tok/s | overhead vs base (p50) |
|---|---|---|---|---|---|
| base | - | 91.2 | 84.1 | 351 | - |
| shared_r8 | v1 | 88.8 | 77.0 | 360 | -2.7% |
| shared_r16 | v1 | 82.6 | 77.7 | 387 | -9.4% |
| shared_r64 | v1 | 83.3 | 79.3 | 384 | -8.6% |
| distinct32_r16 | v1 | 156.5 | 143.6 | 204 | +71.6% |
| shared_r8 | v2 | 108.7 | 91.0 | 294 | +19.2% |
| shared_r16 | v2 | 104.5 | 99.8 | 306 | +14.6% |
| shared_r64 | v2 | 179.4 | 163.9 | 178 | +96.7% |
| distinct32_r16 | v2 | 118.2 | 110.5 | 271 | +29.5% |

- **L3 (one shared adapter, batch 32):** with v1 the unmerged arms are within
  ±10% of base, on both sides, so the overhead is below this machine's
  resolution. With v2 they are 15–97% slower, consistent with the kernel arm: v2
  pays for a per-row gather even when every row uses the same adapter. On a
  real deployment the one-adapter case should use the loop path.
- **L4 end to end (32 distinct adapters):** v2 is 25% faster than v1 per decode
  step (118 vs 156 ms). Attention in `PagedTorchBackend` (a Python loop over
  sequences) dilutes the kernel difference here; on GPU with FlashInfer the
  LoRA share of a step will be larger.

## Goodput arm (L4)

### goodput (SLO TTFT <= 1195 ms, TPOT <= 111.1 ms; offered 1.36 req/s = load factor x base capacity 2.72)

| N | popularity | kernel | goodput req/s | attainment | TTFT p50 / p99 ms | pool hit rate | loads | saturation tok/s |
|---|---|---|---|---|---|---|---|---|
| 1 | uniform | v1 | 1.08 | 67% | 193 / 2550 | 98% | 1 | 35.2 |
| 1 | uniform | v2 | 0.91 | 56% | 239 / 1012 | 98% | 1 | 25.3 |
| 4 | uniform | v1 | 0.91 | 56% | 258 / 1483 | 92% | 4 | 33.5 |
| 4 | uniform | v2 | 0.74 | 46% | 260 / 1774 | 92% | 4 | 28.0 |
| 4 | zipf | v1 | 0.71 | 44% | 324 / 1222 | 92% | 4 | 47.2 |
| 4 | zipf | v2 | 1.18 | 73% | 275 / 2483 | 92% | 4 | 33.2 |
| 16 | uniform | v1 | 0.64 | 40% | 596 / 1864 | 52% | 23 | 33.5 |
| 16 | uniform | v2 | 1.04 | 65% | 242 / 1598 | 50% | 24 | 31.7 |
| 16 | zipf | v1 | 0.74 | 46% | 391 / 2225 | 71% | 14 | 31.4 |
| 16 | zipf | v2 | 1.08 | 67% | 323 / 3544 | 71% | 14 | 38.4 |
| 64 | uniform | v1 | 0.88 | 54% | 344 / 2489 | 2% | 47 | 39.7 |
| 64 | uniform | v2 | 0.81 | 50% | 370 / 1510 | 2% | 47 | 40.6 |
| 64 | zipf | v1 | 0.44 | 27% | 549 / 2793 | 52% | 23 | 37.3 |
| 64 | zipf | v2 | 0.94 | 58% | 329 / 4401 | 52% | 23 | 31.7 |
| 256 | uniform | v1 | 0.00 | 0% | 10755 / 15663 | 0% | 48 | 53.0 |
| 256 | uniform | v2 | 1.62 | 100% | 126 / 258 | 0% | 48 | 68.6 |
| 256 | zipf | v1 | 1.62 | 100% | 139 / 279 | 42% | 28 | 67.3 |
| 256 | zipf | v2 | 1.62 | 100% | 104 / 204 | 42% | 28 | 67.3 |

This arm ran for about 25 minutes while the other agent's training job was
active, and the machine's speed drifted a lot during it: the closed-loop
saturation throughput of the *same kind* of cell rose from about 25–35 tok/s at
the start (N=1) to 53–69 tok/s at the end (N=256), which is the machine getting
faster, not more adapters helping. **Goodput across cells is therefore not
comparable, and no N-scaling conclusion can be drawn from it.** What the run
does show:

- every cell of the full grid (N = 1/4/16/64/256 x uniform/Zipf x v1/v2) runs
  end to end through the scheduler, the pool and the SLO scoring, with 48
  requests per cell and an 8-slot pool;
- **pool behaviour is a property of the workload, and it is as expected.**
  Uniform popularity over 64 or 256 tenants gives a 0–2% hit rate (almost every
  request loads its adapter); Zipf s=1.1 keeps 42–52% hits at the same N. Each
  load of a 4-layer r=16 adapter (5.6 MB) took 14–245 ms here, noisy, all of it
  host-to-device copying on a contended unified-memory machine;
- **one cell shows the predicted failure mode, once:** N=256, uniform, v1
  collapsed (0% within SLO, TTFT p99 15.7 s, 81 admission waits), while v2 on the
  identical request stream met the SLO for every request with zero waits. The
  mechanism is consistent with the PRD's prediction that "L4 with the v1 kernel
  at large N under uniform popularity" is the most likely not-earned result:
  slower steps keep more requests in flight, every in-flight request pins a
  slot, new tenants then wait at the head of the queue, and queueing runs away.
  It is a single sample on a noisy machine and is reported as a lead for the
  H100 run, not a result.

## Capacity arm (L5)

### capacity (full 16-layer 1B, 80 GB x 0.9 usable)

base copy 2.30 GiB; adapter slot r=8: 10.8 MiB, r=16: 21.5 MiB, r=64: 86.0 MiB; KV 32 KiB/token

| KV tokens per tenant | merged copies | pool, r=8 | pool, r=16 | pool, r=64 |
|---|---|---|---|---|
| 0 | 31 | 6639 | 3319 | 829 |
| 2048 | 30 | 954 | 834 | 475 |
| 8192 | 28 | 267 | 257 | 208 |

The bytes are measured (the pool's allocation on MPS matched the formula
`n_slots+1` x per-slot bytes exactly) and scaled from 4 to 16 layers; the
tenant counts are arithmetic on those bytes for an 80 GB GPU with 10% held back.

- **Weights only, the pool holds about 100x more tenants than merged copies**
  (3,319 r=16 adapters all resident, against 31 full copies).
- **KV cache is what actually limits tenants.** If every tenant is given 2,048
  tokens of KV of its own, the pool's advantage shrinks to about 28x at r=16
  (834 vs 30), and at 8,192 tokens to about 9x. The fair comparison depends on
  how much concurrent context each tenant needs, so the table states it.
- "All resident" is the conservative count. With host offload, the number of
  *registered* tenants is bounded by host RAM, and only the active set must fit
  in GPU slots; that is what the goodput arm exercises with 256 tenants and 8
  slots.

