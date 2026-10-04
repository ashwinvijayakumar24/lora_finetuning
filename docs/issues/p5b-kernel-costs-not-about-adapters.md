# P5b: the first kernels were slow for reasons that had nothing to do with adapters

**Status:** fixed with a per-forward host-side batch plan; guarded by tests.

## What happened

The first local benchmark run (Apple M4, MPS, 1B-shaped model with 2 layers)
showed two costs that did not depend on the number of adapters at all:

1. **v1 (loop) was dominated by host syncs.** At batch 1 with a single adapter,
   the adapted projections took **+486%** over base with v1 and +21% with v2.
   v1 called `torch.unique(row_slots)` and `torch.nonzero(row_slots == s)` inside
   every projection: two device-to-host synchronisations, 7 projections x layers
   times per forward pass. On MPS each sync costs far more than the tiny matmuls
   it guards.
2. **v2 (BGMV) was crushed by prefill.** The open-loop goodput run took minutes
   per cell. BGMV gathers a private copy of `A` and `B` for every row. A decode
   row needs that (each row may be a different adapter), but a 512-token prefill
   chunk is 512 rows that all share one adapter: the gather moved
   `512 x rank x (in + out)` values (about 170 MB for one down_proj at r=16, fp16)
   to do the work of a single matmul.

A third, smaller one: slicing `B[slot, :, :r]` for an adapter whose rank is
below the pool's `max_rank` gives a strided tensor, and MPS printed "MPS mm
implementation has a known issue with this shape, dtype and slice.
Dispatching to metal implementation instead", a slower path.

## How it was found

The first `scripts/p5b_bench.py` run: the kernel arm's numbers for (1), a
goodput run that looked hung for (2) (a `faulthandler` dump showed it inside
`lora_delta_bgmv`, making progress), and a warning in the real-1B gate for (3).

## Root cause

Both kernels rediscovered, on the device and once per projection, structure the
scheduler already knew in Python: which sequence uses which slot and how many
tokens each contributes.

## Fix

`MultiLoRAModelGPU.selection_for` builds a `BatchPlan` once per forward pass
from the scheduler's per-sequence slots and query lengths:

- `groups`: rows per distinct slot (v1 loops over these; no syncs);
- `decode_rows` / `decode_slots`: one row per decoding sequence (v2 runs BGMV
  over just these);
- `segments`: `(start, end, slot)` per prefill chunk (v2 runs one ordinary
  matmul pair per segment, the SGMV idea from Punica).

v1 also makes the rank-sliced `B` contiguous, which is P5a's layout and avoids
the MPS fallback.

## Why it matters

Without the fix, the L4 comparison would have measured "host syncs vs no host
syncs" and "prefill weight-gather vs none", not "loop over adapters vs batched
gather". Both artifacts would also have shown up on CUDA, only smaller.

## Guarding tests

- `tests/test_p5b_pool.py::test_planned_kernels_match_unplanned`: the planned
  v1 is bit-identical to the unplanned one, the planned v2 matches it within fp16
  noise, prefill segments in v2 are bit-identical to v1, and base rows are exact
  zeros.
- `tests/test_p5b_pool.py::test_plan_with_only_base_rows_skips_the_kernel`
- The scheduler gate (`tests/test_p5b_scheduler.py`) runs entirely on the
  planned path.
