# The first PACE gate ran the serving tests on the CPU, and v1 diverged there

**Phase:** P5b / PACE · **Severity:** test-infrastructure bug (fixed) plus one open numerics question ·
**Found by:** the first PACE gate job (13901868, L40S node), 2026-10-08.

## What happened

The gate failed 2 of 44 real-model tests, so none of the 49 dependent jobs ran (by design).

1. `tests/test_p5b_real.py` picked its device as "MPS if available, else CPU", so on a
   CUDA node the real-model serving test ran on the **CPU in fp16**. There the v1 (loop)
   kernel's batched output diverged from the single-request reference at the second
   generated token; v2 matched. The same test passes on the Mac's ARM CPU (re-run
   locally with `PLAYPARSE_TEST_DEVICE=cpu`) and on MPS.
2. `tests/test_p5a_real_oracle.py::test_adapter_is_not_vacuous` required a random
   adapter to change one 16-token greedy continuation. On PACE's x86 CPU it did not,
   although the adapter moved the logits by more than 0.5.

## Root cause

1. Device selection did not consider CUDA. The x86 fp16 divergence is most likely a
   near-tie flipped by a different accumulation order (batched vs single-request matmul
   shapes on x86 fp16 kernels), but this has **not** been verified by inspecting the
   logit margin at the divergence point. It stays open.
2. A single greedy string is a fragile signal of "the adapter does something".

## Fix

1. CUDA is preferred when present (`PLAYPARSE_TEST_DEVICE` still overrides), so the
   gate now tests the path that actually serves.
2. The sanity test now requires the top token to change at one or more of the 63
   teacher-forced prompt positions, alongside the logit-difference check.

The gate now prints a per-test summary (`-rfEs`) and, after the tests, trains for 40
steps and scores the result on 64 eval plays, so the whole train → eval path is proven
on CUDA before any real job starts.

## Open

Confirm the x86 CPU fp16 v1 divergence is a near-tie (print the top-2 margin at the
first differing step). It does not affect CUDA serving, which is what the claims cover.
