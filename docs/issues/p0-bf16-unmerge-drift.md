# P0: merge then unmerge on a bf16 base does not restore the weight exactly

## What happened

On an fp32 base, `merge()` followed by `unmerge()` gives back the original weight
to about 1e-7. On a bf16 base it does not. For a 2048 x 2048 bf16 layer with an
r=16 adapter (B ~ N(0, 0.02)), about 8.7% of the weights ended up different from
the original after a merge/unmerge round trip. The largest change was 1.2e-4,
against a mean absolute weight of 1.1e-2.

## How it was found

While writing `merge()`, by reasoning about where rounding happens, and then
confirmed with a probe script and a test.

## Root cause

bf16 has an 8-bit mantissa, so a weight near 0.01 can only move in steps of
roughly 6e-5. `merge()` computes `W + delta` in fp32 and rounds the result back to
bf16. `unmerge()` computes `(rounded W) - delta` and rounds again. The two
roundings do not cancel, so some weights land one bf16 step away from where they
started. This is plain floating-point arithmetic, not a bug in the layer.

## Why it matters

Serving code (P5a/P5b) may want to switch adapters on one base model. Doing that
with repeated merge/unmerge on a bf16 base slowly corrupts the base weights, one
rounding step at a time. The safe patterns are:

- keep the adapter unmerged (the extra matmul path) when adapters change often,
- or keep a pristine copy of the base weights and restore from it instead of
  calling `unmerge()`.

## Fix

No code change: the behaviour is inherent. `LoRALinear.merge()` documents it, and
the guarding test pins the size of the effect so a future change that makes it
worse (for example, merging in bf16 instead of fp32) is caught.

## Guarding test

`tests/test_lora_linear.py::test_bf16_unmerge_is_not_bit_exact` asserts that the
drift is non-zero but stays below 1e-2.
