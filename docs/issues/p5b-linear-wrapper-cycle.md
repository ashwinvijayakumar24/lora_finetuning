# P5b: two `linear()` wrappers re-installed in turn called each other forever

**Status:** fixed; guarded by a test.

## What happened

P5a replaces the engine's `components_gpu.linear` with `lora_linear` (for
`LoRALinear` weights). P5b adds a second wrapper, `multi_lora_linear` (for
`PooledLoRALinear` weights). Each wrapper remembers the function it replaced
and delegates to it for weights it does not own, so they stack:

```
components_gpu.linear -> multi_lora_linear -> lora_linear -> engine linear
```

Both installers were "idempotent" by checking only the top of the stack
(`if components_gpu.linear is my_wrapper: return`). In the test suite the P5a
reference model and the P5b pooled model are built in the same process, in
varying order. The sequence

1. P5a installs `lora_linear` (delegate: the engine's `linear`),
2. P5b installs `multi_lora_linear` (delegate: `lora_linear`),
3. something calls P5a's installer again (every `LoRAModelGPU()` does),

made step 3 see `multi_lora_linear` on top, conclude it was not installed, and
install `lora_linear` again with `multi_lora_linear` as its delegate. Now
`lora_linear -> multi_lora_linear -> lora_linear -> ...`, and the first forward
pass on a plain weight died with `RecursionError`.

## How it was found

The first run of `tests/test_p5b_pool.py`. The composition test failed with a
`RecursionError` only when it ran after the module fixture had built both kinds
of model.

## Root cause

"Is my wrapper installed?" was answered by looking at one link of a chain.

## Fix

Each wrapper now exposes `next_linear()`, the function it delegates to, and
`lora_engine.linear_chain_contains(fn)` walks the whole stack. Both installers
return early if their wrapper is anywhere in it. This is a small change to the
P5a module (`playparse/serving/lora_engine.py`); its public API is unchanged.

The suggested engine change from
[p5a-linear-no-module-identity](p5a-linear-no-module-identity.md) (an explicit
hook in `linear()` instead of replacing a module global) would remove this
whole class of problem.

## Guarding test

`tests/test_p5b_pool.py::test_install_composes_with_p5a_and_preserves_plain_weights`
alternates the two installers five times and checks after each that a plain
weight still gives exactly `x @ w.T`. It fails with `RecursionError` on the old
installers.
