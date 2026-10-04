# P5a: the first real-model oracle run thrashed swap on a 16 GB laptop

**Status:** fixed by sharing one fp32 copy of the weights between HF and the engine.

## What happened

The first version of `tests/test_p5a_real_oracle.py` loaded HF Llama 3.2 1B in
fp32 (~5 GB) for the PEFT reference, freed it, then loaded the engine's own fp32
NumPy weights (~5 GB) for the engine side, and finally merged the adapter into a
*copy* of the targeted projections (~4 GB more). With other agents running MPS
jobs on the same machine, the process sat at ~25% CPU and ~250 MB resident for
over 12 minutes while macOS swap grew to 17 GB. It was killed without a result.

## How it was found

Watching `ps` and `sysctl vm.swapusage` while the run made no progress.

## Root cause

Two avoidable full copies:

1. The engine's NumPy weights are bit-identical to the HF fp32 parameters (both
   are the bf16 checkpoint cast to fp32, which is exact), so loading them again
   doubled memory for nothing.
2. `merge_adapter(..., inplace=False)` keeps the original projections alive
   alongside the merged ones.

## Fix

- The fp32 engine runs on zero-copy NumPy views of the HF model's parameters
  (`tensor.numpy()` shares memory). One tensor is compared against the
  safetensors file to prove the values are the engine loader's values.
- The HF model is deleted as soon as its reference logits and tokens are taken.
  The weight dict then holds the only references, so the in-place merge frees
  each original matrix as it replaces it. Peak stays near 5 GB.
- Each fixture computes everything up front and keeps only small results
  (logits for one prompt and token lists); tests assert on those.
- The "zero adapter merged" check runs on the fp16 path and the tiny model
  instead of the fp32 path, which would need a second 4 GB copy.

## Guarding test

The structure of `tests/test_p5a_real_oracle.py` itself (`hf_refs` fixture). The
safetensors probe inside that fixture fails if the shared views ever stop
matching the engine loader's values.
