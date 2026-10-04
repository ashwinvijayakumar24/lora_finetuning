# P5a: the engine can only quantize weights it loads from disk

**Status:** worked around with a duplicated 10-line loop.

## What happened

The plan for a quantized base with a merged adapter is "merge in full precision,
then quantize once". The engine's only quantizing entry point is
`engine.loader.load_weights_gpu_quant(weights_path, ...)`. It loads the
safetensors file and quantizes in one step, so it cannot take weights that were
merged in memory.

## How it was found

Writing `merge_then_quantize` and looking for the engine function to call.

## Root cause

Loading and quantizing are fused in one function in `engine/loader.py`.

## Fix

`playparse.serving.merge.quantize_engine_weights` repeats the loader's loop. It
uses the engine's own `quantize_int8_perchannel`, `quantize_int4_group`,
`QuantWeight` and `_QUANTIZED_SUFFIXES`, so the quantization math is not
duplicated, only the loop over weight names. `_QUANTIZED_SUFFIXES` is a private
name; if the engine renames it, the import fails loudly rather than drifting.

## Suggested engine change

Split `load_weights_gpu_quant` into `load_weights_gpu` plus a public
`quantize_weights(weights, mode, group_size)`.

## Guarding test

`tests/test_p5a_lora_math.py::test_merge_then_quantize_equals_quantizing_merged_weight`
checks that the result equals quantizing the merged fp16 weight with the engine's
quantizer.
