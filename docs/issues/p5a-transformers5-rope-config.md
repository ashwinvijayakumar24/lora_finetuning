# P5a: transformers 5 writes a config.json the engine cannot read

**Status:** worked around in the test helper; affects any checkpoint saved with transformers ≥ 5.

## What happened

The fast oracle tests build a tiny random Llama with `transformers` and save it
with `save_pretrained`, so the engine can load it from disk like the real model.
With transformers 5.10.1 the saved `config.json` has a `rope_parameters` block
and no top-level `rope_theta` or `rope_scaling` keys. The engine reads
`config["rope_theta"]` (`engine/model.py`, `engine/model_gpu.py`) and
`config.get("rope_scaling")`, so it would fail with a `KeyError`. If only
`rope_theta` had been present, it would have silently dropped the Llama 3 RoPE
frequency scaling, which changes every attention score.

The real Llama 3.2 1B `config.json` was written by transformers 4.45 and still
has the old keys, so the engine itself is unaffected today.

## How it was found

Printing the saved config while probing PEFT's on-disk format, before writing
the tiny-model fixture.

## Root cause

transformers 5 renamed and nested the RoPE configuration. The engine predates
that change.

## Fix

`playparse/serving/_testing.py::build_tiny_checkpoint` writes `rope_theta` and
`rope_scaling` back into `config.json` after saving.

## Why it matters later

A merged PlayParse model saved with `save_pretrained` under transformers 5 (for
example to serve it with vLLM or to keep it as a checkpoint) will have the same
problem if anyone points the engine at it. Any such export should copy the base
model's `config.json`, or apply the same patch.

## Guarding test

`tests/test_p5a_tiny_oracle.py::test_tiny_checkpoint_config_is_engine_readable`
