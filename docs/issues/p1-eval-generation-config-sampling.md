# p1-eval: the checkpoint's generation config samples by default

## What happened

`generation_config.json` in the Llama 3.2 1B Instruct checkpoint sets
`do_sample: true, temperature: 0.6, top_p: 0.9`. A plain `model.generate(...)` call
therefore samples, so outputs change from run to run.

## Why it matters

PRD §6 requires greedy decoding for every model rung. Sampled outputs would make
R1–R3 numbers irreproducible, and the CIs would mix sampling noise with real
differences between rungs.

## How it was found

Reading the checkpoint's `generation_config.json` while writing the predictor.

## Fix

`HFPredictor.predict_batch` passes `do_sample=False, temperature=None, top_p=None`
explicitly, so the checkpoint defaults cannot leak in. The decoding mode is recorded
in the config as `"decoding": "greedy"`.

## Guarding test

`tests/test_hf_predictor.py::test_r1_smoke_on_mps` (slow) runs the same batch twice
and asserts identical text.
