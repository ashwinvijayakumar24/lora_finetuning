# P3: scoring an adapter with the wrong prompt style fails silently

## What happened

With two prompt styles (`full` and `minimal`), an adapter can be evaluated or
served with a style other than the one it was trained with. Nothing in the
adapter directory recorded the style, and the eval CLI defaults to `full`. Running
`python -m playparse.eval.run --rung lora --adapter <minimal adapter>` without
`--prompt-style minimal` would have loaded the adapter, generated, and written a
result.

## Why it matters

The output would not crash or look broken. It would be valid JSON at a somewhat
lower exact match, which reads like a weaker adapter rather than a wrong command.
A sweep that mixed styles could draw a false conclusion about a hyperparameter.
The config hash records the style, so the mistake is visible afterwards, but only
to someone who checks.

## How it was found

While adding `--prompt-style` for the prompt ablation, before running the minimal
adapter's eval.

## Fix

`LoRAPredictor` now looks for the training run's `run_spec.json` (next to the
adapter, or at `<output_dir>` above `checkpoints/step_N/adapter`) and raises if its
`data.prompt_style` differs from the requested style. A spec without the field
predates the option and means `full`, so every earlier adapter still loads with
the default. An adapter copied out of its run directory is not checked; serving
code that copies adapters should carry the style along.
Test: `tests/test_prompt_style.py::test_lora_predictor_refuses_mismatched_style`.
