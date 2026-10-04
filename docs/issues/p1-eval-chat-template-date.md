# p1-eval: the Llama 3.2 chat template puts today's date in every prompt

## What happened

The Llama 3.2 Instruct chat template writes `Today Date: <date>` into the system
header. When `date_string` is not passed, it calls `strftime_now("%d %b %Y")`, and
transformers provides that function. So the rendered prompt for R1–R3 changes every
calendar day.

## Why it matters

The eval promises that the same config gives the same numbers. With a drifting date,
an R2 run on Monday and a rerun on Tuesday feed the model different tokens. Greedy
decoding can then produce different outputs, and the config hash would not notice,
because the date never appears in the config. It would also mean an adapter is
evaluated on a slightly different prompt than it was trained on.

## How it was found

Reading the checkpoint's `tokenizer_config.json` chat template while wiring up the
HF predictor, before any model run.

## Root cause

The template's fallback (`"26 Jul 2024"`) only applies when `strftime_now` is
undefined, and transformers defines it.

## Fix

`playparse/eval/baselines/hf_predictor.py` renders every prompt through
`render_chat`, which passes `date_string=CHAT_DATE_STRING` (`"26 Jul 2024"`, the
template's own fallback). The value is also recorded in the predictor config, so it
is part of the config hash. **The P2 training collator must use the same function or
the same constant**, or adapters will be trained on a different prompt than they are
evaluated on.

## Guarding test

`tests/test_hf_predictor.py::test_chat_date_is_pinned` (runs whenever the tokenizer
files are present; no model load).
