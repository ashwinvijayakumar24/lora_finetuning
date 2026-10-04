# P2: the Llama 3.2 chat template writes today's date into every prompt

## What happened

The rendered training prompt contained the line `Today Date: 04 Oct 2026` inside
the system header. That line comes from the chat template, not from
`playparse/prompt.py`. The template fills it with the current date every time it
runs.

## Why it matters

A fine-tuned adapter should be queried with exactly the prompt it was trained on.
With the date left in, the prompt changes every day. A model trained on Monday and
evaluated on Tuesday sees a slightly different prompt. The difference is small
(a few tokens), but it is a train/serve mismatch that nobody chose, and it also
makes token counts and cached results depend on the calendar.

## How it was found

While inspecting the rendered prompt string for the BOS (beginning-of-sequence
token) check in `tests/test_collate.py`, the date line stood out.

## Root cause

The Llama 3.x template starts with:

```jinja
{%- if not date_string is defined %}
    {%- if strftime_now is defined %}
        {%- set date_string = strftime_now("%d %b %Y") %}
    {%- else %}
        {%- set date_string = "26 Jul 2024" %}
```

Recent versions of `transformers` provide `strftime_now`, so the template takes
the "today" branch.

## Fix

`playparse/train/collate.py` always passes `date_string=DEFAULT_DATE_STRING`
(`"26 Jul 2024"`, the template's own fallback) to `apply_chat_template`. Every
prompt for training, generation, and evaluation goes through `render_prompt` /
`encode_prompt`, so the pin applies everywhere.

**Action for other phases:** any code that renders a prompt on its own (the
baselines, serving, the eval harness) must use `encode_prompt` or pass the same
`date_string`. Otherwise it will query the adapter with a different prompt.

## Guarding test

`tests/test_collate.py::test_prompt_is_date_independent` shows that two
different dates give two different prompts, and that `render_prompt` always
contains the pinned date.
