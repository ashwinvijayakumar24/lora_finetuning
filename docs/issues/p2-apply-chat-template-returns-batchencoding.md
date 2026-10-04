# P2: `apply_chat_template(tokenize=True)` returns a dict, not a list

## What happened

In transformers 5.10, `tokenizer.apply_chat_template(messages, tokenize=True)`
returns a `BatchEncoding` (a dict with `input_ids` and `attention_mask`). Older
versions returned a plain list of token ids.

## Why it matters

Code written against the old behaviour keeps running but does the wrong thing.
For example, `len(ids)` returns 2 (the number of dict keys) instead of the
number of tokens. `ids + completion_ids` raises, which is at least loud, but
a length check would silently pass.

## How it was found

Inspecting the tokenizer before writing `collate.py`.

## Fix

PlayParse never tokenizes through `apply_chat_template`. `encode_prompt` renders
the template to a string (`tokenize=False`) and then tokenizes the string with
`add_special_tokens=False`. One code path, the same result on every version.

## Guarding test

`tests/test_collate.py::test_prompt_matches_template_tokenization` compares our
ids to the template's own tokenization, accepting either return type.
