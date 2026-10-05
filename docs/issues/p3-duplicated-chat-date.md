# P3: the pinned chat-template date was defined twice

## What happened

The pinned date that keeps the Llama chat template from writing today's date into
the prompt (see `p2-chat-template-date` and `p1-eval-chat-template-date`) existed
as two separate constants: `DEFAULT_DATE_STRING` in `playparse/train/collate.py`
and `CHAT_DATE_STRING` in `playparse/eval/baselines/hf_predictor.py`. Both held
`"26 Jul 2024"`.

## Why it matters

The two phases fixed the same bug independently. As long as the values agree,
nothing is wrong. If either were ever edited, every trained adapter would be
evaluated on a prompt it was never trained on. Nothing would fail: scores would
just be a little lower, and the cause would be very hard to find.

## How it was found

Flagged during P3 integration planning, when the eval harness had to start
scoring trained adapters.

## Root cause

P1 (eval) and P2 (training) were built in parallel branches, and each needed the
pin before the other had merged.

## Fix

One `CHAT_DATE_STRING` now lives in `playparse/prompt.py`, next to
`SYSTEM_PROMPT` (which is unchanged). Collation, generation, and the HF predictor
import it. `hf_predictor.render_record` exposes the harness's prompt text without
loading a model.

## Guarding tests

`tests/test_prompt_parity.py`:

- `test_single_date_constant_shared_by_train_and_eval`: both sides reference the
  same object, and every `date_string` default is that object.
- `test_train_and_eval_prompt_token_ids_identical`: for every fixture record,
  `encode_prompt` (training) and the harness rendering give byte-identical ids.
- `test_batched_left_padded_eval_ids_match_training_ids`: the same holds after
  the harness's left-padded batch tokenization, once pads are removed.
