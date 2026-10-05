# P3: "no system message" still renders a dated system header

## What happened

The prompt ablation's `minimal` style sends only the user turn, with no system
message. The Llama 3.2 chat template does not then omit the system block. It
writes one anyway:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

posteam: PHI
desc: ...<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

So a "minimal" prompt is 35 fixed tokens plus the play, not just the play. Passing
an empty system message renders exactly the same text.

## Why it matters

Two consequences, one a trap and one a limit.

- **The date pin is still load-bearing.** It would be natural to assume that with
  no system message there is no date, and to stop passing `date_string`. Then the
  template writes today's date (checked: an unpinned render on 2026-10-04 shows
  "Today Date: 04 Oct 2026"), and the prompt an adapter was trained on differs
  from the one it is served with on every other day. This is the same failure as
  `p1-eval-chat-template-date` and `p2-chat-template-date`, reintroduced by a
  prompt change.
- **35 tokens is the floor for the stock template.** Removing them would need a
  custom chat template, which every consumer (training, harness, serving, vLLM)
  would have to share. At 35 of ~113 tokens per full-train example it is still
  about 30% of the minimal prompt, so it is a possible later saving, not a free one.

## How it was found

Rendering the template for each candidate style before choosing one
(`docs/benchmarks/p3-prompt-ablation.md`, "Choosing the minimal prompt").

## Fix

No code change was needed for correctness: `render_prompt`, `encode_prompt`, and
`HFPredictor` pass `CHAT_DATE_STRING` for every style.
`tests/test_prompt_style.py::test_minimal_prompt_is_much_shorter_and_still_dated`
pins that the minimal prompt starts with the template's system header, contains
the pinned date, and has exactly 35 fixed tokens. A custom template that drops
the header is left as an option, not done.
