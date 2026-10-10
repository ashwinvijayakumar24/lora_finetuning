# T2: a fine-tuned 1B model vs frontier API models

**Date:** 2026-10-10 · **Eval:** eval_lite (2,795 plays from 2024; game-clustered paired
bootstrap, 2,000 resamples) · **Frontier rung (R4):** OpenAI Chat Completions, same system
prompt and 8 fixed few-shot examples as R2, `reasoning_effort=low`, prices verified from
OpenAI's pricing page on 2026-10-10. Artifacts: `results/eval_lite/r4_gpt-5.5/`,
`results/eval_lite/r4_gpt-5.4-mini/`.

| System | Exact match [95% CI] | R5 − system (paired) | Cost per 1k plays | p50 latency |
|---|---|---|---|---|
| **R5: Llama 3.2 1B + LoRA** | **98.53 [97.7, 99.0]** | — | **≈ $0.04** (see note) | 2.3 s per batch of 32 |
| R4: gpt-5.5, 8-shot | 97.96 [97.0, 98.6] | **+0.57 [+0.17, +1.00]** | $6.40 (measured) | 1.7 s per play |
| R4: gpt-5.4-mini, 8-shot | 95.24 [93.8, 96.1] | +3.29 [+2.68, +4.06] | $1.00 (measured) | 1.3 s per play |
| R0: regex | 99.14 [98.6, 99.5] | −0.61 [−1.14, −0.27] | ≈ $0 (CPU) | ~0.01 ms |

Per bucket (exact match %): gpt-5.5 fumble 80, penalty_stands 82, lateral 62, challenge 95,
normal 99.9; R5 fumble 85, penalty_stands 94, lateral 45, challenge 96, normal 100.

**Cost note.** The API costs are measured spend. R5's figure is an estimate: the eval ran
at 15.0 plays/s on one A100 80GB through plain Hugging Face `generate` (batch 32), i.e.
~67 GPU-seconds per 1k plays, priced at an assumed ~$2/hour A100 rental. That is
unoptimized; a production server (vLLM) would be faster still.

## Verdict: T2 earned

The pre-registered condition was "R5 within CI of R4 at a measured fraction of the cost".
R5 is *above* gpt-5.5 (+0.57 points, CI excludes 0) at roughly 1/170th of the cost per play.
The frontier model is better on laterals (62 vs 45), which need multi-step yardage
reasoning, and worse on the official-yardage buckets (fumble, penalty_stands), where the
fine-tuned model has absorbed nflverse's stat conventions from 294k examples and the
8-shot prompt cannot convey them.

This is the decision-ladder answer for a narrow, high-volume extraction task: a small
fine-tuned model beats a frontier model on quality *and* cost, while a hand-written
regex is still best of all on this clean, fixed-format text.
