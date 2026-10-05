# P3 prompt ablation: does a fine-tuned adapter still need the system prompt?

**Date:** 2026-10-04 · **Hardware:** Apple M4, 16 GB unified memory, MPS (no other agents' jobs) · **Model:** Llama 3.2 1B Instruct, bf16 base, fp32 LoRA · **Eval:** the P3 pilot's 1,014 plays from `eval_lite.jsonl` (2024) · **CIs:** 95%, cluster bootstrap over games; differences are paired (same plays, same resampled games)

## Headline

**No.** An adapter trained and served with no system message at all matches the
full-prompt adapter on quality, at about 40% of the tokens per example.

1. **Quality is the same within the resolution of this test.** Overall exact
   match is 93.2% without the system prompt versus 93.7% with it. The paired
   difference is −0.5 points with a 95% interval of [−1.5, +0.4], so a drop larger
   than 1.5 points is ruled out, and so is any gain. Credit F1 (93.4 vs 93.5) and
   fantasy-point error (0.232 vs 0.231) are also equal. On 964 of the 1,014 plays
   the two adapters produce the same parsed answer.
2. **Two buckets lean worse on eval, but validation does not confirm it.**
   Challenge (−4.0 points, [−8.9, 0.0]) and lateral (−6.9, [−18.2, 0.0]) touch zero
   at the upper end. On every 2023 validation play of those buckets (330 challenge
   plays, three times the eval count, plus 16 laterals) the gap is −0.9 [−2.8, +0.9]
   on challenge, and the minimal adapter wins on laterals (10 vs 8 of 16). The one
   rule the system prompt spells out for challenges ("credit the final ruling after
   REVERSED") is learned from the labels alone: 179 vs 180 of 193 reversed val
   plays are right.
3. **The savings are large.** Per training example, 124.8 tokens instead of
   310.8 (−60%). Per eval request, 94.8 prompt tokens instead of 280.8 (−66%). On
   the same 30 training steps on an idle machine, a step takes 3.2 s instead of
   9.2 s (2.8× faster). On the same 256 eval plays run back to back, throughput is
   1.40 plays/s instead of 0.80 (1.75×). Peak Metal memory drops by about 3 GiB in
   both training and inference.
4. **For the GPU runs:** one full-train epoch shrinks from 87.9 M to 33.2 M real
   tokens (−62%), and from 102.0 M to 47.3 M tokens as actually batched (micro-batch
   16, padded to the longest example in each batch; −54%). **Recommendation: train
   R5 and the sweep with `data.prompt_style: minimal`**, with one full-prompt
   control at 50k (below).

## What was compared

Both arms are the same LoRA recipe on the same data. The only difference is the
prompt each adapter was trained and evaluated with.

| | full (the P3 pilot, reused) | minimal (this run) |
|---|---|---|
| Messages | system prompt (186 tokens) + user turn | user turn only |
| Fixed template tokens per example | 221 | 35 |
| Training data | 2,400 plays, sha256 `006528ff…` | the same file (sha256 and keys checked against the pilot manifest) |
| Recipe | r=16, alpha=32, all 7 linear projections, dropout 0.05, lr 2e-4, warmup 3%, cosine to 10%, micro-batch 1 × accumulation 8, 1 epoch = 300 steps, padded to a multiple of 64, seed 0 | identical (`configs/p3_prompt_ablation.yaml` differs from the pilot config only in `data.prompt_style`, the data path, and the output directory) |
| Validation | 200 val plays for val loss, 90 balanced val plays for generation | identical keys |
| Eval | 1,014 plays, sha256 `3f7df5cb…`, greedy, batch 16, bf16, `max_new_tokens` 256 | identical, with `--prompt-style minimal` |
| Eval config hash | `eb572750ffd7cf31` (unchanged by this work) | `5fa4c6f9713f41b3` |

The full-prompt arm was not retrained. Its adapter and committed results from
`results/p3_local_pilot/` are reused. The minimal adapter's sha256 is
`a93ec3b7…` (in `results/p3_prompt_ablation/summary.json`).

### Choosing the minimal prompt

Two candidates were considered: no system message, or a one-line system message
such as "Extract fantasy stat credits as JSON." (7 more tokens). No system message
was chosen, for three reasons:

- **It is the cleaner test.** The question is whether the adapter needs
  instructions at all. A one-line instruction would leave that open.
- **The one-liner buys nothing the adapter can use.** Its only plausible benefit is
  steering the base model before training, and the base model scores 0% even with
  the full 186-token prompt (R1). Within about 15 training steps the adapter has
  learned the format either way.
- **Serving picks the adapter by name, not by prompt.** In the multi-adapter
  server (P5b) a request already names its adapter, so the prompt does not need to
  say which task it is.

One surprise from rendering the template: with no system message, the Llama 3.2
chat template still writes its own system block, "Cutting Knowledge Date: December
2023 / Today Date: …". An empty system message renders exactly the same text. So
the minimal prompt has 35 fixed tokens, not zero, and **the pinned date
(`CHAT_DATE_STRING`) is still needed**. Without it the prompt would carry today's
date and change daily. See
[`p3-minimal-prompt-keeps-template-header`](../issues/p3-minimal-prompt-keeps-template-header.md).

## Results

### Exact match per bucket (%)

| bucket | n | full | minimal | minimal − full (paired) | minimal − R0 (paired) | only minimal right / only full right |
|---|---|---|---|---|---|---|
| **overall** | 1014 | **93.7** [91.7, 95.3] | **93.2** [91.0, 94.7] | **−0.5** [−1.5, +0.4] | −4.4 [−6.2, −3.0] | 8 / 13 |
| penalty_stands | 100 | 80.0 [73.0, 87.8] | 81.0 [73.8, 88.2] | +1.0 [−2.5, +4.5] | −19.0 [−26.2, −11.7] | 2 / 1 |
| fumble | 100 | 76.0 [68.4, 83.7] | 76.0 [67.6, 83.3] | 0.0 [−6.6, +6.6] | −16.0 [−24.4, −7.9] | 5 / 5 |
| lateral | 29 | 48.3 [30.0, 66.7] | 41.4 [23.5, 58.8] | −6.9 [−18.2, 0.0] | −20.7 [−37.5, −6.9] | 0 / 2 |
| challenge | 100 | 96.0 [92.1, 100.0] | 92.0 [86.7, 97.0] | −4.0 [−8.9, 0.0] | −4.0 [−8.7, 0.0] | 1 / 5 |
| penalty_nullified | 185 | 99.5 | 99.5 | 0.0 | 0.0 | 0 / 0 |
| interception | 100 | 100.0 | 100.0 | 0.0 | 0.0 | 0 / 0 |
| td | 100 | 100.0 | 100.0 | 0.0 | 0.0 | 0 / 0 |
| two_point | 100 | 100.0 | 100.0 | 0.0 | 0.0 | 0 / 0 |
| normal | 200 | 100.0 | 100.0 | 0.0 | 0.0 | 0 / 0 |

Paired differences use `playparse.eval.bootstrap.paired_diff_ci` with 2,000
resamples of the 232 games (seed 0). Both adapters remain behind the R0 regex by
the same margin and on the same three buckets as the pilot found (full − R0 is
−3.9 [−5.6, −2.6]). Removing the system prompt does not change what the small
adapter gets wrong: yard arithmetic on spot fouls and fumbles.

| | full | minimal | minimal − full |
|---|---|---|---|
| Valid JSON / strictly the JSON object | 100% / 100% | 100% / 100% | 0 |
| Credit F1 | 93.5 [91.0, 95.3] | 93.4 [91.1, 95.1] | −0.1 [−0.9, +0.6] |
| Game PPR fantasy-point MAE | 0.231 [0.135, 0.346] | 0.232 [0.131, 0.353] | +0.001 [−0.057, +0.062] |
| Mean output tokens | 32.0 | 32.1 | same |

### Where the two adapters differ

On eval, only aggregate error types were examined (the 2024 play text is not
read). Minimal's five extra challenge misses are three spurious credits, one
different credit set, and one wrong value. Its two extra lateral misses are
missing credits.

To see what those look like, both adapters were run on every 2023 validation
play of the two buckets (`p3_prompt_ablation.py valdiag`; 346 plays):

| val bucket | n | full | minimal | minimal − full (paired) | only minimal right / only full right |
|---|---|---|---|---|---|
| challenge | 330 | 310 (93.9%) | 307 (93.0%) | −0.9 [−2.8, +0.9] | 4 / 7 |
| … with REVERSED in the text | 193 | 180 | 179 | | |
| lateral | 16 | 8 | 10 | +12.5 [−12.5, +36.4] | 3 / 1 |

Reading the eleven discordant val challenge plays:

- **Minimal's misses are about fumble possession, not about the review.** Five of
  its seven misses add a `fumble_lost` when the ball went out of bounds or was
  recovered by the offense, or drop one that the defense recovered. One misreads a
  reversed incompletion that became a strip-sack, and one credits −1 rushing yards
  where the label has no credits. Neither prompt says anything
  about fumble possession, so this is not knowledge the system prompt carried.
- **Full's misses include crediting reversed touchdowns.** Three of its four
  misses credit a `pass_td` or `rec_td` that the replay REVERSED, which the system
  prompt explicitly tells it not to do. Minimal got all three right.

So the eval lean on challenge looks like noise in a 100-play bucket, and the
REVERSED instruction is not doing measurable work once the adapter is trained.

### Training

| | full (pilot) | minimal |
|---|---|---|
| Tokens per example (mean, real) | 310.8 | 124.8 |
| … padded to a multiple of 64 | 348.2 | 159.7 |
| Loss-carrying share of tokens | 11.3% | 28.1% |
| **Controlled step time** (same first 30 steps, same examples, idle machine) | 9.18 s | 3.25 s (2.8× faster) |
| Controlled throughput, all tokens / loss tokens | 268 / 28 tokens/s | 303 / 80 tokens/s |
| Whole-run median step time | 10.7 s | 4.2 s |
| Sum of step times, 300 steps | 1.57 h | 0.34 h |
| Wall time including validation | 2.11 h (shared machine) | 0.60 h |
| Peak Metal memory, driver total / live tensors | 11.6 / 8.3 GiB | 8.5 / 6.1 GiB |
| Train loss, steps 1–5 | 0.46 | 2.69 |
| Val loss at steps 75 / 150 / 225 / 300 | 0.0063 / 0.0057 / 0.0012 / 0.00057 | 0.0038 / 0.0014 / 0.0010 / 0.00057 |
| Generation val (90 balanced) at 100 / 200 / 300 | 77.8 / 85.6 / 91.1% | 78.9 / 84.4 / 92.2% |

How to read the timing rows:

- **The controlled row is the fair one.** The pilot shared the laptop with
  another agent's benchmarks, so its whole-run numbers are inflated. To get a
  clean number, the full-prompt config was rerun for 30 steps on an idle machine
  (`runs/p3_prompt_ablation/full_probe`, no validation, same seed, so the same 240
  examples in the same order) and compared with the minimal run's first 30 steps,
  which also ran on an idle machine.
- **The speedup (2.8×) is larger than the token ratio (2.2× padded).** A step's
  time does not fall exactly with token count. Attention cost grows faster than
  linearly with length, and at 11.5 GiB the full-prompt run sits near the 16 GB
  machine's swap threshold.
- **The minimal run itself was not perfectly clean.** From about step 40 to step
  110 a CPU-heavy token count and two test-suite runs shared the machine, which is
  why its whole-run median (4.2 s) is above its first-30-step median (3.2 s).
- **The higher first loss is expected and brief.** Without instructions the base
  model has no idea what format to produce, so the first steps start from a loss
  of 2.7. By step 15 it is 0.22, and from step 75 on the minimal run's val loss is
  at or below the pilot's.

### Inference

| | full | minimal |
|---|---|---|
| Mean prompt tokens per eval play | 280.8 | 94.8 (−66%) |
| Mean prompt + output tokens (KV cache per sequence) | 312.8 | 126.9 (−59%) |
| **Throughput, same 256 plays back to back, batch 16** | 0.80 plays/s | 1.40 plays/s (1.75×) |
| Peak Metal memory in that run, driver / live | 8.9 / 3.5 GiB | 6.4 / 3.1 GiB |
| Throughput over all 1,014 plays | 0.76 plays/s (shared machine) | 1.11 plays/s (slowed from 2.1 to 1.0 over the run) |

The whole-set throughput numbers are not comparable. The pilot's eval ran on a
shared machine. The minimal eval started at 2.1 plays/s and fell to 1.0 as the
laptop heated and swap filled (4.8 of 6 GB in use). The 256-play control runs both
adapters back to back on the same plays. Both control reruns reproduced their main
run's outputs exactly (256/256 identical raw strings), so greedy decoding here is
deterministic and the speed difference is the prompt length, not different
outputs.

## Projected savings for the GPU runs

All token counts below are measured on the full train split (294,016 plays,
`scripts/p3_prompt_ablation.py tokens` and `gpu_projection`). The minimal prompt is
exactly 186 tokens shorter for every example, which a test pins.

| Per epoch of full train | full | minimal | change |
|---|---|---|---|
| Prompt tokens / example | 266.3 | 80.3 | −70% |
| Real tokens | 87.9 M | 33.2 M | −62% |
| Padded to a multiple of 64 | 100.1 M | 44.2 M | −56% |
| As batched by `train_default.yaml` (micro-batch 16, pad to batch max, mean of 3 shuffles) | 102.0 M | 47.3 M | **−54%** |
| Padding overhead in that batching | 16% | 43% | |

What this means in GPU time, using the P3 plan's unmeasured assumption of
15,000–30,000 tokens/s for LoRA on one H100:

- **R5 (2 epochs):** about 1.9–3.8 h with the full prompt, 0.9–1.8 h without.
- **The 20 sweep runs at 50k plays:** roughly 8–14 GPU hours become about 4–6.5.
- **Recovering more:** padding becomes the largest waste once the prompt is short
  (43% overhead). Grouping examples of similar length into the same micro-batch
  would bring the batched count close to the 33.2 M real tokens, about 62% below
  today's full-prompt cost. That is a separate change, not made here.

**Serving prefill** falls by the same factor as the prompt: about 66% fewer prefill
tokens per request (281 → 95 on the eval set), and 59% less KV cache per sequence,
so more requests fit in a batch. One caveat: a server with automatic prefix caching
(vLLM's APC, for example) already shares the identical system-prompt prefix
across requests, so for such a server the full prompt costs less than its length
suggests. Training has no such cache, so the training savings above are real in
full.

## Recommendation

**Use `data.prompt_style: minimal` for R5 and the sweep**, and evaluate those
adapters with `--prompt-style minimal`. Quality is unchanged within ±1.5 points
overall, and validation shows no bucket-level loss, while training cost halves.

Three conditions go with it:

1. **Run one control.** Add a single full-prompt run to the 50k sweep (the base
   config with `data.prompt_style=full`). This ablation is one seed per arm at 1%
   of the data. The control confirms the result at 20× the data for about one
   sweep run's cost.
2. **Watch `val_exact_match/challenge` in R5.** It is the bucket that leaned worse
   on eval here, even though validation did not confirm it.
3. **Keep the baselines' prompt.** R1–R4 keep the full system prompt. The prompt is
   part of each rung's method, and a prompted base model does need instructions.
   Rungs that compare two adapters (for example R5 against R7, the full fine-tune)
   should use the same style.

The default in code stays `full`, so nothing changes for existing configs or
results until a config opts in.

## What this does and does not show

It shows:

- At pilot scale, the system prompt does not improve a fine-tuned adapter's
  quality on any bucket that validation can resolve.
- Removing it cuts training compute per epoch by more than half and inference
  prefill by two thirds, with lower peak memory in both.

It does not show:

- **Seed variance.** Each arm is a single training run. The eval interval covers
  play sampling, not training randomness. The validation check and the 50k
  control are the guards against a lucky or unlucky seed.
- **Behavior at full scale.** With 120× more data, any information the prompt adds
  should matter even less, but that is an expectation, not a measurement.
- **Natural-distribution accuracy.** The eval set is 80% hard plays by design.
- **H100 timings.** The token counts transfer exactly. The M4 step-time ratio
  (2.8×) will not, since an H100 is far from memory-bound at these lengths. Expect
  the speedup to track the token ratio (about 2.2×).

## Reproduce

```bash
python scripts/p3_prompt_ablation.py data --src <dir with the pilot's train_pilot.jsonl and eval_pilot.jsonl>
python scripts/p3_prompt_ablation.py tokens
python scripts/p3_prompt_ablation.py gpu_projection
python scripts/train.py --config configs/p3_prompt_ablation.yaml \
    --set data.val="\"$PLAYPARSE_PROCESSED_DIR/val.jsonl\""      # add --resume latest after an interruption
python scripts/p3_prompt_ablation.py eval --name lora_minimal --prompt-style minimal \
    --adapter runs/p3_prompt_ablation/train/checkpoints/step_0000300/adapter
# controls and diagnostics: see the docstring of scripts/p3_prompt_ablation.py
python scripts/p3_prompt_ablation.py report
```

Provenance: library code was committed before each run. Result files show
`git_dirty: true` because the ablation script and docs were being edited during
the runs.

## Artifacts

`results/p3_prompt_ablation/`:

- `summary.json`: both arms and R0, paired differences (minimal − full, minimal −
  R0, full − R0) for every metric and bucket, discordant counts, timing and memory
  controls, training summaries, the val diagnostic, and the adapter hash
- `data_check.json` (the copied subsets match the pilot manifest),
  `token_stats.json` (both styles: pilot train, eval, full train),
  `gpu_projection.json`, `val_diag_manifest.json`
- `eval/lora_minimal/` (harness result, predictions, memory), `eval/lora_full_timing/`
  and `eval/lora_minimal_timing/` (the 256-play speed control)
- `extra/valdiag_{full,minimal}/` (val challenge and lateral diagnostic)
- `train/` (metrics, run spec and metadata, per-step val generations, adapter
  config), `full_probe/` (the 30-step full-prompt throughput control)
