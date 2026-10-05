# P3 local pilot: a 2,400-example LoRA run on the M4

**Date:** 2026-10-04 · **Hardware:** Apple M4, 16 GB unified memory, MPS (shared with another agent's jobs) · **Model:** Llama 3.2 1B Instruct, bf16 base, fp32 LoRA · **Eval:** 1,014 plays from `eval_lite.jsonl` (2024) · **CIs:** 95%, cluster bootstrap over games

This is a pilot at about 1% of the training data, on a laptop. It is indicative,
not a ladder result. It answers two questions: does the integrated pipeline
(train → validate → harness → serve-format adapter) work on the real model, and
what does a small adapter already learn?

## Headline

1. **Fine-tuning works, and fast.** 300 optimizer steps on 2,400 plays take the
   base model from **0.0%** exact match (zero-shot R1, valid JSON only 24.7% of the
   time) to **93.7%** exact match with 100% valid, schema-exact JSON. On the
   108-play slice where few-shot R2 was also run, R2 reaches 16.7%.
2. **The pilot adapter does not beat the regex.** R0 scores 97.6% on the same
   plays. The adapter is significantly worse on three buckets (paired
   differences below exclude zero): **penalty_stands −20 points, fumble −16,
   lateral −14**. It ties R0 on the other six buckets. Of 1,014 plays, the
   adapter is right where R0 is wrong on only **2**; R0 is right where the
   adapter is wrong on **42**.
3. **The errors are yard arithmetic, not reading.** On val plays from the losing
   buckets, almost every adapter error has the right players and stats but the
   wrong yardage. The labels use official yards, which for a spot foul (an
   accepted holding penalty "enforced at" a yard line) or a fumble must be
   computed from field positions; the play text states a different number. The
   regex has an explicit rule for this. 148 examples per bucket did not teach it.
4. **Cost is dominated by the fixed prompt.** 89% of every training example's
   tokens are prompt, 71% is the identical template (system prompt plus chat
   headers, 221 tokens), and only 11% (≈35 tokens) carries the loss.

## Setup

| | |
|---|---|
| Training data | 2,400 plays from `train.jsonl` (2015–2022), seed 20263: all 166 laterals, ~148 each of the other 7 non-normal buckets, 1,200 normal |
| Distribution | **Deliberately enriched**: 50% non-normal vs 16% in train (fumble 6.1% vs 1.4%, lateral 6.9% vs 0.06%). The adapter saw far more hard plays, relatively, than a natural sample would give |
| LoRA | Hand-written `playparse.lora`, r=16, alpha=32, all 7 linear projections, dropout 0.05; 11,272,192 trainable parameters |
| Optimizer | AdamW, lr 2e-4, warmup 9 steps (3%), cosine to 2e-5, grad clip 1.0 |
| Batching | Micro-batch 1 × accumulation 8 = 8 examples/step, 1 epoch = 300 steps, padded to a multiple of 64 |
| Validation | Val loss on 200 proportional-stratified val plays every 75 steps; greedy generation on 90 bucket-balanced val plays (10/bucket) every 100 steps |
| Eval set | All 814 non-normal plays of `eval_lite.jsonl` + 200 seeded normal plays (seed 20265); 232 games |
| Decoding | Greedy, `max_new_tokens` 256, batch 16, bf16 for R1 and the adapter (the adapter's training precision) |

Commands: see the docstring of `scripts/p3_pilot.py` and `configs/p3_local_pilot.yaml`.
Every key and file hash of the subsets is in `results/p3_local_pilot/data_manifest.json`.

## Results

### Exact match per bucket (%)

| bucket | n | pilot adapter | R0 regex | R1 zero-shot | adapter − R0 (paired) |
|---|---|---|---|---|---|
| **overall** | 1014 | **93.7** [91.7, 95.3] | **97.6** [96.5, 98.5] | 0.0 [0.0, 0.0] | **−3.9** [−5.5, −2.6] |
| penalty_stands | 100 | 80.0 [73.0, 87.8] | 100.0 [100, 100] | 0.0 | **−20.0** [−27.2, −12.8] |
| fumble | 100 | 76.0 [68.4, 83.7] | 92.0 [86.3, 97.1] | 0.0 | **−16.0** [−23.4, −8.7] |
| lateral | 29 | 48.3 [30.0, 66.7] | 62.1 [42.9, 79.2] | 0.0 | **−13.8** [−27.6, −3.4] |
| challenge | 100 | 96.0 [92.1, 100.0] | 96.0 [91.8, 99.1] | 0.0 | 0.0 [−3.0, +2.9] |
| penalty_nullified | 185 | 99.5 [98.1, 100.0] | 99.5 [98.1, 100.0] | 0.0 | 0.0 |
| interception | 100 | 100.0 | 100.0 | 0.0 | 0.0 |
| td | 100 | 100.0 | 100.0 | 0.0 | 0.0 |
| two_point | 100 | 100.0 | 100.0 | 0.0 | 0.0 |
| normal | 200 | 100.0 | 100.0 | 0.0 | 0.0 |

The last column is the paired difference (same plays, resampling games), which
is tighter than comparing the two separate intervals.

| | pilot adapter | R0 | R1 |
|---|---|---|---|
| Valid JSON (after extraction) | 100% | 100% | 24.7% |
| Strictly the JSON object, nothing else | 100% | 100% | 24.7% |
| Credit F1 | 93.5 [91.0, 95.3] | 97.0 [95.2, 98.2] | 2.0 [1.2, 3.0] |
| Game PPR fantasy-point MAE | 0.23 [0.13, 0.35] | 0.06 [0.03, 0.11] | 4.94 [3.94, 6.09] |
| Mean output tokens | 32 | — | 126 (212 of 1,014 hit the 256 cap) |
| Throughput (batch 16) | 0.76 plays/s | 74,000 plays/s (CPU) | 0.32 plays/s |

The fantasy MAE on this set mixes complete games with "top-up" plays from other
games, so it is only comparable across rungs here, not with other benchmarks.

### R2 (fixed 8-shot) on a 108-play slice

Few-shot prompts are about four times longer, so R2 ran on a bucket-balanced
slice (12 per bucket, seed 20264; 0.10 plays/s at batch 4).

| Rung | Exact (correct / 108) | Valid |
|---|---|---|
| Pilot adapter | 98 | 100% |
| R0 | 102 | 100% |
| R2 | 18 (16.7% [10.8, 23.8]) | 70.4% |
| R1 | 0 | — |

Adapter − R2 on the slice: +74.1 points [+66.7, +81.4].

### What R1 and R2 get wrong

R1 almost never produces the schema: 356 outputs are not a JSON object of the
right shape, and many invent stats (`interception`, `reception`, `sack`,
penalty names). Even the 250 valid outputs are never exactly right. R2's
examples fix the format most of the time (70% valid) but not the content. This
matches the P1 smoke runs and is why the jump to 93.7% is attributable to
fine-tuning, not to the prompt.

## Error analysis (val, not test)

The eval plays come from the frozen 2024 test season, so their text was not read
for diagnosis. Aggregate counts only: on eval, the adapter's penalty_stands
errors are 17 "right stat, wrong value" and 5 spurious credits; its fumble
errors are mostly wrong values (14) and spurious `fumble_lost` or `rush_yds`
credits.

To see why, the adapter and R0 were run on 112 val (2023) plays from the losing
buckets (all 16 laterals plus 48 seeded penalty_stands and 48 fumbles;
`p3_pilot.py valdiag`). The pattern reproduces: penalty_stands 77.1% vs R0
100%, fumble 81.2% vs 91.7%, lateral 43.8% vs 43.8%.

Reading those val errors:

- **Penalty stands (spot fouls).** Nearly every miss is an offensive holding
  "enforced at" a yard line behind the end of the run. The label credits the
  yards gained up to the spot of the foul. Example: "left end to LV 15 for 12
  yards … Offensive Holding, 10 yards, enforced at LV 25" has gold
  `rush_yds: 2`; the adapter wrote 12. Getting this right means subtracting yard
  lines, which the regex does with an explicit rule
  (`docs/issues/p1-eval-spot-foul-regex-escape.md`).
- **Fumbles.** Most misses are off by one or two yards: the label uses official
  yards, which differ from the stated gain on many fumble plays
  (`docs/issues/p1-official-yards-vs-text.md`). A few are convention cases, such
  as aborted snaps, where gold has no credits.
- **Laterals.** Multi-player plays where yards must be split between the catcher
  and the lateral receiver; both the adapter and R0 are weak here.

The val generation subset during training (10 plays per bucket) showed 100% on
penalty_stands at step 300. Ten plays happened to contain no spot fouls, which
is a reminder that a 90-play validation set is a coarse instrument.

## Training run

| | |
|---|---|
| Wall time | 2.11 h (15:46–17:53), including validation and other agents' jobs |
| Sum of step times | 1.57 h for 300 steps |
| Steady throughput | median 233 input tokens/s (26 loss tokens/s), 10.7 s per 8-example step; p90 271 tokens/s |
| Contention | Another agent ran real-model benchmarks during the run. Windows dropped to 11–40 tokens/s while macOS swapped; the run was paused (SIGSTOP) for 3 minutes to let one benchmark finish |
| Peak memory | 11.6 GiB Metal driver total, 8.3 GiB live tensors (the 8-example val-loss batches at up to 640 tokens set the peak) |
| Val loss | 0.0063 (step 75) → 0.0057 (150) → 0.0012 (225) → 0.00057 (300) |
| Train loss | 0.46 (steps 1–5) → about 0.01 from step 100 on |
| Generation val (balanced, 90 plays) | 77.8% (step 100) → 85.6% (200) → 91.1% (300); lateral 20% → 30% → 40%, fumble 50% → 70% → 80% |
| Validation cost | 4 val-loss passes 2.5–6.5 min each, 3 generation passes ~5 min each: about 35 min of the 2.1 h |
| Adapter | 43 MB (fp32), PEFT format; sha256 in `results/p3_local_pilot/summary.json` |

The best checkpoint by balanced val exact match was the last one (step 300), and
both curves were still improving. One epoch at this size is under-trained, not
over-fit.

## Prompt versus completion tokens

| | train pilot (2,400) | full train (294,016) | eval set (1,014) |
|---|---|---|---|
| Prompt tokens / example | 275.7 | 266.3 | 280.8 |
| Completion tokens / example | 35.1 | 32.6 | 31.8 |
| Prompt share | 88.7% | 89.1% | 89.8% |
| Fixed template (system prompt 186 + chat headers) | 221 | 221 | 221 |
| Padding to a multiple of 64 | +12.0% | +13.9% | +10.3% |
| Total tokens | 0.75 M | 87.9 M | 0.32 M |

Why it matters: training compute scales with all tokens, but only completion
tokens carry gradient. Roughly nine of every ten tokens the H100 processes in
R5 are prompt. A shorter system prompt, or one learned into the adapter and
dropped at inference, would cut training and serving cost by up to 70%. That is
a candidate ablation, not a change made here: `SYSTEM_PROMPT` is frozen for
comparability with R1–R4.

The full train split is 87.9 M tokens per epoch, not the PRD's 57 M estimate
(PRD §13 assumed ~150 tokens per play; the real mean is 299). At the pilot's
local 233 tokens/s, one epoch would take about 105 hours.

## Merged versus unmerged

On the first 64 eval plays, merging the adapter into the bf16 base
(`--merge`) gave byte-identical output on 63 and a different output on 1 (wrong
both ways). Merging rounds the merged weights to bf16, which can flip a near-tie
token, as expected from `docs/issues/p0-bf16-unmerge-drift.md`. Merged
throughput was 1.0 plays/s versus 0.76 unmerged.

## What the pilot shows, and what it does not

It shows:

- The integrated pipeline works on the real model: the hand-written LoRA trains
  on MPS, saves PEFT-format adapters, the stratified val callback tracks bucket
  progress, and the harness scores the adapter exactly like R0–R2.
- Fine-tuning is the step that makes a 1B model usable for this task (0% → 94%),
  far beyond prompting (R2 17% on the slice).
- Where a small adapter fails: rules that need yard-line arithmetic.

It does not show:

- **Anything about claims T1–T3 at full scale.** 2,400 plays is ~1% of train, one
  epoch, no tuning, a shifted training distribution. The H100 run trains on
  ~120× more data, including ~6,700 penalty_stands and ~4,000 fumbles.
- **That the adapter cannot beat the regex.** The pilot was still improving
  when it stopped.
- **Clean timing.** The machine was shared, and throughput varied 20× between
  windows.
- **Generality of the eval.** The eval set over-represents hard buckets by
  design (80% non-normal) and topup plays come from games outside the 18 full
  games.

Provenance: result files carry the git SHA they ran at. They show
`git_dirty: true` because the pilot script and docs were being edited during the
runs; the library code under `playparse/` was committed.

## Artifacts

`results/p3_local_pilot/`:

- `data_manifest.json`, `token_stats.json`, `full_dataset_token_lengths.json`
- `train/` (metrics.jsonl, run spec and metadata, per-step val generations, adapter config)
- `eval/{lora,r0,r1,r2}/` (harness result.json + predictions.jsonl)
- `extra/` (merged check, val diagnostic for adapter and R0)
- `summary.json` (all rungs, paired differences, R2 slice comparison, adapter hash)
