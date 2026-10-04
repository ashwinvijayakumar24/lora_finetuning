# PRD — PlayParse: fine-tuning and serving a fantasy-football play parser

**Status:** draft · **Owner:** Ashwin · **Last updated:** 2026-10-04
**Base model:** Llama 3.2 1B Instruct · **Compute:** Apple M4 (correctness), PACE H100 (training and benchmarks)

---

## 1. Summary

PlayParse teaches Llama 3.2 1B to read raw NFL play-by-play text and output the
fantasy-relevant stats in it: who gained yards, who caught the ball, who scored, who
fumbled. A deterministic scorer then turns those stats into exact fantasy points
under any scoring system.

The project has two halves, and both matter equally:

1. **Adapter track (applied AI).** Build an exact eval. Then climb a *decision
   ladder*: regex → prompting → retrieved examples → frontier model → LoRA → QLoRA →
   full fine-tune → distillation. Publish where each rung wins and where it loses.
2. **Serving track (infra).** Add LoRA to the from-scratch `llm_inference_engine` and
   `llm_serving_layer`: merged adapters, unmerged adapters, then many adapters batched
   together on one base model. Gate every adapter release on the frozen eval.

The finished artifact is a README in the style of `llm_serving_layer`: scoped claims,
each marked earned or not earned, each backed by a committed result file.

## 2. Why this project

### Why this task

- **Exact ground truth, for free.** nflverse play-by-play data puts the official play
  text (`desc`) in the same row as the correct structured fields. Every metric is an
  exact match, so no LLM judge is needed.
- **A real gap between easy and hard cases.** A regex handles
  `"22-D.Henry left tackle to TEN 32 for 7 yards"`. It struggles with laterals,
  fumbles recovered by the offense, plays nullified by penalties, overturned
  challenges, and two-point tries. That spread makes the decision ladder informative:
  the interesting result is *where* ML earns its keep, not one overall accuracy number.
- **Distillation can be measured exactly.** With ground truth and a teacher model both
  available, you can train students on each and measure exactly what distillation
  costs. Usually that number can only be estimated.
- **Cheap iteration.** Inputs are about 40 tokens and outputs about 60. Full sweeps on
  a 1B model are affordable.
- **Passion fit.** It extends the `fantasy_football` draft tool.

### Why it's a good learning vehicle

Every item on the LoRA learning list maps to a phase and an artifact. See
[`LEARNING_MAP.md`](LEARNING_MAP.md).

## 3. Goals and non-goals

### Goals

| ID | Goal |
|---|---|
| G1 | A frozen, exact, bucketed eval for play → stats extraction |
| G2 | A decision-ladder table covering quality, cost, latency, and engineering effort |
| G3 | A hand-written LoRA layer and training loop, each verified against a reference implementation |
| G4 | A measured explanation of every LoRA knob on this task |
| G5 | A controlled distillation experiment: ground truth vs teacher vs filtered teacher |
| G6 | LoRA serving in your own engine and serving layer, including heterogeneous multi-adapter batching |
| G7 | An adapter registry with an eval gate and rollback |

### Non-goals

- Predicting fantasy points. That is forecasting, a different problem.
- Live game ingestion.
- Kicker, defense/special-teams, and return-yardage scoring. This is v1 scope, and
  the scorer is designed so they could be added later.
- Library-only training. TRL's `SFTTrainer` and similar tools may be used as
  *references* to diff against, never as the implementation.

## 4. Task specification

### Input

The play text plus light context. The prompt is rendered through the Llama 3.2 chat
template.

```
posteam: PHI
desc: (3:12) (Shotgun) 1-J.Hurts pass short right to 11-A.Brown to DAL 22 for 14 yards
(21-T.Diggs). FUMBLES (21-T.Diggs), RECOVERED by DAL-55-D.Lawrence at DAL 20.
```

### Output

JSON with a fixed schema:

```json
{"nullified": false,
 "credits": [
   {"player": "J.Hurts", "stat": "pass_yds",    "value": 14},
   {"player": "A.Brown", "stat": "rec",         "value": 1},
   {"player": "A.Brown", "stat": "rec_yds",     "value": 14},
   {"player": "A.Brown", "stat": "fumble_lost", "value": 1}]}
```

**Stat vocabulary (v1):** `pass_yds, pass_td, int, rush_yds, rush_td, rec, rec_yds,
rec_td, fumble_lost, two_pt`.

**Rules the output must follow:**

- Player names use the abbreviated form that appears in `desc` (`A.Brown`). Matching
  to player IDs is done afterward, not by the model.
- Sacks produce no rushing yards. Scrambles count as runs.
- `nullified: true` means the play was wiped out by a penalty or replay, so
  `credits` is empty.
- Credits are listed in canonical order (passer, then receiver, then rusher; and
  within a player, in vocabulary order). A fixed order makes exact match well defined.

### Scorer

`ffscore/scorer.py` maps credits → points under a scoring config (standard, half-PPR,
full PPR, or your 14-team ESPN league). The model never computes points. The
model's job is reading; arithmetic stays in plain code.

## 5. Data

### Source

nflverse play-by-play, loaded via `nflreadpy` or the nflverse-data GitHub releases.
Columns used (checked against the official dictionary,
`nflverse/nflreadr/data-raw/dictionary_pbp.csv`):

| Purpose | Columns |
|---|---|
| Identity and splits | `game_id`, `play_id`, `season` |
| Input | `desc`, `posteam` |
| Play shape | `play_type`, `complete_pass`, `aborted_play`, `penalty`, `replay_or_challenge`, `replay_or_challenge_result` |
| Players | `passer_player_name`, `receiver_player_name`, `rusher_player_name`, `td_player_name`, `fumbled_1_player_name`, `lateral_receiver_player_name`, `lateral_rusher_player_name` |
| Values | `passing_yards`, `receiving_yards`, `rushing_yards`, `touchdown`, `pass_touchdown`, `rush_touchdown`, `interception`, `fumble_lost`, `two_point_attempt`, `two_point_conv_result`, `lateral_reception`, `lateral_rush` |

### Ground-truth builder

`data/ground_truth.py` converts each row into the output schema. This module gets
its own tests, because every downstream number depends on it.

**Known noise sources:**

- `receiving_yards` and `rushing_yards` *exclude* yards gained after a lateral, and
  the lateral columns record only the *last* lateral. Lateral plays therefore have
  imperfect ground truth. They are tagged as a bucket and reported separately.
- Some structured fields may disagree with the text. This is caught by the audit below.

**Two validation layers:**

1. **Hand audit.** You read 200 plays, stratified by bucket, and check the
   ground-truth output against the text yourself. The per-bucket error rate is
   published.
2. **Game-level cross-check.** Sum the per-play credits into per-player game totals.
   Compare them with nflverse's official weekly player stats, which include
   `fantasy_points` and `fantasy_points_ppr` (verify the exact names in P1). Any
   disagreement is builder error and gets fixed or documented.

### Splits (by season, frozen before training)

| Split | Seasons | Purpose |
|---|---|---|
| train | 2015–2022 | Training and sweeps. Sweeps run on a fixed 50k-play subset. |
| val | 2023 | Early stopping and knob selection |
| test | 2024 | Touched only for final ladder numbers |

Splitting by time mimics deployment, where the model always parses a season it hasn't
seen. Scope v1 to `play_type ∈ {pass, run, no_play}` plus two-point attempts. Other
play types are filtered out, not labeled as empty.

### Buckets

Every test play gets exactly one tag:
`normal | penalty_nullified | penalty_stands | fumble | interception | lateral | challenge | two_point | td`.
Precedence goes from rarest to most common, so a lateral touchdown counts as `lateral`.
**Every metric is reported per bucket.** The headline numbers alone would hide the
interesting part.

## 6. Evaluation

### Metrics

| Metric | Definition | Why it matters |
|---|---|---|
| Valid-output rate | Output parses and matches the schema | The first effect fine-tuning has on behavior |
| Play exact match | Credits list equals ground truth exactly | Strict quality |
| Credit F1 | Precision and recall over individual credits | Partial credit, shows *which* credits fail |
| Game fantasy-point MAE | Mean absolute error of per-player game PPR points, after summing the plays | What a fantasy user actually feels |
| $ / 1k plays | API cost, or GPU-seconds × rental rate | Economics |
| p50 / p99 latency | Measured on your serving layer, or on the API for the frontier rung | Serving cost |
| Eng-hours | A running log of time spent per rung | The honest cost of each approach |

### Protocol

- The test set and the eval code are hashed. Every result file carries the eval hash,
  the config hash, the git SHA, and the Slurm job id.
- Decoding is greedy at temperature 0 for every model rung.
- Confidence intervals are bootstrapped over *games*, not plays, since plays within a
  game are correlated.

## 7. The decision ladder

| Rung | System | Question it answers |
|---|---|---|
| R0 | Hand-written regex parser, with a hard one-day time budget | Do you need ML at all? |
| R1 | Llama 1B, zero-shot with the schema in the prompt | What does the base model already know? |
| R2 | Llama 1B, 8 fixed few-shot examples | How far does prompting go? |
| R3 | Llama 1B, 8 *retrieved* similar plays as examples | Does retrieval beat fixed examples? This is the RAG rung. |
| R4 | Frontier model (Claude), few-shot | What is the quality ceiling, and what does it cost? |
| R5 | **LoRA on ground truth** | Does fine-tuning win? |
| R6 | QLoRA on ground truth | What does 4-bit cost? |
| R7 | Full fine-tune of the 1B model on ground truth | Does LoRA leave quality on the table? |
| R8 | LoRA on teacher (R4) labels, unfiltered | What does distillation cost without ground truth? |
| R9 | LoRA on teacher labels, rejection-sampled | How much does filtering recover? |

R7 is feasible because the model is small. Full fine-tuning of 1.2B parameters fits
on one H100. That turns "when does LoRA underperform full fine-tuning" from an
awareness topic into a measurement.

## 8. Training system

### 8.1 LoRA layer (P0)

`lora/lora_linear.py` wraps a frozen `nn.Linear`:

```
y = x·Wᵀ + (α/r) · dropout(x)·Aᵀ·Bᵀ
A ~ Kaiming-uniform (r × in),  B = 0 (out × r)   → the update starts at exactly zero
```

- `lora/inject.py` replaces target modules by name pattern: `q_proj`, `k_proj`,
  `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- Adapter save and load use PEFT's safetensors format and key names. That means
  vLLM can load your adapters directly as the serving reference (L7).
- **Acceptance criteria:**
  - Logits match PEFT with identical init, to <1e-5.
  - Only A and B have `requires_grad`.
  - A parameter and optimizer-memory table covers full fine-tuning vs r ∈ {4, 8, 16, 64}.

### 8.2 Data and collation (P1–P2)

- Prompts are rendered with the model's chat template. The completion is the JSON plus EOS.
- **Completion-only loss:** prompt-token labels are set to `-100`.
- Right-padding for training. Packing is off in v1, so masking stays simple.
- **Required tests:**
  - The mask covers exactly the completion tokens.
  - The BOS token is not duplicated after templating.
  - EOS is present in the labels.
  - No example is truncated. Assert this; don't truncate silently.

### 8.3 Training loop (P2)

- A hand-written loop with AdamW, cosine schedule, warmup, gradient clipping, and
  bf16 autocast.
- **Gradient accumulation** with correct loss normalization: divide by the total
  number of completion tokens across micro-batches, not per micro-batch.
  - Test: 4 micro-batches × accumulation 4 must give the same gradient as one batch
    of 16, within tolerance.
- Checkpoints are adapter-only, plus optimizer state for resume.
  - Test: resume, then continue training, must reproduce an uninterrupted run's loss.
- Logging covers train loss, val loss, val exact match every N steps, tokens/sec, and
  peak memory.

### 8.4 Default config and sweep plan (P3)

These defaults are common starting values. The sweeps exist to confirm or overturn
them on this task.

| Knob | Default | Sweep (one knob varied at a time, on the 50k subset) |
|---|---|---|
| rank r | 16 | 2, 4, 8, 16, 64 |
| alpha α | 32 (α = 2r) | α = r, α = 2r, fixed α = 16 across ranks |
| targets | all-linear | {q,v}, {q,k,v,o}, all-linear, at matched trainable-parameter budget too |
| dropout | 0.05 | 0, 0.05, 0.1 |
| learning rate | 2e-4 | 5e-5, 1e-4, 2e-4, 5e-4, 1e-3 |
| effective batch | 64 | — |
| epochs | 2 | Early stop on val exact match |
| training-set size | full | 1k, 5k, 20k, 100k, full (the data-size curve) |

**Deliverable:** a table plus written observations for each knob. For example: what
overfitting looked like (train loss down, val exact match flat), and where the rank
curve flattens.

### 8.5 QLoRA (P4)

- The base model is loaded in 4-bit NF4 via bitsandbytes, with double quantization.
  The adapter trains in bf16.
- Compared against R5 on quality, step time (the dequantization overhead), and peak
  memory.
- **Stretch:** QLoRA on Llama 3.2 3B and 8B. At 1B, QLoRA isn't *needed*. Running it
  at a larger size shows the regime it was invented for.

### 8.6 Distillation (P4)

- **Teacher:** R4's frontier model, using the same few-shot prompt, with the shared
  prefix prompt-cached to cut cost.
- **Unfiltered (R8):** one teacher sample per play.
- **Rejection-sampled (R9):** k = 3 samples per play. A label is kept only if it
  passes all of these checks:
  1. Schema valid.
  2. Every yardage value appears literally in `desc`.
  3. Every player name appears literally in `desc`.
  4. All k samples agree.
- **Ground truth is never used in the filter.** A real distillation setting has no
  ground truth.
- The filter's *precision* is reported against ground truth: of the labels it kept,
  how many were correct? That is the measurement this dataset uniquely allows.
- **Deliverable:** student exact match vs number of training examples, with three
  curves (ground truth, raw teacher, filtered teacher) on one plot.

### 8.7 Forgetting check (P6)

Run a small general benchmark, the same one before and after fine-tuning. A slice of
MMLU or HellaSwag through `lm-eval-harness` is enough. Report the drop. This is
catastrophic forgetting (the model losing general ability while it learns the new
task), measured on your own adapter.

## 9. Serving system

### 9.1 Single adapter in the engine (P5a) — `llm_inference_engine`

- `engine/lora.py` loads a PEFT-format adapter and maps its keys to engine weights.
- **Merged mode:** `W' = W + (α/r)·B·A` at load time, with zero runtime cost.
- **Unmerged mode:** the low-rank path is added inside `linear()` in
  `engine/components_gpu.py`, the existing quantization chokepoint. That means
  int8 and int4 bases get LoRA with no extra work.
- Oracle: engine output must equal HF + PEFT output, in the same style as the existing
  correctness suite.

### 9.2 Many adapters in the serving layer (P5b) — `llm_serving_layer`

Each request carries an `adapter_id`. For one linear layer with a mixed batch:

```
y = x·Wᵀ                        ← shared base matmul (unchanged)
  + sᵢ · (x·Aᵢᵀ)·Bᵢᵀ            ← per-row low-rank path, i = that row's adapter
```

- **Adapter pool:** fixed GPU slots with LRU eviction to host memory. This is the
  paged-KV idea applied to adapters. Requests whose adapter isn't resident wait;
  they are not rejected.
- **Kernels:** v1 loops over the unique adapters in a batch. v2 uses gather + batched
  matmul (the BGMV pattern from Punica). A stretch v3 is a Triton or CUDA BGMV kernel,
  which follows on naturally from your decode-attention kernel.
- **Prefix cache:** the radix-tree key must include `adapter_id`. Otherwise a cached
  prefix computed under adapter A gets reused for adapter B. Hidden states differ per
  adapter, so that reuse would be a silent correctness bug.
- **Adapters available for traffic:**
  - The real NFL PlayParse adapter, in several registry versions.
  - Optionally a college-football variant trained on cfbfastR play-by-play. That is a
    different text format, so it is a genuinely different adapter. Verify the data's
    availability first.
  - Random-initialized synthetic adapters at r ∈ {8, 16, 64} for the N = 16–256 scale
    tests. The S-LoRA paper load-tests the same way.

### 9.3 Registry and eval gate (P6)

- `registry/` stores each adapter version with its eval report, config hash, and
  training data hash.
- **Promotion rule:** a candidate must beat or tie the current version on play exact
  match, and must not regress any bucket by more than a set tolerance. A regression in
  a single bucket counts, not only the overall average.
- Rollback is a pointer swap. The serving layer picks up the change on its next
  adapter load.
- **Demo:** deliberately train a broken adapter, with the prompt-token mask turned
  off, and show the gate rejecting it.

## 10. Claims (earned / not earned)

Each claim gets a row in the final README with its result and artifact path. Expect
some to come back not earned, and publish them anyway.

### Adapter track

| | Claim | Earned if |
|---|---|---|
| T1 | Fine-tuning beats prompting at 1B | R5 exact match > R2 and R3, with non-overlapping CIs |
| T2 | A 1B adapter matches the frontier model | R5 within CI of R4, at a measured fraction of the cost |
| T3 | ML beats the regex where it matters | R5 > R0 on the hard buckets (lateral, fumble, penalty, challenge) |
| T4 | QLoRA costs little quality | R6 within 1 point of exact match of R5 |
| T5 | LoRA matches full fine-tuning on this task | R5 within CI of R7 |
| T6 | Filtering recovers distillation quality | R9 closes most of the R8 → R5 gap at a stated data size |
| T7 | Fine-tuning causes little forgetting | General-benchmark drop below a threshold stated in advance |

### Serving track

| | Claim | Earned if |
|---|---|---|
| L1 | Merged adapter adds no latency | TTFT and decode tok/s within noise of the base model |
| L2 | Merged equals unmerged | Greedy tokens are identical, logits within fp16 tolerance |
| L3 | Unmerged overhead is small at low rank | Decode overhead measured at r = 8/16/64, batch 1 and 32, with the threshold stated in advance |
| L4 | Multi-LoRA scales with N | Goodput at N = 1/4/16/64/256, uniform vs Zipf-skewed popularity, v1 vs v2 kernel |
| L5 | Adapters beat one merged model per tenant | Maximum tenants per H100: merged copies vs adapter pool |
| L6 | The prefix cache is adapter-safe | Zero cross-adapter hits under fault injection |
| L7 | Your serving layer is competitive with vLLM multi-LoRA | Same workload, goodput within a stated factor |

The most likely not-earned results are L4 with the v1 kernel at large N under uniform
popularity, and L7.

## 11. Phases and milestones

| Phase | Scope | Exit criterion | Est. |
|---|---|---|---|
| **P0** | `LoRALinear`, injection, PEFT oracle, parameter/memory table | Oracle tests green | 1 wk |
| **P1** | Data loader, ground-truth builder, audit, cross-check, buckets, scorer, eval harness, R0–R4 | Eval hashed and frozen. Baseline rows filled in. | 2 wks |
| **P2** | Collation and masking, training loop, gradient accumulation, checkpoints | All loop tests green. Loss falls on a 1k-play overfit test. | 1 wk |
| **P3** | R5, full sweep, data-size curve (ground truth), R7 | Sweep table plus written notes. T1–T3 and T5 resolved. | 2 wks |
| **P4** | R6 QLoRA, R8–R9 distillation | Distillation plot. T4 and T6 resolved. | 2 wks |
| **P5a** | Engine LoRA, merged and unmerged | L1–L3 resolved | 1 wk |
| **P5b** | Serving-layer multi-LoRA, adapter pool, prefix-cache keying, vLLM comparison | L4–L7 resolved | 3 wks |
| **P6** | Registry, eval gate, forgetting check, README, boundaries section | T7 resolved. README published. | 1 wk |

That totals about 13 weeks. P0–P2 can start now. P5a only needs one adapter, so it
can run in parallel with P3–P4.

## 12. Repository layout

```
llm_finetuning/
├── PRD.md, LEARNING_MAP.md, README.md
├── data/          load_pbp.py · ground_truth.py · buckets.py · splits.py · audit/
├── ffscore/       schema.py · scorer.py · scoring_configs/
├── lora/          lora_linear.py · inject.py · qlora.py · io.py (PEFT-format save/load)
├── train/         collate.py · loop.py · configs/
├── distill/       teacher.py · reject.py
├── eval/          harness.py · metrics.py · bootstrap.py · baselines/{regex_parser.py, prompts/, retrieval.py}
├── registry/      registry.py · gate.py
├── scripts/       slurm/ (PACE job scripts)
├── results/       committed artifacts behind every number
└── tests/
```

The serving changes land in the sibling repos (`llm_inference_engine/engine/lora.py`
and `llm_serving_layer/serving/…`), each with its own tests and version tag.

## 13. Compute and cost

- **Training:** roughly 380k train plays × ~150 tokens ≈ 57M tokens per epoch. A
  LoRA epoch on a 1B model should take on the order of an hour on one H100. Measure
  this in P2 and replace the estimate. Sweeps use the 50k subset, at a few minutes per run.
- **Teacher labels:** capped at 100k plays × k = 3 samples, with the few-shot prefix
  prompt-cached. Price a 1k-play pilot before committing to the full run.
- **Benchmarks:** H100 or H200 on PACE, the same as `llm_serving_layer`. Every
  artifact records its job id.

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| The regex is nearly as good as the adapter | Report it as a result. That is the decision-tree answer. If the gap is tiny everywhere, add a harder input format: injury/news blurbs → status, with teacher labels. |
| Ground-truth noise, especially on laterals | Hand audit, game-level cross-check, a separate lateral bucket, and published error rates |
| The base model memorized nflverse text | R1 measures what it already knows, and it is reported |
| Output JSON is token-heavy, which hurts latency | Optional ablation: a compact line format vs JSON, comparing exact match and tokens per play |
| Multi-LoRA scope balloons | v1 loop kernel plus the pool are the commitment. v2 and v3 kernels are stretch goals. |
| Teacher cost overruns | The 1k-play pilot sets the budget, and the cap is enforced in code |

## 15. Boundaries (for the final README)

The final README gets one paragraph each on: full fine-tuning (measured here as R7),
RLHF and DPO (what they are for, and why they are not needed for an extraction task),
catastrophic forgetting (measured as T7), and when LoRA underperforms (large domain
shifts; T5 shows whether this task counts as one). Multi-node training is out of
scope. It is training-side infrastructure, a different lane from serving.

## 16. Decisions log

| # | Question | Decision (2026-10-04) |
|---|---|---|
| 1 | Credit names: `desc`-style or player IDs? | **`desc`-style names** (`A.Brown`). Resolution to IDs, if ever needed, happens outside the model. |
| 2 | Include penalty yardage on `penalty_stands` plays? | **No.** Penalty yards are ignored, matching how fantasy stats are scored. |
| 3 | Which frontier model is the teacher / cheap API rung? | **Deferred.** Decide after pricing the 1k-play pilot (needs an API key; see `docs/BLOCKERS.md`). |
