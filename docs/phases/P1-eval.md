# P1 (eval half): harness, metrics, bootstrap, and baselines R0–R4

This is the eval side of Phase P1. It covers the scoring code, the harness that runs
any ladder rung and writes a traceable result file, the regex baseline (R0), and the
prompting, retrieval, and frontier baselines (R1–R4). The ground-truth dataset is
built in a separate workstream. Everything here builds against that dataset's agreed
JSONL format and is tested on hand-made fixtures, so the two halves can merge in
either order.

## What was built

| File | Role |
|---|---|
| `playparse/eval/extract.py` | Pulls the first JSON object out of raw model text |
| `playparse/eval/metrics.py` | Per-play scoring, per-game sums, metric formulas |
| `playparse/eval/bootstrap.py` | Game-level (cluster) bootstrap CIs and paired differences |
| `playparse/eval/_ppr.py` | Stand-in full-PPR scorer (the default `score_fn`) |
| `playparse/eval/harness.py` | `run_eval`, the `Predictor` protocol, artifacts, provenance |
| `playparse/eval/run.py` | CLI: `python -m playparse.eval.run --rung r0..r4` |
| `playparse/eval/baselines/regex_parser.py` | R0 |
| `playparse/eval/baselines/hf_predictor.py` | R1 (zero-shot), R2 (fixed few-shot), R3 wiring |
| `playparse/eval/baselines/fewshot_r2.json` | R2's eight committed train examples |
| `playparse/eval/baselines/retrieval.py` | R3's TF-IDF retriever |
| `playparse/eval/baselines/frontier.py` | R4 Anthropic client |
| `scripts/regex_dev_estimate.py` | R0 dev-set estimate on train seasons |
| `scripts/bench_local_throughput.py` | Local R1/R2 throughput and label-length stats |

## Input format

The harness reads one JSON record per play:

```json
{"game_id": "...", "play_id": 1234, "season": 2024, "week": 3, "season_type": "REG",
 "posteam": "PHI", "desc": "...", "bucket": "fumble", "label": "<PlayLabel.to_json()>"}
```

`load_records` rejects any record missing `game_id`, `play_id`, `desc`, `bucket`, or
`label`. `(game_id, play_id)` must be unique, because it is the cache key.

## Metric definitions, and why each one exists

Every rung's raw output goes through the same two steps. First, extraction takes the
first complete JSON object in the text, skipping prose and markdown fences. Second,
`PlayLabel.from_json` validates it against the schema. Using one path for all rungs
keeps "valid" meaning the same thing across the whole ladder.

| Metric | Definition | Why |
|---|---|---|
| Valid-output rate | Extraction found an object and it passed the schema | The first thing fine-tuning changes. Base models often fail here before they fail on content. |
| Strict-valid rate | The whole stripped output *is* the JSON object | Shows whether a model learned the output contract or is being rescued by lenient extraction. A fine-tuned adapter should be near 100%. |
| Play exact match | Same `nullified` flag and the same multiset of credits (`PlayLabel.matches`) | The strict headline quality number. Order does not matter, duplicates do. |
| Credit precision / recall / F1 | Over individual credits as multisets, micro-averaged | Partial credit. Shows *which* credits fail: hallucinations lower precision, misses lower recall. |
| Game fantasy-point MAE | Mean over (game, player) of the absolute PPR-point error after summing each player's credits for the game | What a fantasy user actually feels. Many small yardage slips may cost little, while one missed touchdown costs 6 points. |
| p50 / p99 latency | Wall time of the batch a play was in | The wait a request in that batch would see. Throughput (plays/sec) is reported separately. |
| $ / 1k plays | Sum of predictor-reported `cost_usd`, or prediction time × `--gpu-usd-per-hour` | Puts API and local rungs on one economic axis. |

Details that change the numbers:

- **Invalid output counts as "no credits".** Every gold credit becomes a false
  negative, and the play cannot exact-match. A model that crashes into garbage is
  never rewarded with an undefined (skipped) score.
- **A nullified gold play has no credits.** Predicting credits on it adds false
  positives, so precision catches models that ignore "No Play".
- **Fantasy MAE uses the union of players** credited in gold or in the prediction. A
  hallucinated or misspelled player (`D.Smyth`) counts as error twice: once for the
  real player's missing points and once for the invented player's points.
- **Fantasy MAE is overall-only.** It is a per-game quantity and cannot be split by
  bucket. Bucket tables show "-" for it.
- **Players are keyed by their desc-style name.** Two players with the same
  abbreviated name in one game would merge. That affects gold and prediction the same
  way, so it does not bias the comparison.
- **`score_fn` takes per-player stat totals**, not single credits. Real leagues have
  non-additive rules (a 100-yard bonus, for example) that only make sense on totals.
  When `playparse/ffscore/scorer.py` lands, pass its function as `score_fn`.

## Confidence intervals: cluster bootstrap over games

Plays in one game share a quarterback, a defense, the weather, and a play-caller, so
their errors are correlated. Resampling plays as if they were independent gives
intervals that are too narrow.

`bootstrap.py` resamples whole games with replacement. It works because every metric
above is a *ratio of sums*. For example, exact match = (exact plays) / (plays), and
credit F1 = 2·TP / (2·TP + FP + FN). So each game is reduced to one row of sums
(plays, valid, exact, TP, FP, FN, absolute fantasy error, player count). One
bootstrap resample is then a weighted sum of rows (`weights @ matrix`), and 1000
resamples are a single matrix product.

- **Seeded and deterministic.** The weights come from `numpy.random.default_rng(seed)`.
- **Shared resamples.** Overall and every bucket use the *same* resampled games
  within one run. Two rungs scored on the same games with the same seed also see the
  same resamples. `paired_diff_ci` uses this to put a CI on the difference between
  two rungs. That is the right test for claims like T1 ("R5 > R2 with non-overlapping
  CIs"), and it is tighter than comparing two separate intervals.
- **Percentile intervals.** Resamples where a metric is undefined (for example,
  precision with no predicted credits) are dropped for that metric.

The test `test_cluster_ci_wider_than_row_ci_on_correlated_data` builds data where some
games are easy and some are hard. The cluster CI comes out more than 2.5× wider than
the naive row CI on the same data, which is the failure the cluster bootstrap exists
to prevent.

## Harness design

```python
class Predictor(Protocol):
    name: str
    batch_size: int
    def config(self) -> dict: ...                       # everything that changes outputs; hashed
    def predict_batch(self, records) -> list[str | Prediction]: ...

@dataclass
class Prediction:
    text: str
    usage: dict[str, float]   # input_tokens, output_tokens, ..., optional cost_usd

run_eval(predictor, records, *, rung, out_dir=None, eval_path=None, n_boot=1000,
         seed=0, alpha=0.05, score_fn=ppr_points, gpu_usd_per_hour=None,
         resume=True, extra_meta=None, progress=None) -> dict
```

The harness never looks inside a predictor. It owns everything that must be the same
for every rung:

- **Batching and timing.** Each `predict_batch` call is timed with `perf_counter`.
- **Resumable caching.** Each batch's rows are appended to
  `<out>/predictions.jsonl` and flushed. A restarted run reuses rows whose
  `config_hash` matches and only predicts the rest. If the file holds rows from a
  *different* config, the run refuses to continue rather than mix two systems in one
  result. Use `resume=False` (CLI `--no-resume`) to start over.
- **Provenance.** `result.json` records the eval-file sha256 (or a hash of the
  records when there is no file), a sha256 over the eval code (`extract`, `metrics`,
  `bootstrap`, `harness`, `_ppr`, and `schema`), the config and its hash, the git sha
  and dirty flag, hardware (platform, CPU, RAM, accelerator when torch is loaded,
  `SLURM_JOB_ID`), the UTC date, and the bootstrap settings.
- **Outputs.** `result.json` holds overall and per-bucket metrics, each as
  `{point, lo, hi}`, plus latency, throughput, cost, and usage totals.
  `predictions.jsonl` is rewritten at the end with the scoring attached: gold,
  parsed prediction, valid, exact, TP/FP/FN, and the parse error.

## R0: regex baseline

### Design

`parse_desc(desc, posteam)` works in a fixed order:

1. If the text says `the play was REVERSED.`, keep only the text after it. That is
   the play that counts.
2. `No Play` anywhere means nullified, with no credits.
3. Cut the text at the first `PENALTY on`. Penalties that stand do not change the
   stats, and their text names players who must not be credited.
4. Two-point tries: only `ATTEMPT SUCCEEDS` counts (a defensive return after the try
   is ignored). Credit `two_pt` to the passer and receiver, or to the rusher.
5. Drop preambles that name a player before the action (`Direct snap to ...`,
   `... in at QB`).
6. Find the first offensive player (`jersey-Name`, not team-prefixed). The verb after
   the name decides the play type: `pass` (complete, incomplete, or intercepted),
   `sacked`, a botched snap (`FUMBLES (Aborted)`), `spiked`, or otherwise a run.
7. Compute yards, touchdowns, laterals, and fumbles. A fumble is lost when the
   recovering team differs from `posteam`. Without `posteam`, it falls back to the
   capitalized `RECOVERED by` that the text uses for a change of possession.

The surprise was step 7's yardage. Official yards differ from the stated "for N
yards" after downfield offensive fouls and backward fumbles, and the regex computes
both from field spots. See `docs/issues/p1-eval-official-yards-vs-stated-gain.md`.

### Dev-set estimate (train seasons 2015–2022, NOT the test set)

These numbers come from `scripts/regex_dev_estimate.py --seasons 2015-2022`. They are
scored against an **approximate** labeler built from the structured columns, not the
frozen ground truth (see `docs/issues/p1-eval-gt-convention-questions.md`). The regex
was also developed against these same seasons, so treat the result as an optimistic
upper bound. The real R0 number is the one measured on the 2024 test split once the
dataset is frozen.

Artifact: `results/r0_dev_train/result.json` (293,986 plays, 2,173 games, 200
cluster-bootstrap resamples).

| Bucket | Plays | Exact match % [95% CI] | Credit F1 % |
|---|---|---|---|
| **Overall** | 293,986 | **99.82** [99.80, 99.85] | 99.84 |
| normal | 248,258 | 99.95 [99.93, 99.97] | 99.95 |
| penalty_nullified | 20,307 | 100.00 [100.00, 100.00] | – |
| td | 10,016 | 99.91 [99.86, 99.96] | 99.96 |
| fumble | 4,092 | 93.94 [93.28, 94.75] | 95.09 |
| penalty_stands | 3,961 | 99.85 [99.72, 99.95] | 99.88 |
| interception | 3,322 | 100.00 [100.00, 100.00] | 100.00 |
| challenge | 2,925 | 99.04 [98.71, 99.37] | 99.37 |
| two_point | 994 | 97.69 [96.70, 98.62] | 97.92 |
| lateral | 111 | 37.84 [30.40, 45.88] | 70.07 |

Game fantasy-point MAE (PPR): 0.021 points per player-game [0.016, 0.026]. R0
processes about 100,000 plays/sec on one CPU core.

**What this means for the ladder.** On this text format, a one-day regex is nearly
perfect on the common buckets. PRD risk #1 ("the regex is nearly as good as the
adapter") looks likely to come true for `normal`, `td`, `interception`, and
`penalty_*`. The headroom for ML is in `fumble` (6% misses), `two_point`,
`challenge`, and `lateral`. That is where claim T3 will be decided. Two caveats make
the regex look better here than it may be on the real eval:

- `penalty_nullified` is 100% partly by construction. Both the approximate labeler
  and the regex key off "No Play".
- The approximate labeler shares the regex author's reading of the conventions.

### Effort log (one-day budget)

About 4 hours in total, inside the budget:

| Step | Time |
|---|---|
| Read sample descs per bucket, write the first parser | ~1.5 h |
| First dev run (98.6% on 2018); fix preambles, names with spaces, "TOUCHDOWN NULLIFIED" | ~0.5 h |
| Spot-based yardage for offensive fouls and fumbles, including the escaping bug | ~1.25 h |
| Bobbled-snap-then-pass, defensive two-point, fumble touchdowns; unit tests | ~0.75 h |

Deliberately not done: multi-word surnames (`H.Krieger Coble`), handoffs after a
bobbled snap, fumbles out of the end zone, and lateral yardage conventions.

## R1–R3: prompting and retrieval baselines

All three use Llama 3.2 1B Instruct through HF transformers (`HFPredictor`):

- **Prompt.** `build_fewshot_messages` calls `playparse.prompt.build_messages` for
  the system prompt and for every user turn, so the format matches what a fine-tuned
  adapter will see. Few-shot examples are prior user/assistant turns. With no
  examples, the result is exactly `build_messages` (R1).
- **Chat template.** It is rendered by the tokenizer with `date_string` pinned (see
  `docs/issues/p1-eval-chat-template-date.md`). The text is tokenized with
  `add_special_tokens=False` because the template already emits
  `<|begin_of_text|>`. A test checks for exactly one BOS.
- **Decoding.** Greedy, requested explicitly because the checkpoint defaults to
  sampling (`docs/issues/p1-eval-generation-config-sampling.md`). Batched with left
  padding, using the reserved `<|finetune_right_pad_id|>` token as the pad. Generation
  stops on any of the checkpoint's EOS ids.
- **`max_new_tokens = 256`.** Gold labels in train seasons are p50 26, p99 88, and
  max 105 tokens in compact form. Base models pretty-print, roughly doubling the
  length, so 256 gives headroom. Usage records `hit_max_new_tokens`, so truncation is
  visible in results.
- **Device and dtype.** MPS with fp16 on the Mac, CUDA with bf16, CPU with fp32.
- **R2** uses `fewshot_r2.json`: eight real 2020 (train) plays covering a pass, a run,
  an incompletion, a passing TD, an interception, a lost fumble, a holding no-play,
  and a two-point pass. They are hand-checked, canonical, and their hash is in the
  config.
- **R3** uses `TfidfRetriever` from scikit-learn, which is already installed. Before
  indexing, descriptions are normalized (player tokens become `PLAYER`, team spots
  become `SPOT`, digits become `#`), so similarity reflects the *shape* of a play
  rather than who was in it. It uses word 1–2 grams with sublinear TF and cosine
  similarity. The query's own game is excluded, so the query and its teammates never
  leak in. Examples are ordered least similar first, so the closest one sits right
  before the query. Build the index from a train JSONL, with `--index-size` to
  subsample.

Local throughput on the M4 is in `docs/benchmarks/p1-local-throughput.md`.

## R4: frontier model (built, not run)

`AnthropicPredictor` (`frontier.py`):

- The model id is a **required parameter**, because the choice is deferred (PRD
  decision #3).
- It uses the same R2 few-shot examples and the same message builder. The last
  example turn carries a `cache_control: ephemeral` breakpoint, so the shared prefix
  (system prompt plus examples) is cached across plays. If the prefix is below the
  model's minimum cacheable length (512–4096 tokens depending on the model),
  `cache_read_input_tokens` stays at 0. Check it in the pilot.
- Usage becomes cost through a `Price` table (input, output, cache write at 1.25×,
  cache read at 0.1× by default). The defaults are list prices for a few current
  models, and `prices=` overrides them.
- It retries 408/409/429/5xx and connection errors with exponential backoff, and does
  not retry other 4xx errors.
- `temperature` is sent only when set, because the newest models reject sampling
  parameters. `extra_create_kwargs` passes model-specific options such as thinking or
  effort.
- It reads `ANTHROPIC_API_KEY` (or any SDK credential) only when no client is
  injected. All tests use a mock client.

## How to run each rung

Run from the repository root with `python -m`, so the worktree's code is the code
that runs (see `docs/issues/p1-eval-editable-install-shadows-worktree.md`).

```bash
# R0
python -m playparse.eval.run --rung r0 --data <eval.jsonl> --out results/r0_<split>
# R1 / R2 (set PLAYPARSE_WEIGHTS if the weights are not in the default location)
python -m playparse.eval.run --rung r1 --data <eval.jsonl> --out results/r1_<split> --batch-size 8
python -m playparse.eval.run --rung r2 --data <eval.jsonl> --out results/r2_<split> --batch-size 8
# R3
python -m playparse.eval.run --rung r3 --data <eval.jsonl> --train <train.jsonl> --out results/r3_<split>
# R4 (needs ANTHROPIC_API_KEY)
python -m playparse.eval.run --rung r4 --model <model-id> --data <eval.jsonl> --out results/r4_<split>
# Smoke run on a few plays
python -m playparse.eval.run --rung r1 --data tests/fixtures/eval_mini.jsonl --out /tmp/r1_smoke --limit 5
# R0 dev estimate (train seasons only; the script refuses others)
python scripts/regex_dev_estimate.py --seasons 2015-2022 --out data/cache/r0_dev_train
```

Rerunning a command resumes it. Add `--gpu-usd-per-hour` to price local compute.

## Tests

`tests/test_eval_metrics.py`, `test_eval_bootstrap.py`, `test_eval_harness.py`,
`test_regex_parser.py`, `test_eval_baselines.py`, and `test_hf_predictor.py`. The
default suite runs in a few seconds. The tokenizer tests skip when the weights are
absent, and the one slow test (`test_r1_smoke_on_mps`) loads the real 1B model on
MPS.

## Issues found

- `docs/issues/p1-eval-chat-template-date.md`: the prompt drifted daily.
- `docs/issues/p1-eval-generation-config-sampling.md`: the checkpoint samples by default.
- `docs/issues/p1-eval-official-yards-vs-stated-gain.md`: official yards differ from the text.
- `docs/issues/p1-eval-spot-foul-regex-escape.md`: a dead regex failed silently.
- `docs/issues/p1-eval-gt-convention-questions.md`: conventions to reconcile with the builder.
- `docs/issues/p1-eval-editable-install-shadows-worktree.md`: stale-code risk in scripts.

A smaller one: the bootstrap determinism test first compared results with `==`, and
NaN never equals NaN. It now compares JSON serializations.

## Open items

- Rerun R0 on the real ground truth when it lands, and diff the result against
  `results/r0_dev_train/result.json`.
- Swap `_ppr.ppr_points` for the real scorer through `score_fn`.
- R4 needs an API key: run a 1k-play pilot, check cache hits, and pick the model.
- Run R1–R3 on eval_lite. Local runs are feasible but slow (see the benchmark doc),
  and the H100 is better for the full test split.
