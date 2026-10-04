# P4 — Distillation data pipeline (teacher labels and rejection sampling)

**Status:** built and tested against a mock teacher. No real teacher labels yet: the
1k-play pricing pilot needs an API credential (blocker B2). Everything up to that
call is ready, including `scripts/p4_pilot.py --dry-run`.

Code: `playparse/distill/reject.py`, `playparse/distill/teacher.py`,
`scripts/p4_pilot.py`. Tests: `tests/test_distill_reject.py`,
`tests/test_distill_teacher.py`, `tests/test_p4_pilot.py`.

## What distillation means here

Distillation is training a small model (the *student*, our 1B adapter) on labels
produced by a large model (the *teacher*, a frontier model). It matters because most
real tasks have no ground truth. This project is unusual: nflverse gives us ground
truth for every play, so we can train on both kinds of labels and *measure* how much
quality the teacher route loses. Rungs R8 and R9 answer that:

- **R8:** one teacher sample per play, used as is.
- **R9:** three samples per play, kept only if they pass a filter.

Claim T6 is earned if R9 closes most of the gap between R8 and R5 (training on
ground truth).

## Why rejection sampling works

Rejection sampling here means: generate several candidate labels, and throw away the
play unless the candidates pass checks that a correct label would pass. It works
because teacher errors are not all alike.

- **Some errors are detectable from the input alone.** If the teacher credits
  22 receiving yards but the play text says "to DAL 22 for 14 yards", the number 22
  came from the yard line. A correct label's yardage always appears in a "for N
  yards" phrase, so a value that does not is evidence of a mistake.
- **Random errors rarely repeat.** If the teacher is unsure, independent samples
  tend to disagree. Requiring all three to agree removes many of those guesses.
- **What remains is mostly right, but not all.** A teacher that is *confidently*
  wrong in the same way three times passes the agreement check. A wrong player whose
  name is in the text (say, the tackler) passes the name check. These errors survive,
  which is why the filter's precision must be measured, not assumed.

The cost is yield: every dropped play is a play you paid for and cannot train on.
The drop statistics say how much each check costs.

### The four checks

1. **Schema:** every sample parses as a `PlayLabel`. A surrounding Markdown code fence
   is stripped first, because that is formatting, not a wrong answer.
2. **Yardage:** every `pass_yds`, `rush_yds`, and `rec_yds` value appears in `desc` as
   a gain phrase. Recognized: "for 14 yards", "for 1 yard", "for -3 yards", "for no
   gain" (0), "for a loss of 3 yards" (-3). The sign matters. Field positions ("to DAL
   22") and penalty distances ("Offensive Holding, 10 yards") are not gain phrases, so
   they are not accepted as evidence. On lateral plays every segment's gain counts.
3. **Names:** every credited player appears in `desc` as a whole name. A plain
   substring test would accept "J.Brown" inside "A.J.Brown"; the check uses name
   boundaries instead (see `docs/issues/P4-name-substring-match.md`).
4. **Agreement:** all k samples parse and are the same label, compared with
   `PlayLabel.matches` (order-insensitive).

All enabled checks run on every play, even after one fails. That lets
`drop_stats` report three views: how often each check fails, which check failed
*first* (these sum to the number of drops), and which check was the *only* failure
(what turning that check off would recover).

`FilterConfig.r8()` turns off everything except "sample 0 parses". An unparseable
string cannot become a training target, so even the unfiltered rung needs that.

## Why the filter can never use ground truth

The point of R9 is to estimate what distillation achieves *in a setting without
ground truth*. If the filter consulted ground truth, even indirectly, it would keep
labels no real filter could identify, and the R9 result would be an overestimate
that no one could reproduce on a new task.

The code enforces this structurally:

- `filter_record` reads `record["desc"]` and nothing else.
- The teacher client is shown only `game_id`, `play_id`, `posteam`, and `desc`.
- Training records are built from a fixed list of fields; the dataset's `label` is
  replaced by the teacher's label, never copied.
- Tests pass "tripwire" records that raise if `label` is read, through both the
  filter and the whole pipeline.

Ground truth enters only afterward, in `filter_precision` / `precision_report`. That
measures, of the labels the filter kept, how many were exactly right. This
measurement is something only a dataset like ours allows. Run it on the R8 file too:
R8's precision is the raw teacher accuracy, and the difference is what the filter
bought.

## Orchestration: cache, resume, budget

`label_dataset(records, client, cache_dir, n_samples=3, budget=...)` collects samples
in chunks.

- **Cache.** Each sample is appended to `teacher_cache.jsonl` keyed by
  `(game_id, play_id, sample_idx)`, flushed and fsynced per chunk. A crash loses at
  most the chunk in flight. On rerun, only missing samples are requested; a torn last
  line from a crash is skipped. Running R8 (k=1) first and R9 (k=3) later reuses
  sample 0 and asks only for samples 1 and 2.
- **Teacher id.** Lines carry a `teacher_id` (model plus prompt hash in the pilot
  script). Changing the prompt starts a fresh namespace instead of silently mixing
  labels from two prompts.
- **Budget.** `Budget(max_calls=..., max_usd=..., est_usd_per_call=...)` is a hard cap
  on *cumulative* spend, recorded in `usage.json` beside the cache. Before each chunk
  the projected cost is checked and the chunk is trimmed to fit. Dollar projections
  use the observed cost per call once there is one, and the pilot's estimate before
  that. If the estimate was too low, the overrun is at most one chunk, it is flagged
  in the report, and any resume refuses to spend more.
- **Write order.** The ledger is saved before the samples are appended. A crash in
  between over-counts spend, which errs on the safe side of the cap.

### The client interface

```python
class TeacherClient(Protocol):
    def __call__(self, records: Sequence[Mapping], n_samples: int) -> TeacherBatch: ...

TeacherBatch(outputs: list[list[str]], usage: Usage)
Usage(calls, input_tokens, output_tokens, cache_read_tokens, cost_usd)
```

`outputs[i][j]` is sample j of record i. Samples must be drawn independently with
sampling on; k greedy samples always agree and make the agreement check meaningless.
A client that reports zero calls is charged the worst case (`records × n_samples`).
The frontier-client module can satisfy this with a thin adapter.

## Training files

`build_training_sets(records, cache_dir, out_dir, k=3)` writes, from one cache:

| File | Contents |
|---|---|
| `r8_teacher_raw.jsonl` | Sample 0 of each play whose sample 0 parses; `label_source: "teacher_raw"` |
| `r9_teacher_filtered.jsonl` | Plays passing the full filter; `label_source: "teacher_filtered"` |
| `*.n{1000,5000,20000,100000}.jsonl` | Data-size subsets for the distillation curve |
| `r9_decisions.jsonl` | Per-play keep/drop with each check's reason |
| `build_report.json` | Counts, drop statistics, skipped sizes |

Records have the dataset's format (`game_id, play_id, season, week, season_type,
posteam, desc, bucket, label`) plus `label_source`. `bucket` is kept only for
per-bucket reporting.

**Subsets count training examples**, because the PRD curve plots exact match against
"number of training examples". They are prefixes of one deterministic order: each
play's position comes from a seeded hash of its own key. That gives two properties.
Subsets are nested (the 1k set is inside the 5k set), and every curve (ground truth,
R8, R9) walks plays in the same order. The ground-truth curve should use
`playparse.distill.subset_order` so the three curves are comparable. Because R9
drops plays, a 100k-play teacher run will not yield 100k R9 examples; sizes that do
not fit are skipped and listed in the report.

## The pricing pilot

`scripts/p4_pilot.py --dry-run` estimates tokens from the real prompt with
characters ÷ 4 and prices the pilot and the full run for a few candidate models.
With typical play lengths and an assumed 1,500-token few-shot prefix, it estimates
about $6 per 1k plays (k=3) for Claude Opus 5, $2.50 for Sonnet 5, and $1.30 for
Haiku 4.5, *if the shared prefix is prompt-cached*. Without caching, the figures are
about 4.5 times higher. These are estimates from cached list prices, not
measurements.

Two cautions for the real run:

- Prompt caching only applies above a model-specific minimum prefix length (512 to
  4,096 tokens). If the R4 few-shot prefix is shorter than that minimum, every call
  pays full price. Check `cache_read_input_tokens` in the first chunk.
- Current Opus models think adaptively by default, and thinking tokens bill as
  output. The included client sets effort to `low`; pass `--thinking-tokens` to the
  dry run to see the sensitivity.

To run it once B2 is resolved:

```bash
export ANTHROPIC_API_KEY=...
python scripts/p4_pilot.py --model claude-opus-5 --budget-usd 25 \
    --data data/processed/train.jsonl --few-shot-file <R4 few-shot prefix>
```

It writes `runs/p4_pilot/<model>/pilot_report.json` with measured $/1k plays, the
projected full-run cost, R9 keep rate, and R8/R9 precision.

## Open items

- Run the pilot (B2), then choose the teacher (PRD decision 3) and set the full-run
  budget from measured cost.
- Report filter precision per bucket. The mock shows the expected shape; real
  laterals are where the yardage check is least reliable, since ground truth there is
  itself noisy.
- Consider a fifth check that count stats (`rec`, `pass_td`, ...) have value 1. It is
  cheap, but it is not in the PRD, so it is left out until the pilot shows such
  errors occur.
