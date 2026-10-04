# P6 — Adapter registry and eval gate

**Status:** built and tested with fake adapters and synthetic eval artifacts. It has
not yet gated a real adapter, because none exists yet. The PRD §9.3 demo (train a
broken adapter with the prompt mask off and watch the gate reject it) is waiting on
P2/P3.

Code: `playparse/registry/` (`registry.py`, `gate.py`, `resolver.py`, `__main__.py`).
Tests: `tests/test_registry.py`.

## What problem this solves

Training produces many adapters. Only one per name should be served at a time, and
replacing it must be safe. The registry has three jobs:

1. **Remember every version** with enough metadata to reproduce or audit it: which
   training config and data built it, and which eval scored it.
2. **Decide promotions with a fixed rule** (the eval gate), not by eyeballing numbers.
3. **Undo a bad promotion instantly** (rollback), without retraining or redeploying.

## The eval gate, and what it protects against

The eval gate is the check that decides whether a newly trained candidate may replace
the adapter currently being served. It runs five checks and reports all of them, so a
rejection lists every reason at once.

| Check | Rule | What goes wrong without it |
|---|---|---|
| `eval_file_match` | Candidate and current were scored on the same eval file (sha256 equal) | You compare numbers from different test sets and promote on noise |
| `valid_rate` | Valid-output rate ≥ threshold (default 95%) | A broken adapter that emits unparseable text slips through |
| `min_exact_match` | Overall exact match ≥ a floor (set it to R0's score) | A model that loses to the regex ships |
| `overall_exact_match` | Candidate beats or ties current | Quality silently drifts down release by release |
| `bucket_regression` | No bucket drops more than 1.0 point | A gain on easy plays hides a loss on hard ones |

### Why the eval-file check comes first

Exact match is only comparable when both models answered the *same* questions. If the
eval set was rebuilt (a ground-truth bug was fixed, say), the old adapter's 91% and
the new adapter's 93% measure different things. The gate refuses to compare them and
tells you to re-evaluate both on the same file. The eval harness writes the file's
sha256 into every result, which is what makes this check possible.

### Why per-bucket regressions matter

The overall exact match is an average over buckets of very different sizes. Most
plays are `normal`; laterals, challenges, and fumbles are rare. A worked example:

| Bucket | Plays | Current | Candidate |
|---|---|---|---|
| normal | 1,800 | 95.0% | 96.0% |
| lateral | 40 | 60.0% | 45.0% |
| overall | 1,840 | 94.2% | 94.9% |

The candidate wins overall by 0.7 points, yet it got 6 more laterals wrong. Laterals
are exactly the plays a fantasy user notices, and they are where claim T3 ("ML beats
the regex where it matters") is decided. The PRD therefore says a single-bucket
regression counts. The test
`test_gate_single_bucket_regression_rejects_despite_better_overall` encodes this case.

### Small buckets are handled explicitly

In a 12-play bucket, one play is 8.3 points. Gating on that would reject good
candidates at random. So buckets with fewer than `min_bucket_n` examples (default 30)
are **reported but not gated** by default. The decision's reason string still names
them and their drop, so nothing is hidden. Set `small_bucket_policy="enforce"` to
gate them anyway. A bucket that is present in the current eval but missing from the
candidate's always fails.

### The broken-adapter scenario

If the prompt-token mask is off during training, the loss also trains the model to
reproduce the prompt. The result typically rambles or echoes the play text instead of
emitting JSON, so its valid-output rate collapses. The gate's `valid_rate` check
rejects it even when there is no current version to compare against. The test
`test_gate_broken_unmasked_prompt_adapter_rejected` models it with a 6% valid rate.

### Tolerances are points, and rates are fractions

The tolerance is in **percentage points** (1.0 means 0.01 absolute). Metrics in the
eval artifact must be **fractions** in [0, 1]. The loader rejects values above 1,
because an artifact written in percent would make a 1-point tolerance act like a
100-point one (see `docs/issues/P6-percent-scale-eval-artifact.md`).

## What the eval artifact must look like

The gate reads the eval harness's result JSON:

```json
{
  "metadata": {"eval_file_sha256": "<hex>", "git_sha": "...", "config_hash": "..."},
  "overall":  {"n": 2000, "exact_match": 0.91, "valid_rate": 0.998, "credit_f1": 0.95},
  "buckets":  {"lateral": {"n": 41, "exact_match": 0.56, "valid_rate": 1.0}, "...": {}}
}
```

Required: `overall.exact_match`, `overall.valid_rate`, and the eval-file sha256.
Strongly recommended: `n` per bucket (without it the bucket counts as small).
Accepted variants: `per_bucket` for `buckets`; `eval_sha256` or `eval_file_hash` for
the sha key, at top level or under `metadata`; a metric given as
`{"value": x, "ci": [lo, hi]}` (or `point`, `mean`) instead of a bare number. Extra
fields are ignored.

## Registry design

```
registry/
  index.json                   metadata + promoted stack + history   (commit)
  evals/<name>/<version>.json  copy of each eval artifact             (commit)
  adapters/<name>/<version>/   PEFT adapter files                     (gitignored)
```

- **Weights out of git, metadata in git.** Adapter weights are tens of megabytes and
  reproducible from config + data. The index and eval copies are small and are the
  audit trail, so they belong in version control.
- **Each version records** its parent, creation time, adapter sha256 (over all files),
  training-config sha256 (canonical JSON, so key order does not matter), training
  data sha256, the eval artifact copy and its hash, and the eval-file sha256.
- **History lives inside `index.json`.** Every register, promote, reject, and
  rollback appends an entry with a timestamp, the gate configuration used, and every
  check's result. Keeping it in the same file means one atomic write commits both
  the change and the record of why. A separate log could disagree with the index
  after a crash.

### Atomic writes

Every change takes a file lock, reads `index.json`, edits it in memory, writes the
result to a temp file in the same directory, fsyncs it, and renames it over the
original. A rename within one filesystem is atomic: readers see the old file or the
new one, never a mix. The tests simulate a crash just before the rename and halfway
through writing the temp file; in both cases the old index survives byte-for-byte.

Registering also copies the adapter. The copy goes to a temp directory first and is
renamed into place; if writing the index then fails, the copy is removed. If the
process is killed so hard that cleanup cannot run, the next `register` finds a
directory the index does not mention and clears it.

### Promotion and rollback

`promote(name, version)` is the only way to make a version current, and it always
runs the gate. There is deliberately no "force" flag. On success the version is
pushed onto a *promoted stack*. `rollback(name)` pops the stack and points `current`
at the version below it. That is the whole rollback: a pointer swap, no files move.

### How serving picks it up

The serving layer calls `resolve(name)` whenever it loads an adapter:

```python
from playparse.registry import resolve
adapter_dir = resolve("playparse")   # a PEFT directory
```

`resolve` re-reads `index.json` every time and caches nothing, so a promotion or a
rollback takes effect on the next load. `resolve(name, verify=True)` also re-hashes
the files and fails if they changed since registration.

## CLI

```bash
python -m playparse.registry register playparse runs/r5/adapter --eval results/r5_val.json \
    --train-config configs/r5.json --train-data data/processed/train.jsonl
python -m playparse.registry promote playparse v0002          # exit 2 on reject
python -m playparse.registry rollback playparse --reason "lateral bug found in spot-check"
python -m playparse.registry list | show playparse [v0001] | resolve playparse | history
```

`--root` (or `PLAYPARSE_REGISTRY`) points at a different registry directory.

## Open items

- Run the PRD demo once real adapters exist: register R5, then a mask-off adapter,
  and record the rejection output in this file.
- Decide the production `min_exact_match` floor (R0's test score) and whether the
  gate should run on val or test results. Val is the natural choice; test is touched
  only for final ladder numbers.
- The overall comparison is a plain "beat or tie". If repeated evals of the same
  adapter show noise, consider requiring the candidate to stay within the current
  version's CI instead.
