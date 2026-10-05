# P3: the training val callback scored the first N val plays, which are almost all normal

## What happened

`build.exact_match_callback`, the P2 placeholder for in-training generation eval,
decoded `val_recs[:gen_eval_examples]`: the first N records of `val.jsonl` in file
order. The default config used that number (`val_exact_match`) for early stopping
and best-checkpoint selection.

## Why it matters

`val.jsonl` is ordered by game. The first 500 records are a handful of early-2023
games. About 84% of val plays are `normal`, and laterals are 0.04% (16 of 38,102),
so a 500-play head holds roughly 420 normal plays and probably zero laterals.

The R0 benchmark (`docs/benchmarks/r0-real-gt.md`) shows the regex is already at
100% on normal plays. The interesting question (claim T3) is the hard buckets.
An early-stopping signal dominated by normal plays would pick checkpoints for the
wrong reason, and its sampling noise would be large because all plays come from
a few correlated games.

The callback also had its own scoring loop (`PlayLabel.from_json` and `matches`),
separate from the harness's parser (`playparse.eval.extract.parse_prediction`),
which tolerates text around the JSON object. Training and eval could therefore
disagree about what counts as valid.

The same applied to val loss: `data.limit_val` and `train.eval_max_examples` both
take the head of the file.

## How it was found

While replacing the placeholder with the eval harness metrics in P3 integration.

## Root cause

It was a placeholder written before the P1 eval code and bucket labels were
available on the P2 branch.

## Fix

`playparse/train/val_eval.py`:

- `stratified_subset` draws a seeded, bucket-stratified subset (balanced by
  default) from the whole val file.
- `harness_val_callback` decodes with `playparse.train.generate` and scores with
  `playparse.eval.metrics.score_example`, the harness's own parser.
- It logs micro, macro, and natural-share-reweighted exact match, plus per-bucket
  exact match and counts, valid rate, and credit F1.

Any of these keys can be used as `early_stop_metric`. The same fix added
`data.val_loss_examples`, so val loss can use a proportional stratified sample
instead of the head of the file. Both subsets' keys are written to
`run_meta.json`.

## Guarding tests

`tests/test_val_eval.py`:

- the subset is seeded and fills from small buckets;
- the scoring matches a hand count;
- inside `train()`, `val_exact_match/fumble` drives early stopping and the best
  adapter is saved in PEFT format.

`tests/test_train_cli.py` checks that the per-bucket keys reach `metrics.jsonl`
and that the subset keys are recorded in `run_meta.json`.
