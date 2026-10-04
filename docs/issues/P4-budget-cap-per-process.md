# P4: a budget cap kept in memory resets on every resume

**What happened.** A natural first design keeps a running spend counter inside
`label_dataset` and stops when it reaches the cap. The labeling run is meant to be
resumable, though. After a crash or a deliberate stop, rerunning the same command
would start the counter at zero, so the "hard cap" would really be a cap per process.
Three restarts would spend three times the budget.

**How it was found.** Writing the resume test next to the budget test made the
interaction obvious: the cap has to hold across runs that share a cache.

**Root cause.** Spend is a property of the cache directory (what has been bought),
not of one process.

**Fix.** Spend is recorded in `usage.json` beside the cache, written atomically after
every chunk and *before* the samples are appended. A crash between the two
over-counts spend, which is the safe direction. Every run loads the ledger and
checks the cap against the cumulative total. If an estimate was too low and a chunk
overshot the dollar cap, the report says so and later runs refuse to spend more.

**Guarding tests.** `tests/test_distill_teacher.py::test_max_calls_stops_cleanly_and_resumes`
and `::test_max_usd_with_low_estimate_overruns_by_at_most_one_chunk`.
