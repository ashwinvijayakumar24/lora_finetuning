# The eval gate could not read the harness's eval-file hash

**Phase:** P6 · **Severity:** integration bug (fixed before any real promotion) ·
**Found by:** building the gate demo on a real R5 eval artifact, 2026-10-09.

## What happened

`load_eval_summary` returned `eval_file_sha256 = None` for every artifact the eval
harness writes, so the gate's `eval_file_match` check (both versions must be scored on
the same eval file) would have rejected every real candidate.

## Root cause

The registry/gate and the eval harness were built in parallel by different agents
against a described format. The harness writes provenance under `meta`; the gate read
`metadata`. Both had unit tests, but against their own fixtures, and no test fed a real
harness artifact to the gate.

## Fix

The gate reads `meta` (falling back to `metadata`). Guarded by
`tests/test_registry.py::test_gate_reads_the_eval_harness_meta_block`, which loads the
committed R5 artifact. Lesson: an integration test on a real artifact catches what two
sets of self-consistent unit tests cannot.
