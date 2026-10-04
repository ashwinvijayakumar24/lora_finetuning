# p1-eval: `import playparse` from a script can load the main checkout, not the worktree

## What happened

A helper script run from inside the worktree failed with `No module named
'playparse.eval.baselines'`, even though the package existed in the worktree.

## Root cause

The shared virtualenv has `playparse` installed in editable mode from the main
checkout. `python -m pytest` and `python -m playparse...` put the current directory
first on `sys.path`, so they find the worktree's copy. A script run by its file path
gets the script's own directory on `sys.path` instead. `import playparse` then
resolves to the main checkout, which did not have the new subpackage yet.

## Why it matters

The worse case is silent: the main checkout *does* have the module, and the script
runs stale code. The numbers would then come from code other than the code being
committed. The git sha in the result file would also be wrong, because
`harness.git_info` reads the repository that holds the imported package.

## Fix

Scripts under `scripts/` insert the repository root at the front of `sys.path`
(`scripts/regex_dev_estimate.py`, `scripts/bench_local_throughput.py`). The eval CLI
is meant to be run as `python -m playparse.eval.run` from the repository root.

## Guarding test

None automated, because this is a property of the environment. It is documented in
`docs/phases/P1-eval.md` under "How to run each rung".
