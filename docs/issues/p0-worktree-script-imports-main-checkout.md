# P0: scripts run from a git worktree imported the main checkout's code

## What happened

Running `python scripts/p0_param_table.py` from a git worktree failed with
`ImportError: cannot import name 'LoRAConfig' from 'playparse.lora'`. The path in
the traceback was the *main* checkout's `playparse/lora/__init__.py`, which does
not have the P0 code yet.

## How it was found

By running the script without `PYTHONPATH=.`. Earlier runs had used
`PYTHONPATH=.`, which hid the problem.

## Root cause

The shared virtual environment has an editable install of `playparse` (a `.pth`
finder that points at the main checkout). When Python runs `scripts/foo.py`, it
puts `scripts/` on `sys.path`, not the repository root. So `import playparse`
falls through to the editable install, which resolves to the main checkout.

pytest does not have this problem, because `pyproject.toml` sets
`pythonpath = ["."]` (commit 9f99b03). Scripts need the same treatment.

A related trap: `playparse.paths.WEIGHTS` defaults to
`REPO_ROOT.parent / "llm_inference_engine" / "weights"`. Inside a worktree,
`REPO_ROOT.parent` is `.claude/worktrees/`, so the default path does not exist.
Set `PLAYPARSE_WEIGHTS` explicitly when working in a worktree. Tests that need the
real config skip, rather than fail, when it is missing.

## Why it matters

The failure here was loud. The dangerous version is quiet: if the main checkout
had an older but importable `playparse.lora`, the script would have run the old
code and written results attributed to the worktree's git SHA.

## Fix

`scripts/p0_param_table.py` inserts its own repository root at the front of
`sys.path` before importing `playparse`. Future scripts should do the same.

## Guarding test

`tests/test_lora_memory.py::test_script_imports_its_own_checkout` launches a
fresh interpreter from `scripts/` without `PYTHONPATH`, runs the script's module
body, and asserts that the `playparse` it imported lives under the checkout the
test is running from. With the `sys.path` line removed, this test fails in a
worktree (verified). In the main checkout it passes either way, so it only guards
worktree runs.
