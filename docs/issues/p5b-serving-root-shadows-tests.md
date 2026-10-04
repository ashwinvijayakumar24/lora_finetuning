# P5b: putting the serving layer on `sys.path` would hijack `import tests`

**Status:** fixed (for the serving layer and, retroactively, for P5a's engine import); guarded by tests.

## What happened

The first version of `playparse/serving/_serving_path.py` appended the
`llm_serving_layer` repository root to `sys.path`, the same way P5a puts the
engine root there. That makes `import serving` work. It also makes the serving
layer's other top-level directories importable, and two of them collide with
this repository's:

| name | this repo | serving layer |
|---|---|---|
| `tests` | directory, no `__init__.py` (a namespace package) | regular package with `__init__.py` |
| `scripts` | directory, no `__init__.py` | regular package with `__init__.py` |

Python's import system prefers a regular package anywhere on `sys.path` over
a namespace package, even one found earlier. So once the serving root was on
the path, `import tests` returned the serving layer's test package, and
`from tests._lora_util import ...` (used by four existing PlayParse test
files) failed with `ModuleNotFoundError`. Whether it failed depended on test
collection order: if anything imported `tests` before the serving root was
added, the namespace package was already cached and everything passed.

## How it was found

Writing a shared helper for the P5b tests (`tests/_p5b_util.py`) and checking
how the existing helpers are imported. A two-line reproduction confirmed it:

```python
sys.path.insert(0, "."); sys.path.append("<llm_serving_layer>")
import tests; tests.__file__   # -> llm_serving_layer/tests/__init__.py
```

## Root cause

A repository root is not a package boundary. Adding one to `sys.path` exposes
every top-level name in it, not just the one package that was wanted.

## A second instance, already present in P5a

The same mechanism was already live through a different door. P5a's
`ensure_engine_importable()` inserted the engine repository root at
`sys.path[0]`, and the engine also has a regular `tests` package. So any test
module that imported `playparse.serving` before `tests` was first imported
broke every later `from tests._... import`:

```
pytest tests/test_lora_io.py                         # 8 passed
pytest tests/test_p5b_pool.py tests/test_lora_io.py  # ModuleNotFoundError: tests._lora_util
```

The full suite only passed because `test_lora_*` sorts before `test_p5a_*`.
It surfaced when `pytest tests/test_p5b_scheduler.py` (run alone) could not
import `tests._p5b_util`.

## Fix

Both helpers now register exactly the packages they need, by file location,
and never edit `sys.path`: `_engine_path.register_package("engine", root)` and,
in `_serving_path`, the same for `serving` and `bench`. Submodule imports such
as `serving.scheduler.scheduler` resolve through the package's own `__path__`.
None of the three packages imports anything else from its repository root
(checked with a grep over `serving/`, `bench/` and PlayParse's own code). The
P5b tests also import their helper as a top-level module (`from _p5b_util
import ...`), which pytest's default import mode provides from `tests/`.

## Guarding test

`tests/test_p5b_scheduler.py::test_serving_import_does_not_expose_other_packages`
asserts that after `ensure_serving_importable()` the serving layer's root is
not on `sys.path` and that `serving`/`bench` resolve to the serving layer.
(The engine root is not asserted absent: an environment with a legacy
editable install of the engine puts it there independently of PlayParse.) `pytest tests/test_p5b_pool.py tests/test_lora_io.py` is the
order that used to fail.
