"""Locate the sibling ``llm_inference_engine`` checkout and make ``import engine`` work.

The engine is not pip-installed into this project's venv on purpose: PlayParse
extends it from the outside (PRD §12) and must never edit it. Instead the
``engine`` package is imported from the owner's checkout by file location
(:func:`register_package`), without adding the repository root to ``sys.path``.

Resolution order:

1. ``PLAYPARSE_ENGINE_DIR`` if set (Slurm jobs, unusual layouts).
2. The first ``llm_inference_engine`` directory found next to this repository or
   next to any of its parents. Walking upward matters because a git worktree
   lives at ``llm_finetuning/.claude/worktrees/<name>``, so a fixed
   ``../llm_inference_engine`` would point at a directory that does not exist.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ENGINE_DIR_ENV = "PLAYPARSE_ENGINE_DIR"
_ENGINE_REPO_NAME = "llm_inference_engine"


def engine_dir() -> Path:
    """Return the engine repository root (the directory that contains ``engine/``)."""
    override = os.environ.get(ENGINE_DIR_ENV)
    if override:
        path = Path(override).expanduser().resolve()
        if not (path / "engine" / "__init__.py").is_file():
            raise FileNotFoundError(
                f"{ENGINE_DIR_ENV}={override!r} does not contain engine/__init__.py"
            )
        return path

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / _ENGINE_REPO_NAME
        if (candidate / "engine" / "__init__.py").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find a sibling {_ENGINE_REPO_NAME}/ checkout above {here}. "
        f"Set {ENGINE_DIR_ENV} to the engine repository root."
    )


def register_package(name: str, root: Path) -> None:
    """Import ``root/name`` as top-level package ``name`` without touching ``sys.path``.

    Putting a repository root on ``sys.path`` exposes every top-level name in it.
    The engine and the serving layer both have a regular ``tests`` package, which
    then shadows this repository's ``tests`` directory (a namespace package) for
    any later ``from tests._lora_util import ...``
    (docs/issues/p5b-serving-root-shadows-tests.md).
    """
    if name in sys.modules:
        return
    pkg_dir = root / name
    spec = importlib.util.spec_from_file_location(
        name, pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise


def ensure_engine_importable() -> Path:
    """Make ``import engine`` resolve to the engine checkout (idempotent) and return its root.

    Only the ``engine`` package is registered; ``sys.path`` is not modified (see
    :func:`register_package`). If some other ``engine`` package is already
    imported (for example the copy vendored inside ``llm_serving_layer``), it is
    left alone: P5b runs inside the serving layer and must patch *that* module
    object, not a second copy.
    """
    if "engine" in sys.modules:
        mod_file = getattr(sys.modules["engine"], "__file__", None)
        return Path(mod_file).resolve().parent.parent if mod_file else engine_dir()
    root = engine_dir()
    register_package("engine", root)
    return root
