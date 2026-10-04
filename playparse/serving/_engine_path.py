"""Locate the sibling ``llm_inference_engine`` checkout and make ``import engine`` work.

The engine is not pip-installed into this project's venv on purpose: PlayParse
extends it from the outside (PRD §12) and must never edit it. Instead the
engine's repository root is put on ``sys.path`` so that ``import engine``
resolves to the owner's checkout.

Resolution order:

1. ``PLAYPARSE_ENGINE_DIR`` if set (Slurm jobs, unusual layouts).
2. The first ``llm_inference_engine`` directory found next to this repository or
   next to any of its parents. Walking upward matters because a git worktree
   lives at ``llm_finetuning/.claude/worktrees/<name>``, so a fixed
   ``../llm_inference_engine`` would point at a directory that does not exist.
"""
from __future__ import annotations

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


def ensure_engine_importable() -> Path:
    """Put the engine root on ``sys.path`` (idempotent) and return it.

    If some other ``engine`` package is already imported (for example the copy
    vendored inside ``llm_serving_layer``), it is left alone: P5b runs inside the
    serving layer and must patch *that* module object, not a second copy.
    """
    if "engine" in sys.modules:
        mod_file = getattr(sys.modules["engine"], "__file__", None)
        return Path(mod_file).resolve().parent.parent if mod_file else engine_dir()
    root = engine_dir()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
