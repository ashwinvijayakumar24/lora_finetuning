"""Filesystem locations, overridable by environment variables.

Raw data and model weights live outside git. Defaults resolve relative to this
repository so a fresh clone works after `scripts/fetch_data.sh`; the environment
variables let git worktrees and Slurm jobs point at a shared copy.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_path(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value) if value else default


DATA_RAW = _env_path("PLAYPARSE_DATA_DIR", REPO_ROOT / "data" / "raw")
DATA_PROCESSED = _env_path("PLAYPARSE_PROCESSED_DIR", REPO_ROOT / "data" / "processed")
WEIGHTS = _env_path("PLAYPARSE_WEIGHTS", REPO_ROOT.parent / "llm_inference_engine" / "weights")
RESULTS = REPO_ROOT / "results"
