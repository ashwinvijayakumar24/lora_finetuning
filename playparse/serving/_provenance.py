"""Run metadata for P5b artifacts: git SHAs of all three repos, hardware, settings.

Every P5b result file carries this block so a number can be traced to the exact
code that produced it: this repository, the engine checkout actually imported,
and the serving layer (plus the engine commit it vendors).
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_dir(repo: Path) -> Path | None:
    dot = repo / ".git"
    if dot.is_dir():
        return dot
    if dot.is_file():                                   # worktree or submodule: "gitdir: <path>"
        text = dot.read_text().strip()
        if text.startswith("gitdir:"):
            return (repo / text.split(":", 1)[1].strip()).resolve()
    return None


def git_sha(repo: str | Path) -> str | None:
    """HEAD commit of ``repo``: ``git rev-parse``, else read from the .git files directly."""
    repo = Path(repo)
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        pass
    gd = _git_dir(repo)
    if gd is None or not (gd / "HEAD").is_file():
        return None
    head = (gd / "HEAD").read_text().strip()
    if not head.startswith("ref:"):
        return head
    ref = head.split(":", 1)[1].strip()
    for base in (gd, gd.parent.parent if gd.parent.name == "worktrees" else gd):
        if (base / ref).is_file():
            return (base / ref).read_text().strip()
        packed = base / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                if line.endswith(" " + ref):
                    return line.split()[0]
    return None


def git_dirty(repo: str | Path) -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return None


def run_metadata(device: str, settings: dict[str, Any] | None = None, label: str | None = None) -> dict[str, Any]:
    import torch

    from playparse.serving._serving_path import ensure_serving_importable

    repo = Path(__file__).resolve().parents[2]
    serving_root = ensure_serving_importable()
    engine_root = Path(sys.modules["engine"].__file__).resolve().parent.parent
    vendored = serving_root / "vendor" / "llm_inference_engine"
    meta: dict[str, Any] = {
        "label": label or ("CUDA run" if device.startswith("cuda") else "LOCAL / INDICATIVE"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "playparse_git_sha": git_sha(repo),
        "playparse_git_dirty": git_dirty(repo),
        "engine_dir": str(engine_root),
        "engine_git_sha": git_sha(engine_root),
        "serving_layer_dir": str(serving_root),
        "serving_layer_git_sha": git_sha(serving_root),
        "serving_layer_vendored_engine_sha": git_sha(vendored),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "settings": settings or {},
    }
    if sys.platform == "darwin":
        def _sysctl(k):
            try:
                return subprocess.check_output(["sysctl", "-n", k], text=True).strip()
            except Exception:
                return None
        meta["chip"] = _sysctl("machdep.cpu.brand_string")
        mem = _sysctl("hw.memsize")
        meta["mem_gb"] = int(mem) / 2**30 if mem else None
    if device.startswith("cuda") and torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["gpu_mem_gb"] = torch.cuda.get_device_properties(0).total_memory / 2**30
        meta["cuda"] = torch.version.cuda
    return meta
