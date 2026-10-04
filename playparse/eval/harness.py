"""The eval harness: run any predictor over eval records and write a result artifact.

A *predictor* is anything that turns a batch of eval records into raw output strings.
The harness never looks inside it. It owns everything that must be identical across
rungs: batching, timing, caching, scoring, bootstrap CIs, cost accounting, and the
provenance metadata written next to every number.

Artifacts written to `out_dir`:

- `predictions.jsonl`: one row per play (raw output, latency, usage, and the scoring).
  It doubles as a resume cache: rows written by an interrupted run with the same
  config hash are reused, so a slow rung never redoes finished plays.
- `result.json`: metrics overall and per bucket with game-level CIs, latency, cost,
  and metadata (eval-file sha256, eval-code sha256, config hash, git sha, hardware,
  date, rung).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from playparse.eval._ppr import ppr_points
from playparse.eval.bootstrap import bootstrap_ci, resample_weights
from playparse.eval.metrics import (
    ScoreFn,
    build_stat_table,
    metrics_from_sums,
    point_metrics,
    score_example,
)
from playparse.ffscore.schema import PlayLabel
from playparse.paths import REPO_ROOT

REQUIRED_KEYS = ("game_id", "play_id", "desc", "bucket", "label")

# Files whose contents define "the eval": change any of them and old numbers are no
# longer comparable, so their combined hash goes in every result file.
EVAL_CODE_FILES = (
    "playparse/eval/extract.py",
    "playparse/eval/metrics.py",
    "playparse/eval/bootstrap.py",
    "playparse/eval/harness.py",
    "playparse/eval/_ppr.py",
    "playparse/ffscore/schema.py",
)


# --------------------------------------------------------------------------- types


@dataclass
class Prediction:
    """One raw model output plus optional usage.

    `usage` holds token counts (`input_tokens`, `output_tokens`, ...) and, when the
    predictor can price itself (the API rung), `cost_usd`. Predictors may return plain
    strings instead when they have nothing to report.
    """

    text: str
    usage: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class Predictor(Protocol):
    """Maps a batch of eval records to one raw output string (or Prediction) each."""

    name: str
    batch_size: int

    def config(self) -> dict[str, Any]:
        """Everything that changes the outputs (model, prompt, decoding). Hashed."""
        ...

    def predict_batch(self, records: Sequence[dict]) -> list[str | Prediction]: ...


# --------------------------------------------------------------------------- io


def load_records(path: str | Path) -> list[dict]:
    """Read an eval JSONL file and check every record has the required keys."""
    records = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            missing = [k for k in REQUIRED_KEYS if k not in rec]
            if missing:
                raise ValueError(f"{path}:{lineno}: record missing keys {missing}")
            records.append(rec)
    return records


def record_key(rec: dict) -> str:
    return f"{rec['game_id']}#{rec['play_id']}"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def records_sha256(records: Sequence[dict]) -> str:
    """Hash of the records themselves, used when there is no eval file on disk."""
    blob = "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in records)
    return _sha256_bytes(blob.encode())


def eval_code_sha256() -> str:
    h = hashlib.sha256()
    for rel in EVAL_CODE_FILES:
        p = REPO_ROOT / rel
        h.update(rel.encode())
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()


def config_hash(config: dict) -> str:
    return _sha256_bytes(json.dumps(config, sort_keys=True, default=str).encode())[:16]


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=True
            ).stdout.strip()
        except Exception:
            return None

    sha = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"git_sha": sha, "git_dirty": bool(status) if status is not None else None}


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if sys.platform == "darwin":
        try:
            info["cpu"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
        except Exception:
            pass
    try:
        info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        pass
    if "torch" in sys.modules:  # only report accelerators if a rung actually loaded torch
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["accelerator"] = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            info["accelerator"] = "mps"
    return info


# --------------------------------------------------------------------------- run


def _normalize(out: str | Prediction) -> Prediction:
    if isinstance(out, Prediction):
        return out
    if isinstance(out, str):
        return Prediction(out)
    raise TypeError(f"predictor returned {type(out).__name__}, expected str or Prediction")


def _load_cache(path: Path, chash: str) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not path.exists():
        return cache
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line from an interrupted write
            if row.get("config_hash") != chash:
                raise ValueError(
                    f"{path} holds predictions from config {row.get('config_hash')}, not {chash}. "
                    "Use a fresh --out directory or resume=False."
                )
            cache[row["key"]] = row
    return cache


def _ci_block(matrix: np.ndarray, weights: np.ndarray, alpha: float) -> dict[str, Any]:
    pm = point_metrics(matrix)
    cis = bootstrap_ci(matrix, metrics_from_sums, weights=weights, alpha=alpha)
    return {
        "n": pm["n"],
        "n_games": pm["n_games"],
        **{name: ci for name, ci in cis.items()},
    }


def run_eval(
    predictor: Predictor,
    records: Sequence[dict],
    *,
    rung: str,
    out_dir: str | Path | None = None,
    eval_path: str | Path | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    score_fn: ScoreFn = ppr_points,
    gpu_usd_per_hour: float | None = None,
    resume: bool = True,
    extra_meta: dict[str, Any] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Run `predictor` over `records`, score, and (if `out_dir`) write artifacts.

    Latency: a play's latency is the wall time of the batch it was in, because that is
    how long a request in that batch would wait. Throughput (plays/sec) is reported
    separately. Cost: the sum of predictor-reported `usage["cost_usd"]` if any (API
    rungs); otherwise prediction time × `gpu_usd_per_hour` if given (local rungs).
    """
    if not records:
        raise ValueError("no records to evaluate")
    keys = [record_key(r) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate (game_id, play_id) in records")

    cfg = {"rung": rung, "predictor": predictor.name, **predictor.config()}
    chash = config_hash(cfg)

    out_path = Path(out_dir) if out_dir is not None else None
    pred_path = out_path / "predictions.jsonl" if out_path else None
    cache: dict[str, dict] = {}
    if pred_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
        if resume:
            cache = _load_cache(pred_path, chash)
        elif pred_path.exists():
            pred_path.unlink()

    todo = [r for r in records if record_key(r) not in cache]
    n_resumed = len(records) - len(todo)
    bs = max(1, int(getattr(predictor, "batch_size", 1)))
    sink = open(pred_path, "a", encoding="utf-8") if pred_path is not None else None
    try:
        done = n_resumed
        for start in range(0, len(todo), bs):
            batch = todo[start : start + bs]
            t0 = time.perf_counter()
            outs = predictor.predict_batch(batch)
            dt = time.perf_counter() - t0
            if len(outs) != len(batch):
                raise RuntimeError(f"predictor returned {len(outs)} outputs for {len(batch)} records")
            for rec, out in zip(batch, outs):
                p = _normalize(out)
                row = {
                    "key": record_key(rec),
                    "config_hash": chash,
                    "raw": p.text,
                    "usage": p.usage,
                    "latency_s": dt,
                    "batch_size": len(batch),
                }
                cache[row["key"]] = row
                if sink is not None:
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            if sink is not None:
                sink.flush()
            done += len(batch)
            if progress is not None:
                progress(done, len(records))
    finally:
        if sink is not None:
            sink.close()

    # ---- scoring
    golds = [PlayLabel.from_json(r["label"]) for r in records]
    rows = [cache[k] for k in keys]
    scores = [score_example(g, row["raw"]) for g, row in zip(golds, rows)]
    table = build_stat_table(
        [str(r["game_id"]) for r in records], [r["bucket"] for r in records], golds, scores, score_fn
    )
    weights = resample_weights(len(table.game_ids), n_boot, seed)
    overall = _ci_block(table.overall, weights, alpha)
    buckets = {b: _ci_block(m, weights, alpha) for b, m in table.by_bucket.items()}

    # ---- latency, throughput, cost
    lat = np.array([row["latency_s"] for row in rows])
    predict_seconds = float(sum(row["latency_s"] / row["batch_size"] for row in rows))
    usage_tot: dict[str, float] = {}
    for row in rows:
        for k, v in (row.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                usage_tot[k] = usage_tot.get(k, 0.0) + v
    if "cost_usd" in usage_tot:
        cost, cost_basis = usage_tot["cost_usd"], "api usage"
    elif gpu_usd_per_hour is not None:
        cost, cost_basis = predict_seconds * gpu_usd_per_hour / 3600, f"compute time @ ${gpu_usd_per_hour}/h"
    else:
        cost, cost_basis = None, "not priced"
    n = len(records)

    eval_sha = file_sha256(eval_path) if eval_path is not None else records_sha256(records)
    result = {
        "meta": {
            "rung": rung,
            "predictor": predictor.name,
            "config": cfg,
            "config_hash": chash,
            "eval_file": str(eval_path) if eval_path is not None else None,
            "eval_sha256": eval_sha,
            "eval_code_sha256": eval_code_sha256(),
            **git_info(),
            "hardware": hardware_info(),
            "date": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "n_boot": n_boot,
            "seed": seed,
            "alpha": alpha,
            "ci_method": "percentile cluster bootstrap over games",
            "score_fn": getattr(score_fn, "__qualname__", repr(score_fn)),
            "n_resumed": n_resumed,
            **(extra_meta or {}),
        },
        "overall": overall,
        "buckets": buckets,
        "latency": {
            "p50_s": float(np.percentile(lat, 50)),
            "p99_s": float(np.percentile(lat, 99)),
            "mean_s": float(lat.mean()),
            "batch_size": bs,
            "predict_seconds": predict_seconds,
            "plays_per_sec": n / predict_seconds if predict_seconds > 0 else None,
            "note": "per-play latency = wall time of its batch",
        },
        "cost": {
            "basis": cost_basis,
            "total_usd": cost,
            "usd_per_1k_plays": 1000 * cost / n if cost is not None else None,
            "usage_totals": usage_tot,
        },
    }

    if out_path is not None:
        # Rewrite predictions with scoring attached (resume ignores the extra fields).
        tmp = pred_path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for rec, row, s in zip(records, rows, scores):
                f.write(
                    json.dumps(
                        {
                            **row,
                            "game_id": rec["game_id"],
                            "play_id": rec["play_id"],
                            "bucket": rec["bucket"],
                            "gold": rec["label"],
                            "pred": s.pred.to_json() if s.pred is not None else None,
                            "valid": s.valid,
                            "strict_valid": s.strict_valid,
                            "exact": s.exact,
                            "tp": s.tp,
                            "fp": s.fp,
                            "fn": s.fn,
                            "error": s.error,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        tmp.replace(pred_path)
        with open(out_path / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=_json_default)
            f.write("\n")
    return result


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def format_summary(result: dict[str, Any]) -> str:
    """A short human-readable table of a result, for the CLI."""

    def pct(ci: dict) -> str:
        if ci["point"] != ci["point"]:  # NaN
            return "     -          "
        return f"{100 * ci['point']:5.1f} [{100 * ci['lo']:4.1f},{100 * ci['hi']:5.1f}]"

    lines = [
        f"rung={result['meta']['rung']}  n={result['overall']['n']}  games={result['overall']['n_games']}",
        f"{'bucket':<18} {'n':>6}  {'valid%':<18} {'exact%':<18} {'creditF1%':<18}",
    ]
    for name, block in [("OVERALL", result["overall"]), *result["buckets"].items()]:
        lines.append(
            f"{name:<18} {block['n']:>6}  {pct(block['valid_rate']):<18} "
            f"{pct(block['exact_match']):<18} {pct(block['credit_f1']):<18}"
        )
    mae = result["overall"]["fp_mae"]
    lines.append(f"game fantasy-point MAE (PPR): {mae['point']:.3f} [{mae['lo']:.3f}, {mae['hi']:.3f}]")
    lt = result["latency"]
    lines.append(
        f"latency p50={lt['p50_s'] * 1000:.1f}ms p99={lt['p99_s'] * 1000:.1f}ms  "
        f"throughput={lt['plays_per_sec'] or 0:.1f} plays/s"
    )
    c = result["cost"]
    if c["usd_per_1k_plays"] is not None:
        lines.append(f"cost: ${c['usd_per_1k_plays']:.4f} per 1k plays ({c['basis']})")
    return "\n".join(lines)
