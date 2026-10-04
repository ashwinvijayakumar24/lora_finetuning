"""The eval gate: decides whether a candidate adapter may replace the current one.

PRD §9.3 promotion rule, plus the checks that make the comparison meaningful:

1. ``eval_file_match`` - candidate and current were scored on the same eval file
   (sha256 equal). Numbers from different test sets cannot be compared.
2. ``valid_rate``       - the candidate's valid-output rate is at least a threshold.
   This is what catches a broken adapter (e.g. trained with the prompt mask off),
   whose outputs mostly fail to parse.
3. ``min_exact_match``  - an absolute floor on overall exact match. Set it to the R0
   (regex) score so a model that loses to the regex can never ship.
4. ``overall_exact_match`` - candidate beats or ties current on overall exact match.
5. ``bucket_regression``   - no bucket drops by more than ``bucket_tolerance_pts``
   (percentage points, absolute). A gain on the average must not hide a loss on, say,
   laterals.

Every check is evaluated and reported, even after one fails, so a rejection lists
all of its reasons at once.

Eval artifact format
--------------------
The gate reads a JSON object (or a path to one) shaped like this; rates are
fractions in [0, 1]::

    {
      "metadata": {"eval_file_sha256": "<hex>", ...},
      "overall":  {"n": 2000, "exact_match": 0.91, "valid_rate": 0.998, "credit_f1": 0.95,
                   "exact_match_ci": [0.89, 0.93]},
      "buckets":  {"lateral": {"n": 41, "exact_match": 0.56, ...}, ...}
    }

Accepted variations (normalized by :func:`load_eval_summary`): ``per_bucket`` instead
of ``buckets``; ``eval_sha256`` / ``eval_file_hash`` instead of ``eval_file_sha256``,
at top level or under ``metadata``; a metric given as ``{"value": x, "ci": [...]}``
(also ``point`` or ``mean``) instead of a bare number.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

_SHA_KEYS = ("eval_file_sha256", "eval_sha256", "eval_file_hash")
_EPS = 1e-9


class EvalArtifactError(ValueError):
    """The eval artifact is missing a field the gate needs, or is malformed."""


@dataclass(frozen=True)
class MetricSet:
    n: int | None
    exact_match: float
    valid_rate: float | None


@dataclass(frozen=True)
class EvalSummary:
    """The slice of an eval artifact the gate looks at."""

    eval_file_sha256: str | None
    overall: MetricSet
    buckets: dict[str, MetricSet]


def _metric(obj: Mapping[str, Any], key: str, where: str, required: bool) -> float | None:
    if key not in obj:
        if required:
            raise EvalArtifactError(f"{where}: missing metric {key!r}")
        return None
    v = obj[key]
    if isinstance(v, Mapping):
        for k in ("value", "point", "mean"):
            if k in v:
                v = v[k]
                break
        else:
            raise EvalArtifactError(f"{where}.{key}: dict metric needs 'value', 'point', or 'mean'")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise EvalArtifactError(f"{where}.{key}: not a number: {v!r}")
    v = float(v)
    if not 0.0 <= v <= 1.0:
        # Catches artifacts written in percent (91.0) instead of fractions (0.91);
        # comparing those against a points tolerance would be off by 100x.
        raise EvalArtifactError(f"{where}.{key}={v}: rates must be fractions in [0, 1]")
    return v


def _metric_set(obj: Any, where: str) -> MetricSet:
    if not isinstance(obj, Mapping):
        raise EvalArtifactError(f"{where}: expected an object")
    n = obj.get("n", obj.get("n_plays"))
    if n is not None and (isinstance(n, bool) or not isinstance(n, int)):
        raise EvalArtifactError(f"{where}.n: not an int: {n!r}")
    return MetricSet(
        n=n,
        exact_match=_metric(obj, "exact_match", where, required=True),
        valid_rate=_metric(obj, "valid_rate", where, required=False),
    )


def load_eval_summary(artifact: str | Path | Mapping[str, Any]) -> EvalSummary:
    """Parse an eval artifact (path or already-loaded dict) into an :class:`EvalSummary`."""
    if not isinstance(artifact, Mapping):
        artifact = json.loads(Path(artifact).read_text(encoding="utf-8"))
    meta = artifact.get("metadata") or {}
    sha = next((src[k] for src in (meta, artifact) for k in _SHA_KEYS if src.get(k)), None)
    if "overall" not in artifact:
        raise EvalArtifactError("artifact has no 'overall' section")
    buckets_raw = artifact.get("buckets", artifact.get("per_bucket", {})) or {}
    if not isinstance(buckets_raw, Mapping):
        raise EvalArtifactError("'buckets' must be an object mapping bucket name -> metrics")
    return EvalSummary(
        eval_file_sha256=sha,
        overall=_metric_set(artifact["overall"], "overall"),
        buckets={b: _metric_set(m, f"buckets.{b}") for b, m in buckets_raw.items()},
    )


@dataclass(frozen=True)
class GateConfig:
    """Thresholds. Rates are fractions; tolerances are in percentage points."""

    min_valid_rate: float = 0.95
    # Absolute floor on overall exact match. Recommended: the R0 regex score.
    min_exact_match: float = 0.0
    # Largest allowed per-bucket drop in exact match, in points (1.0 = 0.01 absolute).
    bucket_tolerance_pts: float = 1.0
    # Buckets with fewer examples than this (in either eval) are too noisy to gate on:
    # one play in a 20-play bucket is 5 points.
    min_bucket_n: int = 30
    # "skip": small buckets are reported but never block. "enforce": gate them anyway.
    small_bucket_policy: str = "skip"

    def __post_init__(self) -> None:
        if self.small_bucket_policy not in ("skip", "enforce"):
            raise ValueError("small_bucket_policy must be 'skip' or 'enforce'")


@dataclass
class CheckResult:
    name: str
    passed: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateDecision:
    promote: bool
    checks: list[CheckResult]

    @property
    def reasons(self) -> list[str]:
        """Reasons of the failed checks (empty when promoting)."""
        return [f"{c.name}: {c.reason}" for c in self.checks if not c.passed]

    def check(self, name: str) -> CheckResult:
        return next(c for c in self.checks if c.name == name)

    def summary(self) -> str:
        head = "PROMOTE" if self.promote else "REJECT"
        lines = [head] + [f"  [{'ok' if c.passed else 'FAIL'}] {c.name}: {c.reason}" for c in self.checks]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"promote": self.promote, "checks": [asdict(c) for c in self.checks]}


def _pts(x: float) -> str:
    return f"{100 * x:.2f}"


def evaluate_gate(
    candidate: EvalSummary | Mapping[str, Any] | str | Path,
    current: EvalSummary | Mapping[str, Any] | str | Path | None,
    config: GateConfig | None = None,
) -> GateDecision:
    """Run every gate check. ``current=None`` means there is no promoted version yet."""
    cfg = config or GateConfig()
    cand = candidate if isinstance(candidate, EvalSummary) else load_eval_summary(candidate)
    cur = None if current is None else (current if isinstance(current, EvalSummary) else load_eval_summary(current))
    checks: list[CheckResult] = []

    # 1. Same eval file.
    if cur is None:
        checks.append(CheckResult("eval_file_match", True, "no current version; nothing to compare against"))
    elif not cand.eval_file_sha256 or not cur.eval_file_sha256:
        missing = "candidate" if not cand.eval_file_sha256 else "current"
        checks.append(CheckResult("eval_file_match", False, f"{missing} eval artifact has no eval-file sha256"))
    elif cand.eval_file_sha256 != cur.eval_file_sha256:
        checks.append(CheckResult(
            "eval_file_match", False,
            f"evaluated on different eval files (candidate {cand.eval_file_sha256[:12]}, "
            f"current {cur.eval_file_sha256[:12]}); re-evaluate both on the same file",
            {"candidate": cand.eval_file_sha256, "current": cur.eval_file_sha256},
        ))
    else:
        checks.append(CheckResult("eval_file_match", True, f"same eval file {cand.eval_file_sha256[:12]}"))

    # 2. Valid-output rate.
    vr = cand.overall.valid_rate
    if vr is None:
        checks.append(CheckResult("valid_rate", False, "candidate artifact has no overall valid_rate"))
    else:
        ok = vr + _EPS >= cfg.min_valid_rate
        checks.append(CheckResult(
            "valid_rate", ok,
            f"{_pts(vr)}% valid outputs {'>=' if ok else '<'} required {_pts(cfg.min_valid_rate)}%",
            {"candidate": vr, "threshold": cfg.min_valid_rate},
        ))

    # 3. Absolute floor.
    em = cand.overall.exact_match
    ok = em + _EPS >= cfg.min_exact_match
    checks.append(CheckResult(
        "min_exact_match", ok,
        f"exact match {_pts(em)}% {'>=' if ok else '<'} floor {_pts(cfg.min_exact_match)}%",
        {"candidate": em, "threshold": cfg.min_exact_match},
    ))

    if cur is None:
        checks.append(CheckResult("overall_exact_match", True, "no current version; first promotion"))
        checks.append(CheckResult("bucket_regression", True, "no current version; first promotion"))
    else:
        # 4. Beat or tie overall.
        delta = em - cur.overall.exact_match
        ok = delta + _EPS >= 0
        checks.append(CheckResult(
            "overall_exact_match", ok,
            f"candidate {_pts(em)}% vs current {_pts(cur.overall.exact_match)}% "
            f"({'+' if delta >= 0 else ''}{_pts(delta)} pts)",
            {"candidate": em, "current": cur.overall.exact_match, "delta": delta},
        ))

        # 5. Per-bucket regression.
        tol = cfg.bucket_tolerance_pts / 100.0
        failed, skipped, compared = [], [], {}
        for b in sorted(cur.buckets):
            cb = cur.buckets[b]
            nb = cand.buckets.get(b)
            if nb is None:
                failed.append(f"{b}: missing from candidate eval")
                continue
            drop = cb.exact_match - nb.exact_match
            compared[b] = {"candidate": nb.exact_match, "current": cb.exact_match, "drop": drop,
                           "n_candidate": nb.n, "n_current": cb.n}
            ns = [x for x in (nb.n, cb.n) if x is not None]
            small_n = min(ns) if ns else None
            is_small = small_n is None or small_n < cfg.min_bucket_n
            if is_small and cfg.small_bucket_policy == "skip":
                note = "unknown n" if small_n is None else f"n={small_n} < {cfg.min_bucket_n}"
                skipped.append(f"{b} ({note}, drop {_pts(drop)} pts)")
                continue
            if drop > tol + _EPS:
                failed.append(f"{b}: {_pts(cb.exact_match)}% -> {_pts(nb.exact_match)}% "
                              f"(-{_pts(drop)} pts > {cfg.bucket_tolerance_pts:g} pt tolerance)")
        reason = "; ".join(failed) if failed else f"no bucket regressed more than {cfg.bucket_tolerance_pts:g} pt"
        if skipped:
            reason += f"; not gated (too few examples): {', '.join(skipped)}"
        checks.append(CheckResult("bucket_regression", not failed, reason,
                                  {"failed": failed, "skipped": skipped, "buckets": compared}))

    return GateDecision(all(c.passed for c in checks), checks)
