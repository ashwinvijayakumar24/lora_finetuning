"""Cluster bootstrap over games.

Plays within one game share a quarterback, a defense, weather, and a play-caller, so
their errors are correlated. Resampling individual plays would treat them as
independent and produce confidence intervals that are too narrow. Instead we resample
whole games with replacement (a "cluster bootstrap") and recompute each metric from
the resampled games' summed statistics.

Each metric is a ratio of sums (see `metrics.py`), so one resample is a weighted sum of
the per-game rows: `weights @ matrix`. All resamples at once are a single matrix
product, which keeps 1000+ resamples fast.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

MetricFn = Callable[[np.ndarray], dict[str, np.ndarray]]


def resample_weights(n_clusters: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_clusters) counts: how many times each cluster appears per resample.

    Seeded with numpy's PCG64 so the same (n_clusters, n_boot, seed) always gives the
    same resamples. Two rungs evaluated on the same games with the same seed therefore
    see identical resamples, which makes paired differences meaningful.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    w = np.zeros((n_boot, n_clusters))
    np.add.at(w, (np.repeat(np.arange(n_boot), n_clusters), idx.ravel()), 1)
    return w


def bootstrap_ci(
    matrix: np.ndarray,
    metric_fn: MetricFn,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    weights: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Percentile CIs for every metric in `metric_fn`, resampling rows of `matrix`.

    `matrix` is (n_clusters, K): one row of summed statistics per cluster (game). Pass
    a matrix with one row per play to get the naive row bootstrap instead.
    Returns {metric: {"point", "lo", "hi"}}. Resamples where a metric is undefined
    (zero denominator) are dropped from that metric's percentiles.
    """
    matrix = np.asarray(matrix, dtype=float)
    if weights is None:
        weights = resample_weights(matrix.shape[0], n_boot, seed)
    point = metric_fn(matrix.sum(axis=0))
    boot = metric_fn(weights @ matrix)
    out: dict[str, dict[str, float]] = {}
    for name, pt in point.items():
        vals = np.asarray(boot[name], dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0 or np.isnan(pt):
            lo = hi = float("nan")
        else:
            lo, hi = (float(x) for x in np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
        out[name] = {"point": float(pt), "lo": lo, "hi": hi}
    return out


def paired_diff_ci(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    metric_fn: MetricFn,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """CI on metric(A) - metric(B), with both systems scored on the same games.

    Rows of the two matrices must refer to the same games in the same order. Using the
    same resampled games for both systems cancels the shared game-difficulty noise,
    so this is tighter (and more honest) than comparing two independent CIs.
    """
    a = np.asarray(matrix_a, dtype=float)
    b = np.asarray(matrix_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap needs matrices over the same games")
    w = resample_weights(a.shape[0], n_boot, seed)
    pa, pb = metric_fn(a.sum(axis=0)), metric_fn(b.sum(axis=0))
    ba, bb = metric_fn(w @ a), metric_fn(w @ b)
    out = {}
    for name in pa:
        d = np.asarray(ba[name]) - np.asarray(bb[name])
        d = d[~np.isnan(d)]
        lo, hi = (
            (float(x) for x in np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
            if d.size
            else (float("nan"), float("nan"))
        )
        out[name] = {"point": float(pa[name] - pb[name]), "lo": lo, "hi": hi}
    return out
