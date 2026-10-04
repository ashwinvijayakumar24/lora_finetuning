"""Eval metrics: per-example scoring, then aggregation overall and per bucket.

Every headline metric is a ratio of sums (for example exact match = exact plays / all
plays, credit F1 = 2·TP / (2·TP + FP + FN)). So each game is reduced to one vector of
sums ("sufficient statistics"), and any metric can be recomputed from a sum of game
vectors. That is what makes the game-level (cluster) bootstrap cheap: resampling games
is just re-weighting rows of a small matrix. See `bootstrap.py`.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from playparse.eval._ppr import ppr_points
from playparse.eval.extract import parse_prediction
from playparse.ffscore.schema import PlayLabel

ScoreFn = Callable[[Mapping[str, int]], float]

# Columns of the per-game sufficient-statistics matrix.
STAT_FIELDS: tuple[str, ...] = (
    "n",  # plays
    "valid",  # output parsed and passed the schema (after first-JSON-object extraction)
    "strict_valid",  # output was exactly the JSON object, nothing around it
    "exact",  # play exact match
    "tp",  # credits in both prediction and gold (multiset intersection)
    "fp",  # predicted credits not in gold
    "fn",  # gold credits not predicted
    "abs_err",  # sum over (game, player) of |pred points - gold points|
    "n_pairs",  # number of (game, player) pairs in that sum
)
_IDX = {f: i for i, f in enumerate(STAT_FIELDS)}

METRIC_NAMES: tuple[str, ...] = (
    "valid_rate",
    "strict_valid_rate",
    "exact_match",
    "credit_precision",
    "credit_recall",
    "credit_f1",
    "fp_mae",
)


@dataclass(frozen=True)
class ExampleScore:
    valid: bool
    strict_valid: bool
    exact: bool
    tp: int
    fp: int
    fn: int
    pred: PlayLabel | None
    error: str | None


def credit_counts(gold: PlayLabel, pred: PlayLabel | None) -> tuple[int, int, int]:
    """(TP, FP, FN) over credits as multisets. An invalid prediction has no credits."""
    g = gold.credit_multiset()
    p = pred.credit_multiset() if pred is not None else {}
    tp = sum(min(cnt, p.get(c, 0)) for c, cnt in g.items())
    return tp, sum(p.values()) - tp, sum(g.values()) - tp


def score_example(gold: PlayLabel, raw: str) -> ExampleScore:
    pred, strict, err = parse_prediction(raw)
    tp, fp, fn = credit_counts(gold, pred)
    return ExampleScore(
        valid=pred is not None,
        strict_valid=strict,
        exact=pred is not None and pred.matches(gold),
        tp=tp,
        fp=fp,
        fn=fn,
        pred=pred,
        error=err,
    )


def player_totals(labels: Iterable[PlayLabel | None]) -> dict[str, dict[str, int]]:
    """Sum credits per player across plays (None = an invalid prediction, no credits)."""
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for lab in labels:
        if lab is None:
            continue
        for c in lab.credits:
            out[c.player][c.stat] += c.value
    return out


def game_fantasy_error(
    golds: Sequence[PlayLabel], preds: Sequence[PlayLabel | None], score_fn: ScoreFn = ppr_points
) -> tuple[float, int]:
    """(sum of |error|, number of players) for one game.

    Players are the union of those credited in gold and in the prediction, so a
    hallucinated player counts as error too. Players are keyed by their desc-style
    name; two same-named players in one game would merge, which affects gold and
    prediction identically.
    """
    gt = player_totals(golds)
    pt = player_totals(preds)
    players = set(gt) | set(pt)
    err = sum(abs(score_fn(pt.get(p, {})) - score_fn(gt.get(p, {}))) for p in players)
    return float(err), len(players)


@dataclass
class StatTable:
    """Per-game sufficient statistics, overall and per bucket (same game order)."""

    game_ids: list[str]
    overall: np.ndarray  # (G, K)
    by_bucket: dict[str, np.ndarray]  # bucket -> (G, K); fantasy columns are zero


def build_stat_table(
    game_ids: Sequence[str],
    buckets: Sequence[str],
    golds: Sequence[PlayLabel],
    scores: Sequence[ExampleScore],
    score_fn: ScoreFn = ppr_points,
) -> StatTable:
    games = sorted(set(game_ids))
    gidx = {g: i for i, g in enumerate(games)}
    K = len(STAT_FIELDS)
    overall = np.zeros((len(games), K))
    by_bucket: dict[str, np.ndarray] = {}
    per_game_gold: dict[str, list] = defaultdict(list)
    per_game_pred: dict[str, list] = defaultdict(list)
    for gid, bucket, gold, s in zip(game_ids, buckets, golds, scores, strict=True):
        row = np.zeros(K)
        row[_IDX["n"]] = 1
        row[_IDX["valid"]] = s.valid
        row[_IDX["strict_valid"]] = s.strict_valid
        row[_IDX["exact"]] = s.exact
        row[_IDX["tp"]] = s.tp
        row[_IDX["fp"]] = s.fp
        row[_IDX["fn"]] = s.fn
        overall[gidx[gid]] += row
        if bucket not in by_bucket:
            by_bucket[bucket] = np.zeros((len(games), K))
        by_bucket[bucket][gidx[gid]] += row
        per_game_gold[gid].append(gold)
        per_game_pred[gid].append(s.pred)
    # Fantasy MAE is a game-level quantity: it is not decomposable by bucket.
    for gid in games:
        err, n = game_fantasy_error(per_game_gold[gid], per_game_pred[gid], score_fn)
        overall[gidx[gid], _IDX["abs_err"]] = err
        overall[gidx[gid], _IDX["n_pairs"]] = n
    return StatTable(games, overall, dict(sorted(by_bucket.items())))


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def metrics_from_sums(v: np.ndarray) -> dict[str, np.ndarray]:
    """Metrics from summed statistics. `v` has shape (..., K); outputs have shape (...)."""
    v = np.asarray(v, dtype=float)
    col = {f: v[..., i] for f, i in _IDX.items()}
    tp, fp, fn = col["tp"], col["fp"], col["fn"]
    return {
        "valid_rate": _ratio(col["valid"], col["n"]),
        "strict_valid_rate": _ratio(col["strict_valid"], col["n"]),
        "exact_match": _ratio(col["exact"], col["n"]),
        "credit_precision": _ratio(tp, tp + fp),
        "credit_recall": _ratio(tp, tp + fn),
        "credit_f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "fp_mae": _ratio(col["abs_err"], col["n_pairs"]),
    }


def point_metrics(matrix: np.ndarray) -> dict[str, float]:
    """Metrics for one stat matrix (G, K), plus raw counts."""
    totals = matrix.sum(axis=0)
    out = {k: float(x) for k, x in metrics_from_sums(totals).items()}
    out["n"] = int(totals[_IDX["n"]])
    out["n_games"] = int((matrix[:, _IDX["n"]] > 0).sum())
    return out
