import json

import numpy as np

from playparse.eval.bootstrap import bootstrap_ci, paired_diff_ci, resample_weights
from playparse.eval.metrics import STAT_FIELDS, metrics_from_sums

_N = STAT_FIELDS.index("n")
_EXACT = STAT_FIELDS.index("exact")


def _correlated_plays(n_games=60, plays_per_game=40, seed=1):
    """Plays whose correctness depends heavily on the game (shared difficulty).

    Returns (per-play matrix, per-game matrix, game index per play).
    """
    rng = np.random.default_rng(seed)
    game_p = rng.choice([0.2, 0.95], size=n_games)  # some games are easy, some hard
    rows, gidx = [], []
    for g in range(n_games):
        exact = rng.random(plays_per_game) < game_p[g]
        for e in exact:
            r = np.zeros(len(STAT_FIELDS))
            r[_N] = 1
            r[_EXACT] = e
            rows.append(r)
            gidx.append(g)
    plays = np.array(rows)
    gidx = np.array(gidx)
    games = np.zeros((n_games, len(STAT_FIELDS)))
    np.add.at(games, gidx, plays)
    return plays, games


def test_weights_shape_and_deterministic():
    w1 = resample_weights(10, 50, seed=7)
    w2 = resample_weights(10, 50, seed=7)
    assert w1.shape == (50, 10)
    assert (w1.sum(axis=1) == 10).all()
    np.testing.assert_array_equal(w1, w2)
    assert not np.array_equal(w1, resample_weights(10, 50, seed=8))


def test_bootstrap_ci_deterministic_and_contains_point():
    _, games = _correlated_plays()
    a = bootstrap_ci(games, metrics_from_sums, n_boot=500, seed=3)
    b = bootstrap_ci(games, metrics_from_sums, n_boot=500, seed=3)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)  # NaN-safe equality
    em = a["exact_match"]
    assert em["lo"] <= em["point"] <= em["hi"]


def test_cluster_ci_wider_than_row_ci_on_correlated_data():
    plays, games = _correlated_plays()
    cluster = bootstrap_ci(games, metrics_from_sums, n_boot=1000, seed=0)["exact_match"]
    row = bootstrap_ci(plays, metrics_from_sums, n_boot=1000, seed=0)["exact_match"]
    assert cluster["point"] == row["point"]  # same data, same point estimate
    w_cluster = cluster["hi"] - cluster["lo"]
    w_row = row["hi"] - row["lo"]
    # Design effect here is large (intra-game correlation ~0.7, 40 plays/game).
    assert w_cluster > 2.5 * w_row


def test_undefined_metric_gives_nan_ci():
    games = np.zeros((5, len(STAT_FIELDS)))
    games[:, _N] = 1  # no credits anywhere: precision undefined
    out = bootstrap_ci(games, metrics_from_sums, n_boot=50, seed=0)
    assert np.isnan(out["credit_precision"]["lo"])
    assert out["exact_match"]["point"] == 0.0


def test_paired_diff_tighter_than_independent():
    _, games = _correlated_plays()
    better = games.copy()
    # System B gets 2 extra exact plays in every game (capped by n).
    better[:, _EXACT] = np.minimum(better[:, _EXACT] + 2, better[:, _N])
    d = paired_diff_ci(better, games, metrics_from_sums, n_boot=500, seed=0)["exact_match"]
    assert d["point"] > 0 and d["lo"] > 0  # paired CI excludes zero
