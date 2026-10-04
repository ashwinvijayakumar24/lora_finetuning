import math

import numpy as np
import pytest

from playparse.eval._ppr import ppr_points
from playparse.eval.extract import extract_first_json_object, parse_prediction
from playparse.eval.metrics import (
    STAT_FIELDS,
    build_stat_table,
    credit_counts,
    game_fantasy_error,
    metrics_from_sums,
    point_metrics,
    score_example,
)
from playparse.ffscore.schema import Credit, PlayLabel

GOLD = PlayLabel(
    False,
    (
        Credit("J.Hurts", "pass_yds", 14),
        Credit("A.Brown", "rec", 1),
        Credit("A.Brown", "rec_yds", 14),
    ),
)


# ---------------------------------------------------------------- extraction


def test_extract_handles_fences_and_prose():
    raw = 'Sure! Here you go:\n```json\n{"nullified": false, "credits": []}\n```\nDone.'
    assert extract_first_json_object(raw) == '{"nullified": false, "credits": []}'


def test_extract_skips_unparseable_braces_and_respects_strings():
    raw = 'note {not json} then {"nullified": false, "credits": [{"player": "A}B", "stat": "rec", "value": 1}]}'
    got = extract_first_json_object(raw)
    assert got is not None and got.startswith('{"nullified"') and '"A}B"' in got


def test_extract_returns_none_on_truncated_output():
    assert extract_first_json_object('{"nullified": false, "credits": [') is None
    assert extract_first_json_object("") is None
    assert extract_first_json_object(None) is None  # type: ignore[arg-type]


def test_parse_prediction_strictness():
    lab, strict, err = parse_prediction(GOLD.to_json())
    assert lab == GOLD.canonical() and strict and err is None
    lab, strict, _ = parse_prediction("  " + GOLD.to_json() + " trailing")
    assert lab is not None and not strict
    lab, strict, err = parse_prediction('{"nullified": false, "credits": [{"player": "X", "stat": "sacks", "value": 1}]}')
    assert lab is None and not strict and "unknown stat" in err


# ---------------------------------------------------------------- per-example


def test_exact_match_order_insensitive():
    shuffled = PlayLabel(False, tuple(reversed(GOLD.credits)))
    s = score_example(GOLD, shuffled.to_json())
    assert s.valid and s.exact and (s.tp, s.fp, s.fn) == (3, 0, 0)


def test_credit_counts_partial_and_duplicates():
    pred = PlayLabel(
        False,
        (
            Credit("J.Hurts", "pass_yds", 14),
            Credit("A.Brown", "rec", 1),
            Credit("A.Brown", "rec", 1),  # duplicate: one TP, one FP
            Credit("A.Brown", "rec_yds", 13),  # wrong value: FP, and the gold one is FN
        ),
    )
    assert credit_counts(GOLD, pred) == (2, 2, 1)


def test_invalid_output_counts_as_all_missed():
    s = score_example(GOLD, "I cannot help with that")
    assert not s.valid and not s.exact and (s.tp, s.fp, s.fn) == (0, 0, 3)


def test_nullified_play():
    gold = PlayLabel(True, ())
    assert score_example(gold, '{"nullified": true, "credits": []}').exact
    s = score_example(gold, GOLD.to_json())
    assert not s.exact and s.fp == 3 and s.fn == 0


# ---------------------------------------------------------------- fantasy MAE


def test_ppr_points():
    assert ppr_points({"pass_yds": 250, "pass_td": 2, "int": 1}) == pytest.approx(16.0)
    assert ppr_points({"rec": 5, "rec_yds": 60, "rec_td": 1}) == pytest.approx(17.0)
    assert ppr_points({}) == 0.0


def test_game_fantasy_error_union_of_players():
    golds = [GOLD, PlayLabel(False, (Credit("D.Smith", "rush_yds", 10),))]
    preds = [GOLD, PlayLabel(False, (Credit("D.Smyth", "rush_yds", 10),))]  # misspelled
    err, n = game_fantasy_error(golds, preds)
    # J.Hurts 0 error, A.Brown 0 error, D.Smith -1.0, D.Smyth +1.0
    assert n == 4 and err == pytest.approx(2.0)


def test_game_fantasy_error_custom_score_fn():
    golds = [GOLD]
    preds = [None]
    err, n = game_fantasy_error(golds, preds, score_fn=lambda t: float(t.get("rec", 0)))
    assert (err, n) == (1.0, 2)


# ---------------------------------------------------------------- aggregation


def _table():
    game_ids = ["g1", "g1", "g2", "g2"]
    buckets = ["normal", "td", "normal", "normal"]
    golds = [GOLD, GOLD, PlayLabel(True, ()), GOLD]
    raws = [GOLD.to_json(), "garbage", '{"nullified": true, "credits": []}', '{"nullified": false, "credits": []}']
    scores = [score_example(g, r) for g, r in zip(golds, raws)]
    return build_stat_table(game_ids, buckets, golds, scores)


def test_stat_table_overall_and_buckets():
    t = _table()
    assert t.game_ids == ["g1", "g2"]
    m = point_metrics(t.overall)
    assert m["n"] == 4 and m["n_games"] == 2
    assert m["valid_rate"] == pytest.approx(0.75)
    assert m["exact_match"] == pytest.approx(0.5)
    # TP=3 (play 1), FP=0, FN=3 (play 2) + 3 (play 4) = 6
    assert m["credit_precision"] == pytest.approx(1.0)
    assert m["credit_recall"] == pytest.approx(3 / 9)
    assert m["credit_f1"] == pytest.approx(6 / (6 + 0 + 6))
    assert set(t.by_bucket) == {"normal", "td"}
    assert point_metrics(t.by_bucket["td"])["exact_match"] == 0.0
    assert point_metrics(t.by_bucket["normal"])["n"] == 3
    assert math.isnan(point_metrics(t.by_bucket["normal"])["fp_mae"])  # MAE is overall-only


def test_fantasy_mae_overall():
    t = _table()
    # g1: gold = 2x GOLD (Hurts 28 pass yds = 1.12; Brown 2 rec + 28 yds = 4.8);
    #     pred = 1x GOLD (Hurts 0.56; Brown 2.4) -> errors 0.56 + 2.4
    # g2: gold Brown 2.4 + Hurts 0.56; pred nothing -> errors 0.56 + 2.4
    assert point_metrics(t.overall)["fp_mae"] == pytest.approx((0.56 + 2.4) * 2 / 4)


def test_metrics_from_sums_vectorized_and_nan_safe():
    v = np.zeros((3, len(STAT_FIELDS)))
    out = metrics_from_sums(v)
    assert out["exact_match"].shape == (3,) and np.isnan(out["exact_match"]).all()
