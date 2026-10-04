import pytest

from playparse.ffscore.schema import STAT_VOCAB, Credit, PlayLabel
from playparse.ffscore.scorer import (
    CONFIGS,
    get_config,
    round_points,
    score_game,
    score_games,
    score_label,
    stat_totals,
)


def L(*credits, nullified=False):
    return PlayLabel(nullified, tuple(Credit(*c) for c in credits))


TD_PASS = L(("J.Hurts", "pass_yds", 14), ("J.Hurts", "pass_td", 1), ("A.Brown", "rec", 1),
            ("A.Brown", "rec_yds", 14), ("A.Brown", "rec_td", 1))


def test_configs_cover_vocab():
    assert set(CONFIGS) == {"standard", "half_ppr", "ppr", "espn_14team_ppr"}
    for cfg in CONFIGS.values():
        assert set(cfg) == set(STAT_VOCAB)


def test_reception_value_differs_by_config():
    assert score_label(TD_PASS, "standard")["A.Brown"] == pytest.approx(7.4)
    assert score_label(TD_PASS, "half_ppr")["A.Brown"] == pytest.approx(7.9)
    assert score_label(TD_PASS, "ppr")["A.Brown"] == pytest.approx(8.4)
    assert score_label(TD_PASS, "espn_14team_ppr") == score_label(TD_PASS, "ppr")
    assert score_label(TD_PASS, "ppr")["J.Hurts"] == pytest.approx(0.56 + 4)


def test_negative_stats():
    lab = L(("P.Mahomes", "int", 1), ("T.Kelce", "fumble_lost", 1), ("T.Kelce", "rec", 1), ("T.Kelce", "rec_yds", -3))
    pts = score_label(lab, "ppr")
    assert pts["P.Mahomes"] == pytest.approx(-2)
    assert pts["T.Kelce"] == pytest.approx(-2 + 1 - 0.3)


def test_two_point_scores_two_for_each_player():
    lab = L(("C.Williams", "two_pt", 1), ("D.Swift", "two_pt", 1))
    assert score_label(lab, "standard") == {"C.Williams": 2.0, "D.Swift": 2.0}


def test_nullified_scores_nothing():
    assert score_label(L(nullified=True), "ppr") == {}
    assert score_game([L(nullified=True), TD_PASS], "ppr") == score_label(TD_PASS, "ppr")


def test_game_sums_yards_before_scoring():
    plays = [L(("QB", "pass_yds", 7))] * 3
    assert stat_totals(plays) == {"QB": {"pass_yds": 21}}
    assert round_points(score_game(plays, "ppr")["QB"]) == 0.84


def test_score_games_groups_by_game():
    out = score_games([("g1", TD_PASS), ("g2", TD_PASS), ("g1", L(("A.Brown", "rush_yds", 10)))], "ppr")
    assert out[("g1", "A.Brown")] == pytest.approx(9.4)
    assert out[("g2", "A.Brown")] == pytest.approx(8.4)


def test_ppr_matches_nflverse_formula():
    # nflverse fantasy_points_ppr for v1 stats: pass_yds/25 + 4 TD - 2 INT + (rush+rec yds)/10
    # + 6 (rush+rec TD) + 2 (2pt) - 2 (fumbles lost) + receptions
    totals = {"X": {"pass_yds": 287, "pass_td": 2, "int": 1, "rush_yds": 31, "rush_td": 1,
                    "rec": 0, "two_pt": 1, "fumble_lost": 1}}
    from playparse.ffscore.scorer import score_totals
    expected = 287 / 25 + 8 - 2 + 3.1 + 6 + 2 - 2
    assert score_totals(totals, "ppr")["X"] == pytest.approx(expected)


def test_custom_config_validated():
    with pytest.raises(ValueError):
        get_config({"pass_yds": 0.04})
    with pytest.raises(KeyError):
        get_config("yahoo")
    custom = dict(CONFIGS["ppr"], pass_td=6.0)
    assert score_label(TD_PASS, custom)["J.Hurts"] == pytest.approx(6.56)
