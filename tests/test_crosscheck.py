"""Cross-check logic on tiny synthetic frames, plus the full run as a slow test."""
import pandas as pd
import pytest

from playparse import paths
from playparse.data.crosscheck import check_season, compare, gt_totals, official_totals
from playparse.data.load_pbp import ALL_SEASONS, stats_path


def _credits(rows):
    return pd.DataFrame(rows, columns=["game_id", "team", "player", "stat", "value"])


def _official(rows):
    cols = ["player_id", "player_name", "team", "game_id", "season_type", "passing_yards", "passing_tds",
            "passing_interceptions", "rushing_yards", "rushing_tds", "receptions", "receiving_yards",
            "receiving_tds", "sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost",
            "passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions",
            "fumbles_lost_total", "special_teams_tds", "fantasy_points_ppr"]
    base = {c: 0 for c in cols}
    return pd.DataFrame([{**base, **r} for r in rows])[cols]


def test_agreement_and_mae():
    gt = gt_totals(_credits([
        ("g", "PHI", "J.Hurts", "pass_yds", 250), ("g", "PHI", "A.Brown", "rec", 5),
        ("g", "PHI", "A.Brown", "rec_yds", 100), ("g", "PHI", "D.Smith", "rec_yds", 10),
    ]))
    off = official_totals(_official([
        {"player_name": "J.Hurts", "team": "PHI", "game_id": "g", "passing_yards": 250, "fantasy_points_ppr": 10.0},
        {"player_name": "A.Brown", "team": "PHI", "game_id": "g", "receptions": 5, "receiving_yards": 90,
         "fantasy_points_ppr": 14.0},
        {"player_name": "B.Graham", "team": "PHI", "game_id": "g"},  # defender, no v1 stats: ignored
    ]))
    j, s = compare(gt, off)
    assert s["player_games"] == 3 and s["gt_only"] == 1 and s["official_only"] == 0
    assert s["per_stat"]["pass_yds"]["agreement_rate"] == 1.0
    assert s["per_stat"]["rec_yds"]["agreement_rate"] == pytest.approx(0 / 2)
    # A.Brown off by 1.0 point, D.Smith by 1.0 point, Hurts exact
    assert s["ppr_mae_vs_official_v1"] == pytest.approx(2.0 / 3)
    assert s["scorer_reproduces_official"] == 1.0


def test_uncategorized_fumble_is_explained_by_total():
    gt = gt_totals(_credits([("g", "ATL", "D.Dalman", "fumble_lost", 1), ("g", "ATL", "B.R", "rush_yds", 5)]))
    off = official_totals(_official([
        {"player_name": "D.Dalman", "team": "ATL", "game_id": "g", "fumbles_lost_total": 1},
        {"player_name": "B.R", "team": "ATL", "game_id": "g", "rushing_yards": 5, "fantasy_points_ppr": 0.5},
    ]))
    _, s = compare(gt, off)
    fl = s["per_stat"]["fumble_lost"]
    assert fl["agreement_rate"] == 0.0
    assert fl["agreement_rate_incl_uncategorized_fumbles"] == 1.0


def test_official_name_spacing_is_normalized():
    off = official_totals(_official([{"player_name": "D. Thomas", "team": "DEN", "game_id": "g", "receptions": 1}]))
    assert ("g", "DEN", "D.Thomas") in off.index


@pytest.mark.slow
@pytest.mark.parametrize("season", ALL_SEASONS)
def test_full_season_crosscheck(season):
    if not stats_path(season).exists():
        pytest.skip(f"raw data not present under {paths.DATA_RAW}")
    s = check_season(season, extra_play_types=("qb_kneel", "qb_spike")).summary
    for stat in ("pass_yds", "pass_td", "int", "rush_td", "rec_td", "two_pt"):
        assert s["per_stat"][stat]["agreement_rate"] == 1.0, stat
    assert s["player_games_all_stats_agree"] > 0.99
    assert s["ppr_mae_vs_official_v1"] < 0.02
