import pandas as pd
import pytest

from playparse import paths
from playparse.data.load_pbp import ALL_SEASONS, PBP_COLUMNS, in_v1_scope, is_timeout_row, load_pbp_season, pbp_path


def test_scope_filter():
    df = pd.DataFrame(
        {
            "play_type": ["pass", "run", "no_play", "kickoff", "punt", None, "qb_kneel", "extra_point", "pass"],
            "two_point_attempt": [0, 0, 0, 0, 0, 0, 0, 0, 1],
            "desc": ["x", "x", "PENALTY on X, False Start, 5 yards - No Play.", "x", "x", "END QUARTER 1", "x",
                     "x", "TWO-POINT CONVERSION ATTEMPT."],
        }
    )
    assert in_v1_scope(df).tolist() == [True, True, True, False, False, False, False, False, True]
    assert in_v1_scope(df, extra_play_types=("qb_kneel",)).tolist()[6]


def test_two_point_row_kept_even_if_play_type_unexpected():
    df = pd.DataFrame({"play_type": [None], "two_point_attempt": [1.0], "desc": ["TWO-POINT CONVERSION ATTEMPT."]})
    assert in_v1_scope(df).tolist() == [True]


def test_timeouts_filed_as_no_play_are_dropped():
    df = pd.DataFrame(
        {
            "play_type": ["no_play", "no_play", "no_play", "no_play"],
            "two_point_attempt": [0, 0, 0, 0],
            "desc": [
                "Timeout #2 by BAL at 00:31.",
                "(10:08) PENALTY on DET-26-J.Gibbs, False Start, 5 yards, enforced at DET 46 - No Play.",
                "(8:01) ... Penalty on MIA-9-J.Smith, Offensive Offside, offsetting - No Play.",
                "Timeout #3 by TEN at 00:22. penalty was charged due to an injury on the previous play.",
            ],
        }
    )
    assert is_timeout_row(df).tolist() == [True, False, False, True]
    assert in_v1_scope(df).tolist() == [False, True, True, False]


@pytest.mark.slow
@pytest.mark.parametrize("season", ALL_SEASONS)
def test_real_season_loads(season):
    if not pbp_path(season).exists():
        pytest.skip(f"raw data not present under {paths.DATA_RAW}")
    df = load_pbp_season(season)
    assert set(df.columns) == set(PBP_COLUMNS)
    assert set(df.play_type.unique()) <= {"pass", "run", "no_play"}
    assert set(df.season_type.unique()) <= {"REG", "POST"}
    assert not df.duplicated(["game_id", "play_id"]).any()
    assert not df.desc.str.match(r"^\s*Timeout").any()
    assert len(df) > 30_000
