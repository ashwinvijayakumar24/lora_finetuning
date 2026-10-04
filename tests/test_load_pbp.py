import pandas as pd
import pytest

from playparse import paths
from playparse.data.load_pbp import ALL_SEASONS, PBP_COLUMNS, in_v1_scope, load_pbp_season, pbp_path


def test_scope_filter():
    df = pd.DataFrame(
        {
            "play_type": ["pass", "run", "no_play", "kickoff", "punt", None, "qb_kneel", "extra_point", "pass"],
            "two_point_attempt": [0, 0, 0, 0, 0, 0, 0, 0, 1],
        }
    )
    assert in_v1_scope(df).tolist() == [True, True, True, False, False, False, False, False, True]
    assert in_v1_scope(df, extra_play_types=("qb_kneel",)).tolist()[6]


def test_two_point_row_kept_even_if_play_type_unexpected():
    df = pd.DataFrame({"play_type": [None], "two_point_attempt": [1.0]})
    assert in_v1_scope(df).tolist() == [True]


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
    assert len(df) > 35_000
