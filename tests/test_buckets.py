import json
from pathlib import Path

import pytest

from playparse.data.buckets import BUCKETS, PRECEDENCE, assign_bucket, bucket_flags

ROWS = {c["case"]: c["row"] for c in json.loads((Path(__file__).parent / "fixtures" / "gt_cases.json").read_text())}


def test_precedence_is_the_documented_order():
    # rarest first, measured on 2015-2022 (docs/phases/P1-data.md); normal last
    assert PRECEDENCE == (
        "lateral", "two_point", "challenge", "interception", "fumble",
        "penalty_stands", "td", "penalty_nullified", "normal",
    )
    assert set(PRECEDENCE) == BUCKETS and len(BUCKETS) == 9


def base(**kw):
    row = {"desc": "x", "play_type": "pass"}
    row.update(kw)
    return row


@pytest.mark.parametrize(
    "row, expected",
    [
        (base(), "normal"),
        (base(touchdown=1), "td"),
        (base(fumble=1, touchdown=1), "fumble"),
        (base(penalty=1, fumble=1), "fumble"),
        (base(penalty=1, touchdown=1), "penalty_stands"),
        (base(desc="... for 41 yards. Penalty on TEN-25-X, Defensive Holding, declined."), "penalty_stands"),
        (base(penalty=1, play_type="no_play"), "penalty_nullified"),
        (base(interception=1, penalty=1), "interception"),
        (base(replay_or_challenge=1, interception=1), "challenge"),
        (base(two_point_attempt=1, two_point_conv_result="success", replay_or_challenge=1), "two_point"),
        (base(lateral_reception=1, touchdown=1, replay_or_challenge=1), "lateral"),
        (base(desc="... for 3 yards. Lateral to 1-J.Downs for 13 yards"), "lateral"),
        (base(desc="TWO-POINT CONVERSION ATTEMPT. ... - No Play.", play_type="no_play", penalty=1), "two_point"),
    ],
)
def test_precedence_on_combinations(row, expected):
    assert assign_bucket(row) == expected


def test_exactly_one_bucket_and_normal_means_nothing_else():
    for case, row in ROWS.items():
        flags = bucket_flags(row)
        b = assign_bucket(row)
        assert b in BUCKETS
        assert flags[b]
        if b == "normal":
            assert sum(flags.values()) == 1, case


@pytest.mark.parametrize(
    "case, expected",
    [
        ("completion", "normal"),
        ("pass_td", "td"),
        ("lateral_pass_td", "lateral"),
        ("multi_lateral", "lateral"),
        ("penalty_nullified", "penalty_nullified"),
        ("presnap_penalty", "penalty_nullified"),
        ("penalty_stands_spot_foul", "penalty_stands"),
        ("penalty_stands_declined", "penalty_stands"),
        ("replay_reversed_td_to_incomplete", "challenge"),
        ("interception", "interception"),
        ("interception_return_td", "interception"),
        ("sack_fumble_lost", "fumble"),
        ("two_point_success_deadball_penalty", "two_point"),
        ("two_point_nullified", "two_point"),
    ],
)
def test_real_rows(case, expected):
    assert assign_bucket(ROWS[case]) == expected
