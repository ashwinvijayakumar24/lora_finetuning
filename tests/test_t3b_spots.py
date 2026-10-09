"""Field-spot arithmetic (playparse.ffscore.spots), the part of T3b that replaces the
model's subtraction. Every case is worked by hand in the comment next to it."""
from __future__ import annotations

import pytest

from playparse.ffscore.spots import (
    MIDFIELD,
    SpotError,
    advance,
    canonical_spot,
    parse_spot,
    spot_to_yardline100,
    yardline100_to_spot,
    yards_between,
)


@pytest.mark.parametrize("spot,posteam,expected", [
    ("PHI 30", "PHI", 70),   # own 30: 70 yards to go
    ("DAL 22", "PHI", 22),   # opponent's 22
    ("50", "PHI", 50),       # midfield, either way
    ("MID 50", "PHI", 50),   # nflverse yrdln spelling
    ("PIT 50", "PHI", 50),   # 2015-16 desc spelling
    ("PHI 1", "PHI", 99),    # backed up at the own 1
    ("DAL 1", "PHI", 1),     # first and goal at the 1
    ("DAL 0", "PHI", 0),     # the opponent's goal line: a touchdown
    ("PHI 0", "PHI", 100),   # the offense's own goal line: a safety
    ("PHI 49", "PHI", 51),
    ("DAL 49", "PHI", 49),
])
def test_spot_to_yardline100(spot, posteam, expected):
    assert spot_to_yardline100(spot, posteam) == expected


@pytest.mark.parametrize("y,expected", [
    (70, "PHI 30"), (22, "DAL 22"), (50, "50"), (99, "PHI 1"), (1, "DAL 1"),
    (0, "DAL 0"), (100, "PHI 0"), (51, "PHI 49"), (49, "DAL 49"),
])
def test_yardline100_to_spot_is_the_inverse(y, expected):
    assert yardline100_to_spot(y, "PHI", "DAL") == expected
    assert spot_to_yardline100(expected, "PHI") == y


def test_inverse_round_trips_every_yard():
    for y in range(101):
        assert spot_to_yardline100(yardline100_to_spot(y, "KC", "LV"), "KC") == y


@pytest.mark.parametrize("frm,to,expected", [
    ("PHI 30", "PHI 45", 15),   # gain inside own half
    ("PHI 30", "DAL 22", 48),   # across midfield: 20 to the 50, then 28 more
    ("PHI 30", "50", 20),       # to midfield
    ("50", "DAL 38", 12),       # from midfield
    ("DAL 22", "DAL 30", -8),   # loss on the opponent's half
    ("DAL 46", "PHI 47", -7),   # loss back across midfield: 4 yards to the 50, 3 more
    ("DAL 19", "DAL 0", 19),    # touchdown run
    ("PHI 2", "PHI 0", -2),     # tackled in the end zone (safety)
    ("PHI 30", "PHI 30", 0),
])
def test_yards_between(frm, to, expected):
    assert yards_between(frm, to, "PHI") == expected


def test_yards_between_is_antisymmetric():
    assert yards_between("PHI 30", "DAL 22", "PHI") == -yards_between("DAL 22", "PHI 30", "PHI")


def test_any_other_team_is_the_opponent():
    # Arithmetic does not need to know the defense's name.
    assert spot_to_yardline100("XYZ 22", "PHI") == spot_to_yardline100("DAL 22", "PHI") == 22


def test_advance():
    assert advance("NYG 28", 3, "DAL", "NYG") == "NYG 25"      # spot foul: credited to the foul spot
    assert advance("BAL 48", 9, "BAL", "CIN") == "CIN 43"      # across midfield
    assert advance("ATL 19", 19, "MIN", "ATL") == "ATL 0"      # touchdown
    assert advance("BUF 2", -2, "BUF", "NE") == "BUF 0"        # safety
    assert advance("50", 0, "BUF", "NYJ") == MIDFIELD
    with pytest.raises(SpotError):
        advance("ATL 19", 20, "MIN", "ATL")                    # past the goal line


@pytest.mark.parametrize("spot,canon", [("DAL 22", "DAL 22"), ("MID 50", "50"), ("KC 50", "50"),
                                        ("50", "50"), ("DAL 07", "DAL 7"), ("NO 0", "NO 0")])
def test_canonical_spot(spot, canon):
    assert canonical_spot(spot) == canon


@pytest.mark.parametrize("bad", ["", "22", "DAL", "DAL 51", "DAL 100", "dal 22", "DAL  22", "DAL-22",
                                 "DALLAS 22", " DAL 22", "DAL 22 ", "-5", None, 22])
def test_parse_rejects_non_spots(bad):
    with pytest.raises(SpotError):
        parse_spot(bad)


def test_errors():
    with pytest.raises(SpotError):
        spot_to_yardline100("DAL 22", "")
    with pytest.raises(SpotError):
        yardline100_to_spot(101, "PHI", "DAL")
    with pytest.raises(SpotError):
        yardline100_to_spot(-1, "PHI", "DAL")
    with pytest.raises(SpotError):
        yardline100_to_spot(20, "PHI", "PHI")
