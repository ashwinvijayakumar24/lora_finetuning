"""Schema v2 (T3b): strict parsing, canonical serialization, conversion to v1, and
the v2 ground truth on real train-season plays (tests/fixtures/t3b_cases.json,
all from 2019; the expected v2 labels were written by reading each `desc`)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from playparse.data.ground_truth import build_label
from playparse.data.ground_truth_v2 import build_label_v2, literal_kind, los_from_row, spot_in_desc
from playparse.eval.metrics import score_example
from playparse.ffscore.schema import Credit, PlayLabel, SchemaError
from playparse.ffscore.schema_v2 import CreditV2, PlayLabelV2, from_v1, to_v1, v2_text_to_v1_text

CASES = {c["case"]: c for c in json.loads((Path(__file__).parent / "fixtures" / "t3b_cases.json").read_text())}


def Y(player, stat, to, frm=None):
    return CreditV2(player, stat, to=to, from_=frm)


def N(player, stat, value=1):
    return CreditV2(player, stat, value=value)


# ------------------------------------------------------------------ parsing


def test_round_trip_json():
    lab = PlayLabelV2(False, (N("A.Brown", "rec"), Y("J.Hurts", "pass_yds", "DAL 22"),
                              Y("Q.Enunwa", "rec_yds", "NYJ 21", "NYJ 33")))
    text = lab.to_json()
    # canonical order: stat vocabulary order, then player; "from" written before "to"
    assert text == ('{"nullified":false,"credits":[{"player":"J.Hurts","stat":"pass_yds","to":"DAL 22"},'
                    '{"player":"A.Brown","stat":"rec","value":1},'
                    '{"player":"Q.Enunwa","stat":"rec_yds","from":"NYJ 33","to":"NYJ 21"}]}')
    assert PlayLabelV2.from_json(text) == lab.canonical()


def test_parse_canonicalizes_spots():
    lab = PlayLabelV2.from_json('{"nullified":false,"credits":[{"player":"A","stat":"rush_yds","to":"MID 50"}]}')
    assert lab.credits[0].to == "50"


@pytest.mark.parametrize("bad", [
    "not json",
    '{"nullified":false}',
    '{"nullified":"no","credits":[]}',
    '{"nullified":false,"credits":[],"extra":1}',
    # a yardage credit must use spots, not a value
    '{"nullified":false,"credits":[{"player":"A","stat":"rush_yds","value":5}]}',
    '{"nullified":false,"credits":[{"player":"A","stat":"rush_yds","to":"DAL 5","value":5}]}',
    '{"nullified":false,"credits":[{"player":"A","stat":"rush_yds","from":"DAL 5"}]}',
    '{"nullified":false,"credits":[{"player":"A","stat":"rush_yds","to":"DAL 55"}]}',
    '{"nullified":false,"credits":[{"player":"A","stat":"rush_yds","to":22}]}',
    # a count credit keeps v1's value
    '{"nullified":false,"credits":[{"player":"A","stat":"rec","to":"DAL 5"}]}',
    '{"nullified":false,"credits":[{"player":"A","stat":"rec","value":true}]}',
    '{"nullified":false,"credits":[{"player":"","stat":"rec","value":1}]}',
    '{"nullified":false,"credits":[{"player":"A","stat":"sacks","value":1}]}',
    '{"nullified":true,"credits":[{"player":"A","stat":"rec","value":1}]}',
])
def test_strict_parser_rejects(bad):
    with pytest.raises(SchemaError):
        PlayLabelV2.from_json(bad)


# ------------------------------------------------------------------ to_v1


def test_to_v1_computes_yards_from_los_and_from():
    lab = PlayLabelV2(False, (Y("S.Darnold", "pass_yds", "NYJ 21"), N("Ro.Anderson", "rec"),
                              Y("Ro.Anderson", "rec_yds", "NYJ 33"), Y("Q.Enunwa", "rec_yds", "NYJ 21", "NYJ 33")))
    v1 = to_v1(lab, "NYJ 25", "NYJ")
    assert v1.matches(PlayLabel(False, (Credit("S.Darnold", "pass_yds", -4), Credit("Ro.Anderson", "rec", 1),
                                        Credit("Ro.Anderson", "rec_yds", 8), Credit("Q.Enunwa", "rec_yds", -12))))


def test_to_v1_drops_zero_and_merges_like_v1():
    # back to the line of scrimmage = 0 yards = no credit (v1 rule 9)
    zero = PlayLabelV2(False, (Y("A", "rush_yds", "PHI 30"), N("A", "fumble_lost")))
    assert to_v1(zero, "PHI 30", "PHI") == PlayLabel(False, (Credit("A", "fumble_lost", 1),))
    # two legs for one player are summed (v1 rule 11)
    two = PlayLabelV2(False, (Y("A", "rec_yds", "PHI 40"), Y("A", "rec_yds", "PHI 50", "PHI 45")))
    assert to_v1(two, "PHI 30", "PHI") == PlayLabel(False, (Credit("A", "rec_yds", 10 + 5),))
    back = PlayLabelV2(False, (Y("A", "rec_yds", "PHI 40"), Y("A", "rec_yds", "PHI 35", "PHI 45")))
    assert to_v1(back, "PHI 30", "PHI") == PlayLabel(False, ())  # +10 then -10 sums to 0: dropped


def test_to_v1_needs_a_start():
    lab = PlayLabelV2(False, (Y("A", "rush_yds", "PHI 35"),))
    with pytest.raises(SchemaError):
        to_v1(lab, None, "PHI")
    with pytest.raises(SchemaError):
        to_v1(lab, "PHI 30", None)
    assert to_v1(PlayLabelV2(False, (Y("A", "rush_yds", "PHI 35", "PHI 30"),)), None, "PHI") == PlayLabel(
        False, (Credit("A", "rush_yds", 5),))
    assert to_v1(PlayLabelV2(True, ()), None, None) == PlayLabel(True, ())


def test_from_v1_then_to_v1_is_identity():
    v1 = PlayLabel(False, (Credit("J.Hurts", "pass_yds", 48), Credit("A.Brown", "rec", 1),
                           Credit("A.Brown", "rec_yds", 48), Credit("A.Brown", "fumble_lost", 1)))
    v2 = from_v1(v1, "PHI 30", "PHI", "DAL")
    assert [c.to for c in v2.credits if c.is_yardage] == ["DAL 22", "DAL 22"]
    assert to_v1(v2, "PHI 30", "PHI").canonical() == v1.canonical()


# ------------------------------------------------------------------ v2 text -> v1 text


def test_v2_text_conversion_keeps_strictness():
    v2 = '{"nullified":false,"credits":[{"player":"A","stat":"rush_yds","to":"PHI 35"}]}'
    text, err = v2_text_to_v1_text(v2, "PHI 30", "PHI")
    assert err is None and text == '{"nullified":false,"credits":[{"player":"A","stat":"rush_yds","value":5}]}'
    s = score_example(PlayLabel(False, (Credit("A", "rush_yds", 5),)), text)
    assert s.valid and s.strict_valid and s.exact
    wrapped, _ = v2_text_to_v1_text("Here: " + v2 + " done", "PHI 30", "PHI")
    s = score_example(PlayLabel(False, (Credit("A", "rush_yds", 5),)), wrapped)
    assert s.valid and not s.strict_valid and s.exact


@pytest.mark.parametrize("raw", [
    # valid v1 but not v2: must not be rescued
    '{"nullified":false,"credits":[{"player":"A","stat":"rush_yds","value":5}]}',
    "no json here",
    '{"nullified":false,"credits":[{"player":"A","stat":"rush_yds","to":"PHI 77"}]}',
])
def test_invalid_v2_scores_invalid(raw):
    text, err = v2_text_to_v1_text(raw, "PHI 30", "PHI")
    assert err is not None
    s = score_example(PlayLabel(False, (Credit("A", "rush_yds", 5),)), text)
    assert not s.valid and not s.exact


# ------------------------------------------------------------------ v2 ground truth on real plays

EXPECTED_V2 = {
    # "pass short left to 11-Ro.Anderson to NYJ 33 for 8 yards. Lateral to 81-Q.Enunwa to NYJ 35 for 2
    # yards. FUMBLES, RECOVERED by BUF-93-T.Murphy at NYJ 21." LOS NYJ 25. The lateral leg starts
    # where Anderson's catch ended (NYJ 33) and ends at the recovery spot.
    "lateral_reception_fumble": PlayLabelV2(False, (
        Y("S.Darnold", "pass_yds", "NYJ 21"), N("Ro.Anderson", "rec"), Y("Ro.Anderson", "rec_yds", "NYJ 33"),
        Y("Q.Enunwa", "rec_yds", "NYJ 21", "NYJ 33"), N("Q.Enunwa", "fumble_lost"))),
    # "to 87-T.Kelce to DET 36 for 10 yards. Lateral to 25-L.McCoy to DET 13 for 23 yards" (LOS DET 46)
    "lateral_reception": PlayLabelV2(False, (
        Y("P.Mahomes", "pass_yds", "DET 13"), N("T.Kelce", "rec"), Y("T.Kelce", "rec_yds", "DET 36"),
        Y("L.McCoy", "rec_yds", "DET 13", "DET 36"))),
    # "8-L.Jackson right end to BAL 48 for 3 yards. Lateral to 3-R.Griffin III pushed ob at CIN 43"
    "lateral_rush": PlayLabelV2(False, (
        Y("L.Jackson", "rush_yds", "BAL 48"), Y("R.Griffin III", "rush_yds", "CIN 43", "BAL 48"))),
    # "to NYG 23 for 5 yards ... Offensive Holding ... enforced at NYG 25": credited to the foul spot
    "spot_foul": PlayLabelV2(False, (Y("T.Pollard", "rush_yds", "NYG 25"),)),
    # "to CAR 8 for -7 yards. FUMBLES ... RECOVERED by LA-50-S.Ebukam at CAR 10": to the recovery spot
    "fumble_backward": PlayLabelV2(False, (Y("C.Newton", "rush_yds", "CAR 10"), N("C.Newton", "fumble_lost"))),
    # "left end for 19 yards, TOUCHDOWN" from ATL 19: the defense's goal line
    "rush_td": PlayLabelV2(False, (Y("D.Cook", "rush_yds", "ATL 0"), N("D.Cook", "rush_td"))),
    # LOS "MID 50" -> "50"; "ran ob at NYJ 38 for 12 yards"
    "midfield_los": PlayLabelV2(False, (
        Y("J.Allen", "pass_yds", "NYJ 38"), N("D.Singletary", "rec"), Y("D.Singletary", "rec_yds", "NYJ 38"))),
    # "tackled in End Zone for -2 yards, SAFETY" from BUF 2: the offense's own goal line
    "safety": PlayLabelV2(False, (Y("F.Gore", "rush_yds", "BUF 0"),)),
    "no_play": PlayLabelV2(True, ()),
}


@pytest.mark.parametrize("case", sorted(EXPECTED_V2))
def test_ground_truth_v2_real_plays(case):
    row = CASES[case]["row"]
    v1 = build_label(row).label
    assert v1.to_json() == CASES[case]["label"]
    res = build_label_v2(row, v1)
    assert res.label_v2 == EXPECTED_V2[case].canonical()
    assert res.round_trip
    assert to_v1(res.label_v2, res.los, row["posteam"]).canonical() == v1.canonical()


def test_los_from_row():
    assert los_from_row({"yrdln": "MID 50"}) == "50"
    assert los_from_row({"yrdln": "PHI 30"}) == "PHI 30"
    assert los_from_row({"yrdln": None, "yardline_100": 70, "posteam": "PHI", "defteam": "DAL"}) == "PHI 30"
    assert los_from_row({"yrdln": float("nan"), "yardline_100": float("nan")}) is None


def test_literal_spots():
    desc = CASES["spot_foul"]["row"]["desc"]
    assert spot_in_desc("NYG 25", desc) and spot_in_desc("NYG 23", desc) and not spot_in_desc("NYG 2", desc)
    assert literal_kind("NYG 25", desc) == "literal"
    assert literal_kind("ATL 0", CASES["rush_td"]["row"]["desc"]) == "goal_line"
    assert literal_kind("NYG 24", desc) == "computed"
    assert spot_in_desc("50", "to 50 for 3 yards") and not spot_in_desc("50", "for 50 yards")
