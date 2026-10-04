"""Rejection filter: yardage/name parsing, each check, config toggles, stats, precision."""
from __future__ import annotations

from collections.abc import Mapping

import pytest

from playparse.distill.reject import (
    FilterConfig,
    drop_stats,
    filter_precision,
    filter_record,
    filter_samples,
    name_in_desc,
    yardage_values,
)
from playparse.ffscore.schema import Credit, PlayLabel

PASS_DESC = ("(3:12) (Shotgun) 1-J.Hurts pass short right to 11-A.Brown to DAL 22 for 14 yards "
             "(21-T.Diggs). FUMBLES (21-T.Diggs), RECOVERED by DAL-55-D.Lawrence at DAL 20.")
PASS_LABEL = PlayLabel(False, (
    Credit("J.Hurts", "pass_yds", 14), Credit("A.Brown", "rec", 1),
    Credit("A.Brown", "rec_yds", 14), Credit("A.Brown", "fumble_lost", 1)))


def lab(*credits, nullified=False) -> str:
    return PlayLabel(nullified, tuple(Credit(*c) for c in credits)).to_json()


# --------------------------------------------------------------------------- yardage parsing


@pytest.mark.parametrize("desc, expected", [
    (PASS_DESC, {14}),
    ("(10:21) 26-S.Barkley up the middle to PHI 35 for -2 yards (90-D.Smith).", {-2}),
    ("(9:00) 26-S.Barkley left end to PHI 30 for no gain (54-J.Doe).", {0}),
    ("(2:00) 26-S.Barkley right guard to DAL 1 for 1 yard (54-J.Doe).", {1}),
    ("(0:41) 1-J.Hurts pass deep left to 11-A.Brown for 45 yards, TOUCHDOWN.", {45}),
    ("(5:00) 22-D.Henry left tackle to TEN 20 for a loss of 3 yards (99-A.Donald).", {-3}),
    # Unicode minus from a scraped source.
    ("(5:00) 22-D.Henry left tackle to TEN 20 for −3 yards.", {-3}),
    # Lateral: two gain segments, both are valid values.
    ("(0:04) 9-M.Stafford pass short middle to 10-C.Kupp to LA 45 for 5 yards. 10-C.Kupp "
     "lateral to 11-D.Jackson to SF 43 for 12 yards (24-K.Moseley).", {5, 12}),
    # Penalty distance and field positions are not gain phrases.
    ("(8:30) 1-J.Hurts pass short left to 11-A.Brown to PHI 40 for 12 yards. PENALTY on "
     "DAL-21-T.Diggs, Unnecessary Roughness, 15 yards, enforced at PHI 40.", {12}),
    ("(8:30) (Shotgun) 1-J.Hurts pass incomplete short left to 11-A.Brown.", set()),
    ("(1:00) 1-J.Hurts sacked at PHI 25 for -8 yards (sa 90-M.Parsons).", {-8}),
])
def test_yardage_values(desc, expected):
    assert yardage_values(desc) == expected


# --------------------------------------------------------------------------- name matching


@pytest.mark.parametrize("name, desc, ok", [
    ("A.Brown", "1-J.Hurts pass to 11-A.Brown for 3 yards", True),
    ("D.Lawrence", "RECOVERED by DAL-55-D.Lawrence at DAL 20.", True),
    ("A.Brown", "pass to 11-A.Brown.", True),
    ("Brown", "pass to 11-A.Brown for 3 yards", False),          # surname only
    ("J.Brown", "pass to 11-A.J.Brown for 3 yards", False),      # tail of another name
    ("D.Smith", "pass to 19-D.Smith-Schuster for 3 yards", False),
    ("D.Smith-Schuster", "pass to 19-D.Smith-Schuster for 3 yards", True),
    ("A.St. Brown", "pass to 14-A.St. Brown for 9 yards", True),
    ("A.Browns", "pass to 11-A.Brown for 3 yards", False),
    ("a.brown", "pass to 11-A.Brown for 3 yards", False),         # case-sensitive
    ("", "anything", False),
])
def test_name_in_desc(name, desc, ok):
    assert name_in_desc(name, desc) is ok


# --------------------------------------------------------------------------- checks


def test_correct_label_kept_with_k_agreeing_samples():
    good = PASS_LABEL.to_json()
    d = filter_samples(PASS_DESC, [good, good, good])
    assert d.keep and d.label.matches(PASS_LABEL)
    assert all(o.passed for o in d.checks.values())


def test_agreement_is_order_insensitive():
    shuffled = PlayLabel(False, tuple(reversed(PASS_LABEL.credits)))
    import json
    raw = json.dumps({"nullified": False, "credits": [
        {"player": c.player, "stat": c.stat, "value": c.value} for c in shuffled.credits]})
    assert filter_samples(PASS_DESC, [PASS_LABEL.to_json(), raw, " " + raw + "\n"]).keep


def test_invalid_json_dropped_by_schema_and_agreement():
    good = PASS_LABEL.to_json()
    d = filter_samples(PASS_DESC, [good, "Sure! Here is the JSON: {", good])
    assert not d.keep
    assert d.failed == ["schema", "agreement"]
    assert d.first_failure == "schema"


def test_wrong_yardage_dropped():
    s = lab(("J.Hurts", "pass_yds", 22), ("A.Brown", "rec", 1), ("A.Brown", "rec_yds", 22))  # 22 = yard line
    d = filter_samples(PASS_DESC, [s, s, s])
    assert not d.keep and d.failed == ["yardage"]
    assert "pass_yds=22" in d.checks["yardage"].reason


def test_sign_matters_for_negative_yards():
    desc = "(10:21) 26-S.Barkley up the middle to PHI 35 for -2 yards (90-D.Smith)."
    assert filter_samples(desc, [lab(("S.Barkley", "rush_yds", -2))]).keep
    assert not filter_samples(desc, [lab(("S.Barkley", "rush_yds", 2))]).keep


def test_no_gain_allows_zero():
    desc = "(9:00) 26-S.Barkley left end to PHI 30 for no gain (54-J.Doe)."
    assert filter_samples(desc, [lab(("S.Barkley", "rush_yds", 0))]).keep


def test_penalty_yards_credited_as_receiving_dropped():
    desc = ("(8:30) 1-J.Hurts pass short left to 11-A.Brown to PHI 40 for 12 yards. PENALTY on "
            "DAL-21-T.Diggs, Unnecessary Roughness, 15 yards, enforced at PHI 40.")
    s = lab(("J.Hurts", "pass_yds", 27), ("A.Brown", "rec", 1), ("A.Brown", "rec_yds", 27))
    assert filter_samples(desc, [s]).failed == ["yardage"]


def test_hallucinated_name_dropped():
    s = lab(("J.Hurts", "pass_yds", 14), ("D.Smith", "rec", 1), ("D.Smith", "rec_yds", 14))
    d = filter_samples(PASS_DESC, [s, s, s])
    assert not d.keep and d.failed == ["names"]
    assert "D.Smith" in d.checks["names"].reason


def test_count_stats_are_not_yardage_checked():
    desc = "(0:41) 1-J.Hurts pass deep left to 11-A.Brown for 45 yards, TOUCHDOWN."
    s = lab(("J.Hurts", "pass_yds", 45), ("J.Hurts", "pass_td", 1), ("A.Brown", "rec", 1),
            ("A.Brown", "rec_yds", 45), ("A.Brown", "rec_td", 1))
    assert filter_samples(desc, [s]).keep


def test_disagreeing_samples_dropped():
    a = PASS_LABEL.to_json()
    b = lab(("J.Hurts", "pass_yds", 14), ("A.Brown", "rec", 1), ("A.Brown", "rec_yds", 14))  # no fumble
    d = filter_samples(PASS_DESC, [a, a, b])
    assert not d.keep and d.failed == ["agreement"]
    assert "[2]" in d.checks["agreement"].reason


def test_nullified_play_with_no_credits_kept():
    desc = ("(4:10) 1-J.Hurts pass short right to 11-A.Brown to DAL 30 for 8 yards. PENALTY on "
            "PHI-62-J.Kelce, Offensive Holding, 10 yards, enforced at PHI 38 - No Play.")
    s = lab(nullified=True)
    assert filter_samples(desc, [s, s, s]).keep


def test_config_toggles():
    hallucinated = lab(("X.Nobody", "rush_yds", 99))
    # R8: only "does it parse" is enforced, so garbage-but-valid labels are kept.
    d = filter_samples(PASS_DESC, [hallucinated], FilterConfig.r8())
    assert d.keep
    assert d.checks["yardage"].passed is None and d.checks["names"].passed is None
    # Turning one check off lets exactly that failure through.
    only_names_off = FilterConfig(names=False)
    s = lab(("D.Smith", "rec_yds", 14))
    assert filter_samples(PASS_DESC, [s, s, s], only_names_off).keep
    assert not filter_samples(PASS_DESC, [s, s, s]).keep
    # An unparseable sample 0 can never be kept, even with every check off.
    off = FilterConfig(schema=False, yardage=False, names=False, agreement=False)
    assert not filter_samples(PASS_DESC, ["not json"], off).keep
    # schema=False tolerates bad extra samples when agreement is off.
    assert filter_samples(PASS_DESC, [PASS_LABEL.to_json(), "junk"],
                          FilterConfig(schema=False, agreement=False)).keep


def test_code_fenced_output_is_parsed_but_prose_is_not():
    good = PASS_LABEL.to_json()
    fenced = f"```json\n{good}\n```"
    assert filter_samples(PASS_DESC, [fenced, good, f"```\n{good}```"]).keep
    assert filter_samples(PASS_DESC, [fenced], FilterConfig.r8()).keep
    assert not filter_samples(PASS_DESC, [f"Here you go: {good}"]).keep


def test_empty_samples_dropped():
    assert not filter_samples(PASS_DESC, []).keep


class GroundTruthTripwire(Mapping):
    """A record that raises if anything but desc (and ids) is read."""

    def __init__(self, data):
        self._d = data

    def __getitem__(self, key):
        if key in ("label", "ground_truth"):
            raise AssertionError("the filter read ground truth")
        return self._d[key]

    def __iter__(self):
        raise AssertionError("the filter iterated the whole record")

    def __len__(self):
        return len(self._d)


def test_filter_record_never_reads_ground_truth():
    rec = GroundTruthTripwire({"game_id": "g", "play_id": 1, "desc": PASS_DESC,
                               "label": PASS_LABEL.to_json()})
    good = PASS_LABEL.to_json()
    assert filter_record(rec, [good, good, good]).keep
    # And it works on a record with the label stripped entirely.
    assert filter_record({"desc": PASS_DESC}, [good]).keep


# --------------------------------------------------------------------------- stats and precision


def test_drop_stats_counts():
    good = PASS_LABEL.to_json()
    wrong_yds = lab(("J.Hurts", "pass_yds", 22))
    both_bad = lab(("Z.Ghost", "pass_yds", 22))
    decisions = [
        filter_samples(PASS_DESC, [good, good, good]),
        filter_samples(PASS_DESC, [wrong_yds] * 3),
        filter_samples(PASS_DESC, [both_bad] * 3),
        filter_samples(PASS_DESC, ["{", good, good]),
    ]
    s = drop_stats(decisions)
    assert (s["n"], s["kept"], s["dropped"]) == (4, 1, 3)
    assert s["failed_by_check"] == {"schema": 1, "yardage": 2, "names": 1, "agreement": 1}
    assert s["first_failure_by_check"] == {"schema": 1, "yardage": 2, "names": 0, "agreement": 0}
    assert sum(s["first_failure_by_check"].values()) == s["dropped"]
    assert s["sole_failure_by_check"] == {"schema": 0, "yardage": 1, "names": 0, "agreement": 0}


def test_filter_precision():
    gt = {("g", 1): PASS_LABEL, ("g", 2): PlayLabel(True, ()), ("g", 3): PASS_LABEL}
    kept = {("g", 1): PASS_LABEL.to_json(),         # correct
            ("g", 2): PlayLabel(True, ()),           # correct
            ("g", 3): lab(("J.Hurts", "pass_yds", 14)),  # wrong: missing credits
            ("g", 4): PASS_LABEL}                    # no ground truth
    r = filter_precision(kept, gt, buckets={("g", 1): "fumble", ("g", 2): "penalty_nullified",
                                             ("g", 3): "fumble"})
    assert r["n_kept"] == 4 and r["n_evaluated"] == 3 and r["n_correct"] == 2
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["missing_ground_truth"] == 1
    lo, hi = r["precision_ci95"]
    assert lo < 2 / 3 < hi
    assert r["per_bucket"]["fumble"] == {"n": 2, "correct": 1, "precision": 0.5}
    assert filter_precision({}, gt)["precision"] is None
