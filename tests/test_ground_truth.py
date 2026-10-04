"""Ground-truth builder tests on real nflverse rows.

Each fixture in tests/fixtures/gt_cases.json is one real play-by-play row (only the
columns the builder reads). The expected labels below were written by reading the
`desc` text, not by running the builder; the comment on each case quotes the part of
the text that decides it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from playparse.data.ground_truth import build_label, is_nullified
from playparse.ffscore.schema import Credit, PlayLabel

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "gt_cases.json").read_text())
ROWS = {c["case"]: c["row"] for c in FIXTURES}


def L(*credits: tuple[str, str, int], nullified: bool = False) -> PlayLabel:
    return PlayLabel(nullified, tuple(Credit(p, s, v) for p, s, v in credits))


NULL = L(nullified=True)
EMPTY = L()

EXPECTED: dict[str, PlayLabel] = {
    # "B.Nix pass short left to J.Williams to DEN 36 for 3 yards"
    "completion": L(("B.Nix", "pass_yds", 3), ("J.Williams", "rec", 1), ("J.Williams", "rec_yds", 3)),
    # "pass incomplete" -> the play counts, but nobody is credited
    "incompletion": EMPTY,
    # "P.Mahomes pass ... INTERCEPTED by R.Smith"
    "interception": L(("P.Mahomes", "int", 1)),
    # interception overturned on replay to an incompletion: final ruling, no credits, not nullified
    "interception_reversed": EMPTY,
    # sacks never produce rushing yards and do not reduce passing yards
    "sack": EMPTY,
    # "J.Allen sacked ... FUMBLES ... RECOVERED by ARI" -> sack fumble lost counts
    "sack_fumble_lost": L(("J.Allen", "fumble_lost", 1)),
    # "recovered by DAL-1-J.Tolbert" (own team) -> not lost
    "sack_fumble_recovered_by_offense": EMPTY,
    # scrambles are runs
    "scramble": L(("K.Murray", "rush_yds", 12)),
    # challenge REVERSED a spot: the final call "for 3 yards" counts, not the first "for 2 yards"
    "replay_reversed_spot": L(("B.Nix", "rush_yds", 3)),
    "rush": L(("J.Dobbins", "rush_yds", 2)),
    # "for no gain" -> a zero-yard credit is dropped (rule 9)
    "rush_no_gain": EMPTY,
    "pass_td": L(("K.Murray", "pass_yds", 5), ("K.Murray", "pass_td", 1), ("Mi.Wilson", "rec", 1),
                 ("Mi.Wilson", "rec_yds", 5), ("Mi.Wilson", "rec_td", 1)),
    "rush_td": L(("J.Conner", "rush_yds", 3), ("J.Conner", "rush_td", 1)),
    # Text says "for 2 yards", but the ball came out and was recovered a yard back; the
    # official stat (which nflverse and the weekly totals agree on) is 1 yard. Known
    # text/label gap on fumble plays, documented in docs/phases/P1-data.md.
    "reception_fumble_lost": L(("B.Nix", "pass_yds", 1), ("J.McLaughlin", "rec", 1),
                               ("J.McLaughlin", "rec_yds", 1), ("J.McLaughlin", "fumble_lost", 1)),
    # "recovered by GB-80-B.Melton" (own team) -> yards count, no fumble lost
    "rush_fumble_recovered_by_offense": L(("J.Jacobs", "rush_yds", 5)),
    # "J.Hurts FUMBLES (Aborted) ... RECOVERED by GB" -> QB charged; 0-yard rush dropped
    "aborted_snap_qb_lost": L(("J.Hurts", "fumble_lost", 1)),
    # "K.Cousins Aborted. 67-D.Dalman FUMBLES" -> the center fumbled; he is charged
    "aborted_snap_center_lost": L(("D.Dalman", "fumble_lost", 1)),
    # "A.St. Brown ... for 1 yard. Lateral to J.Gibbs for 20 yards, TOUCHDOWN"
    # passer gets all 21 yards; St. Brown 1 rec yd; Gibbs 20 rec yds and the TD, no reception
    "lateral_pass_td": L(("J.Goff", "pass_yds", 21), ("J.Goff", "pass_td", 1), ("A.St. Brown", "rec", 1),
                         ("A.St. Brown", "rec_yds", 1), ("J.Gibbs", "rec_yds", 20), ("J.Gibbs", "rec_td", 1)),
    # "A.Richardson ... for 3 yards. Lateral to J.Downs ... for 13 yards"
    "lateral_run": L(("A.Richardson", "rush_yds", 3), ("J.Downs", "rush_yds", 13)),
    # Six laterals; nflverse records only the last one (J.Brendel) with a lumped yardage.
    # This label is KNOWN WRONG relative to the text and is pinned here so a change to
    # lateral handling is a deliberate decision. Bucket: lateral (noisy).
    "multi_lateral": L(("B.Purdy", "pass_yds", 34), ("R.Bell", "rec", 1), ("R.Bell", "rec_yds", 12),
                       ("J.Brendel", "rec_yds", 2)),
    # "J.Allen pass ... to A.Cooper ... for -2 yards. Lateral to J.Allen for 9 yards, TOUCHDOWN"
    # Cooper's -2 is in the official books as 0 (nflverse receiving_yards = 0), Allen +7.
    "passer_is_lateral_receiver": L(("J.Allen", "pass_yds", 7), ("J.Allen", "pass_td", 1), ("A.Cooper", "rec", 1),
                                    ("J.Allen", "rec_yds", 7), ("J.Allen", "rec_td", 1)),
    # "... Offensive Holding ... - No Play."
    "penalty_nullified": NULL,
    # "False Start ... - No Play." (nothing happened)
    "presnap_penalty": NULL,
    # "J.Starks ... for 32 yards ... PENALTY on GB ... enforced at GB 28": an offensive foul
    # during the run is enforced from the spot, so the official gain is only to the spot (8).
    "penalty_stands_spot_foul": L(("A.Rodgers", "pass_yds", 8), ("J.Starks", "rec", 1), ("J.Starks", "rec_yds", 8)),
    # declined penalty: the play stands in full
    "penalty_stands_declined": L(("D.Carr", "pass_yds", 41), ("A.Cooper", "rec", 1), ("A.Cooper", "rec_yds", 41)),
    # TD reversed to an incompletion: not nullified, just no credits
    "replay_reversed_td_to_incomplete": EMPTY,
    # "for no gain" reversed to "for 1 yard, TOUCHDOWN"
    "replay_reversed_to_td": L(("J.Gibbs", "rush_yds", 1), ("J.Gibbs", "rush_td", 1)),
    # passer AND receiver get two_pt on a successful pass try
    "two_point_pass_success": L(("C.Williams", "two_pt", 1), ("D.Swift", "two_pt", 1)),
    "two_point_run_success": L(("J.Conner", "two_pt", 1)),
    "two_point_fail": EMPTY,
    # "ATTEMPT SUCCEEDS. PENALTY ... enforced between downs": nflverse says no_play, but the
    # try counts (docs/issues/p1-two-point-dead-ball-penalty.md)
    "two_point_success_deadball_penalty": L(("A.Rodgers", "two_pt", 1), ("D.Adams", "two_pt", 1)),
    # "ATTEMPT SUCCEEDS. PENALTY ... Offensive Holding ... - No Play." -> really wiped out
    "two_point_nullified": NULL,
    # Goff fumbles (recovers), then is sacked and fumbles again, recovered by PHI.
    # nflverse lists him once (fumbled_1); the lost one is in fumble_recovery_2.
    "same_player_fumbles_twice": L(("J.Goff", "fumble_lost", 1)),
    # Rush's fumble is recovered by lineman T.Guyton, who then fumbles to HOU -> Guyton lost it
    "recoverer_fumbles_lost": L(("T.Guyton", "fumble_lost", 1)),
    # pick-six: int to the passer, the return TD is defensive and out of scope
    "interception_return_td": L(("D.Jones", "int", 1)),
    # Conner fumbles, teammate McBride recovers in the end zone: Conner keeps his 2 yards,
    # nobody lost a fumble, and a fumble-recovery TD is not a rush/rec TD
    "offense_fumble_recovery_td": L(("J.Conner", "rush_yds", 2)),
    # offsetting fouls, "- No Play."
    "no_play_with_offsetting": NULL,
}


def test_every_fixture_has_an_expectation():
    assert set(ROWS) == set(EXPECTED)


@pytest.mark.parametrize("case", sorted(EXPECTED))
def test_label(case):
    got = build_label(ROWS[case]).label
    assert got.matches(EXPECTED[case]), f"{case}: {ROWS[case]['desc']}\n got {got.to_json()}"


@pytest.mark.parametrize("case", sorted(EXPECTED))
def test_label_roundtrips_through_schema(case):
    label = build_label(ROWS[case]).label
    assert PlayLabel.from_json(label.to_json()) == label.canonical()


@pytest.mark.parametrize("case", sorted(EXPECTED))
def test_credited_names_appear_in_desc(case):
    res = build_label(ROWS[case])
    assert res.names_missing_from_desc == ()
    for c in res.label.credits:
        assert c.player in ROWS[case]["desc"]


def test_lateral_plays_are_flagged_noisy():
    for case in ("lateral_pass_td", "lateral_run", "multi_lateral", "passer_is_lateral_receiver"):
        assert build_label(ROWS[case]).noisy, case
    assert not build_label(ROWS["completion"]).noisy


def test_two_point_dead_ball_penalty_is_not_nullified():
    row = ROWS["two_point_success_deadball_penalty"]
    assert row["play_type"] == "no_play"  # what nflverse says
    assert not is_nullified(row)
    assert is_nullified(ROWS["two_point_nullified"])


def test_zero_yardage_credits_never_emitted():
    for case, row in ROWS.items():
        for c in build_label(row).label.credits:
            if c.stat.endswith("_yds"):
                assert c.value != 0, case


def test_missing_player_name_is_dropped_not_invented():
    row = dict(ROWS["completion"], receiver_player_name=None)
    res = build_label(row)
    assert {c.player for c in res.label.credits} == {"B.Nix"}
    assert res.dropped  # the builder reports what it could not credit


def test_same_player_same_yardage_stat_is_merged():
    # a receiver who is also the lateral receiver gets one summed rec_yds credit
    row = dict(ROWS["lateral_pass_td"], lateral_receiver_player_name="A.St. Brown", td_player_name="A.St. Brown")
    credits = build_label(row).label.credits
    rec_yds = [c for c in credits if c.stat == "rec_yds"]
    assert rec_yds == [Credit("A.St. Brown", "rec_yds", 21)]


def test_nan_values_are_missing():
    row = dict(ROWS["incompletion"], passing_yards=float("nan"), receiving_yards=float("nan"))
    assert build_label(row).label.matches(EMPTY)
