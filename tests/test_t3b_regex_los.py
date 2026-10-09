"""R0 + LOS (the T3b fairness arm). Every play here is a real train-season (2015-2022)
desc; the expected labels are the frozen ground truth for that play, and the comment
gives the spot arithmetic that produces it."""
from __future__ import annotations

import pytest

from playparse.eval.baselines.regex_los import RegexLOSPredictor, parse_desc_los
from playparse.eval.baselines.regex_parser import parse_desc
from playparse.ffscore.schema import Credit, PlayLabel


def L(*credits):
    return PlayLabel(False, tuple(Credit(p, s, v) for p, s, v in credits))


CASES = [
    # spot foul: LOS NYG 28, foul at NYG 25 -> 3 (text says 5)
    ("(4:10) 20-T.Pollard right tackle to NYG 23 for 5 yards (58-T.Davis). PENALTY on DAL-70-Z.Martin, "
     "Offensive Holding, 10 yards, enforced at NYG 25.", "DAL", "NYG 28", L(("T.Pollard", "rush_yds", 3))),
    # forward bounce while behind the line, opponent recovers: min(CIN 23, LOS CIN 25) - LOS = -2
    ("(6:47) 33-R.Burkhead right end to CIN 22 for -3 yards (72-M.Bennett). FUMBLES (72-M.Bennett), RECOVERED by "
     "SEA-54-B.Wagner at CIN 23. 54-B.Wagner for 23 yards, TOUCHDOWN.", "CIN", "CIN 25",
     L(("R.Burkhead", "rush_yds", -2), ("R.Burkhead", "fumble_lost", 1))),
    # backward fumble, first touched at TB 44: measured to the touch, -1 (R0 said -21)
    ("(12:44) (No Huddle) 34-C.Sims right end to TB 44 for -1 yards (53-J.Brinkley). FUMBLES (53-J.Brinkley), "
     "touched at TB 44, RECOVERED by NYG-31-T.Wade at TB 24. 31-T.Wade to TB 24 for no gain (11-A.Humphries).",
     "TB", "TB 45", L(("C.Sims", "rush_yds", -1), ("C.Sims", "fumble_lost", 1))),
    # botched snap he recovered and ran with: BUF 13 - LOS BUF 11 = 2 (R0 gave nothing)
    ("(14:13) (Shotgun) 5-T.Taylor FUMBLES (Aborted) at BUF 5, and recovers at BUF 6. 5-T.Taylor to BUF 13 for "
     "7 yards (90-K.Langford).", "BUF", "BUF 11", L(("T.Taylor", "rush_yds", 2))),
    # ... but one that ends behind the line is a 0-yard rush
    ("(14:20) (Shotgun) 3-C.Palmer FUMBLES (Aborted) at ARI 20, and recovers at ARI 20. 3-C.Palmer to ARI 24 for "
     "4 yards (91-S.Richardson).", "ARI", "ARI 26", L()),
    # touchback out of the end zone: possession lost
    ("(9:42) 31-M.Jones left end to NYG 2 for 5 yards (31-T.Wade). FUMBLES (31-T.Wade), ball out of bounds in "
     "End Zone, Touchback.", "WAS", "NYG 7", L(("M.Jones", "rush_yds", 5), ("M.Jones", "fumble_lost", 1))),
    # lateral chain: receiver 4, middle leg -6 uncredited, last taker KC 32 -> recovery KC 27 = -5, pass -7
    ("(:03) (Shotgun) 11-A.Smith pass short middle to 12-A.Wilson to KC 38 for 4 yards. Lateral to 10-T.Hill to "
     "KC 32 for -6 yards. Lateral to 72-E.Fisher to KC 32 for no gain (54-L.David). FUMBLES (54-L.David), "
     "RECOVERED by TB-93-G.McCoy at KC 27.", "KC", "KC 34",
     L(("A.Smith", "pass_yds", -7), ("A.Wilson", "rec", 1), ("A.Wilson", "rec_yds", 4),
       ("E.Fisher", "rec_yds", -5), ("E.Fisher", "fumble_lost", 1))),
    # catch behind the line then a lateral touchdown: receiver floored at 0, the rest to the lateral taker
    ("(12:40) (Shotgun) 9-B.Petty pass short left to 15-B.Marshall to LA 10 for -6 yards. Lateral to 29-B.Powell "
     "for 10 yards, TOUCHDOWN.", "NYJ", "LA 4",
     L(("B.Petty", "pass_yds", 4), ("B.Petty", "pass_td", 1), ("B.Marshall", "rec", 1),
       ("B.Powell", "rec_yds", 4), ("B.Powell", "rec_td", 1))),
    # a handoff after a recovered fumble: only the final runner, from the line (DAL 16 -> DAL 17 = -1)
    ("(9:28) 2-B.Hoyer to DAL 18 for -2 yards. FUMBLES, and recovers at DAL 18. 2-B.Hoyer to DAL 20 for -2 yards. "
     "Handoff to 24-J.Howard to DAL 17 for 3 yards (98-T.Crawford).", "CHI", "DAL 16", L(("J.Howard", "rush_yds", -1))),
    # a substitution note R0's preamble list missed
    ("(14:55) 7-G.Smith back in at quarterback. 9-K.Walker right guard to SEA 33 for 5 yards (97-D.Lawrence, "
     "58-B.Okereke).", "SEA", "SEA 28", L(("K.Walker", "rush_yds", 5))),
    # text states the official number
    ("(9:31) (No Huddle, Shotgun) 29-D.Murray left end to PHI 48 for 4 yards (53-J.Brinkley, 31-T.Wade). FUMBLES "
     "(53-J.Brinkley), touched at PHI 47, RECOVERED by NYG-93-G.Selvie at PHI 45. 93-G.Selvie to PHI 45 for no "
     "gain (87-B.Celek). Officially, a rush for 3 yards.", "PHI", "PHI 44",
     L(("D.Murray", "rush_yds", 3), ("D.Murray", "fumble_lost", 1))),
]


@pytest.mark.parametrize("desc,posteam,los,expected", CASES)
def test_r0los_real_plays(desc, posteam, los, expected):
    assert parse_desc_los(desc, posteam, los).matches(expected)


def test_ordinary_plays_agree_with_r0():
    for desc, pos, los in [
        ("(3:12) (Shotgun) 1-J.Hurts pass short right to 11-A.Brown to DAL 22 for 14 yards (21-T.Diggs).", "PHI",
         "DAL 36"),
        ("(8:28) 33-D.Cook left end for 19 yards, TOUCHDOWN.", "MIN", "ATL 19"),
        ("(1:00) 4-D.Carr pass incomplete short left to 89-A.Cooper.", "LV", "LV 30"),
    ]:
        assert parse_desc_los(desc, pos, los) == parse_desc(desc, pos)


def test_without_los_it_is_r0():
    desc = CASES[0][0]
    assert parse_desc_los(desc, "DAL", None) == parse_desc(desc, "DAL")


def test_predictor_needs_los():
    p = RegexLOSPredictor()
    rec = {"desc": CASES[0][0], "posteam": "DAL", "los": "NYG 28"}
    assert PlayLabel.from_json(p.predict_batch([rec])[0]).matches(CASES[0][3])
    with pytest.raises(KeyError):
        p.predict_batch([{"desc": "x", "posteam": "DAL"}])
