"""R0 regex parser on hand-written play descriptions (shapes copied from train-season text)."""
import pytest

from playparse.eval.baselines.regex_parser import RegexPredictor, parse_desc
from playparse.ffscore.schema import Credit, PlayLabel


def C(p, s, v):
    return Credit(p, s, v)


CASES = [
    (
        "complete pass",
        "PHI",
        "(3:12) (Shotgun) 1-J.Hurts pass short right to 11-A.Brown to DAL 22 for 14 yards (21-T.Diggs).",
        PlayLabel(False, (C("J.Hurts", "pass_yds", 14), C("A.Brown", "rec", 1), C("A.Brown", "rec_yds", 14))),
    ),
    (
        "completion for no gain keeps zero-yard credits",
        "NE",
        "(1:10) 10-M.Jones pass short right to 16-J.Meyers to NE 32 for no gain (30-M.Carter II).",
        PlayLabel(False, (C("M.Jones", "pass_yds", 0), C("J.Meyers", "rec", 1), C("J.Meyers", "rec_yds", 0))),
    ),
    ("incomplete", "DAL", "(5:16) (No Huddle) 4-D.Prescott pass incomplete short middle to 88-C.Lamb.", PlayLabel(False, ())),
    (
        "run",
        "TEN",
        "(9:50) 22-D.Henry left tackle to TEN 32 for 7 yards (92-J.Madubuike).",
        PlayLabel(False, (C("D.Henry", "rush_yds", 7),)),
    ),
    (
        "negative run",
        "NE",
        "(:09) 37-D.Harris left tackle to NE 40 for -2 yards (22-C.Gardner-Johnson).",
        PlayLabel(False, (C("D.Harris", "rush_yds", -2),)),
    ),
    (
        "scramble touchdown",
        "MIA",
        "(4:39) (No Huddle, Shotgun) 5-T.Bridgewater scrambles right end for 11 yards, TOUCHDOWN.",
        PlayLabel(False, (C("T.Bridgewater", "rush_yds", 11), C("T.Bridgewater", "rush_td", 1))),
    ),
    (
        "passing touchdown",
        "KC",
        "(8:28) 8-M.Moore pass deep right to 10-T.Hill for 40 yards, TOUCHDOWN [97-E.Griffen].",
        PlayLabel(
            False,
            (
                C("M.Moore", "pass_yds", 40),
                C("M.Moore", "pass_td", 1),
                C("T.Hill", "rec", 1),
                C("T.Hill", "rec_yds", 40),
                C("T.Hill", "rec_td", 1),
            ),
        ),
    ),
    ("sack", "SF", "(10:32) (Shotgun) 10-J.Garoppolo sacked at SF 8 for -10 yards (55-Z.Smith).", PlayLabel(False, ())),
    (
        "interception returned for TD credits only the INT",
        "CIN",
        "(4:25) (Shotgun) 14-A.Dalton pass short left intended for 25-G.Bernard INTERCEPTED by 91-Y.Ngakoue "
        "[90-T.Bryan] at CIN 23. 91-Y.Ngakoue for 23 yards, TOUCHDOWN.",
        PlayLabel(False, (C("A.Dalton", "int", 1),)),
    ),
    (
        "fumble lost after reception, yards to recovery spot",
        "IND",
        "(:40) (Shotgun) 12-A.Luck pass short left to 84-J.Doyle to CIN 15 for 15 yards (42-C.Fejedelem). "
        "FUMBLES (42-C.Fejedelem), RECOVERED by CIN-42-C.Fejedelem at CIN 17. 42-C.Fejedelem for 83 yards, TOUCHDOWN.",
        PlayLabel(
            False,
            (
                C("A.Luck", "pass_yds", 13),
                C("J.Doyle", "rec", 1),
                C("J.Doyle", "rec_yds", 13),
                C("J.Doyle", "fumble_lost", 1),
            ),
        ),
    ),
    (
        "fumble recovered by offense is not lost",
        "NE",
        "(:42) (No Huddle) 34-R.Burkhead right guard to HOU 26 for 5 yards (25-K.Jackson). FUMBLES (25-K.Jackson), "
        "recovered by NE-15-C.Hogan at HOU 30. 15-C.Hogan to HOU 30 for no gain (98-D.Reader).",
        PlayLabel(False, (C("R.Burkhead", "rush_yds", 1),)),
    ),
    (
        "strip sack lost",
        "MIA",
        "(11:26) (Shotgun) 7-N.Foles sacked at TB 18 for -7 yards (94-C.Nassib). FUMBLES (94-C.Nassib) "
        "[94-C.Nassib], RECOVERED by TB-93-N.Suh at TB 17. 93-N.Suh to TB 17 for no gain (65-B.Linder).",
        PlayLabel(False, (C("N.Foles", "fumble_lost", 1),)),
    ),
    (
        "aborted snap is a zero-yard run",
        "BAL",
        "(13:43) 5-J.Flacco FUMBLES (Aborted) at BUF 44, and recovers at 50. 5-J.Flacco to 50 for no gain (95-K.Williams).",
        PlayLabel(False, (C("J.Flacco", "rush_yds", 0),)),
    ),
    (
        "bobbled snap then a pass: the pass counts",
        "KC",
        "(12:56) (Shotgun) 15-P.Mahomes to CIN 8 for -5 yards. FUMBLES, and recovers at CIN 8. "
        "15-P.Mahomes pass short left to 10-T.Hill for 3 yards, TOUCHDOWN.",
        PlayLabel(
            False,
            (
                C("P.Mahomes", "pass_yds", 3),
                C("P.Mahomes", "pass_td", 1),
                C("T.Hill", "rec", 1),
                C("T.Hill", "rec_yds", 3),
                C("T.Hill", "rec_td", 1),
            ),
        ),
    ),
    (
        "penalty no play",
        "BAL",
        "(2:41) (Shotgun) 35-G.Edwards right guard to SF 33 for 6 yards (54-F.Warner). PENALTY on BAL, "
        "Illegal Formation, 5 yards, enforced at SF 39 - No Play.",
        PlayLabel(True, ()),
    ),
    (
        "defensive penalty stands: stated gain counts",
        "BUF",
        "(3:27) 17-J.Allen pass short left to 10-C.Beasley to CIN 3 for 8 yards (42-C.Fejedelem). PENALTY on "
        "CIN-42-C.Fejedelem, Face Mask (15 Yards), 2 yards, enforced at CIN 3.",
        PlayLabel(False, (C("J.Allen", "pass_yds", 8), C("C.Beasley", "rec", 1), C("C.Beasley", "rec_yds", 8))),
    ),
    (
        "offensive holding downfield: yards to the foul spot",
        "BUF",
        "(12:39) (Shotgun) 25-L.McCoy right tackle to BUF 35 for 6 yards (29-M.Humphrey). PENALTY on "
        "BUF-62-V.Ducasse, Offensive Holding, 10 yards, enforced at BUF 31.",
        PlayLabel(False, (C("L.McCoy", "rush_yds", 2),)),
    ),
    (
        "touchdown nullified by a spot foul is not a no-play",
        "TEN",
        "(13:23) 22-D.Henry right end for 62 yards, TOUCHDOWN NULLIFIED by Penalty. PENALTY on TEN-82-D.Walker, "
        "Offensive Holding, 10 yards, enforced at TEN 40.",
        PlayLabel(False, (C("D.Henry", "rush_yds", 2),)),
    ),
    (
        "replay reversal parses the corrected play",
        "TB",
        "(3:48) (Shotgun) 3-J.Winston pass incomplete short left to 19-B.Perriman. Tampa Bay challenged the "
        "incomplete pass ruling, and the play was REVERSED. (Shotgun) 3-J.Winston pass short left to 19-B.Perriman "
        "pushed ob at JAX 40 for 13 yards (37-T.Herndon).",
        PlayLabel(
            False, (C("J.Winston", "pass_yds", 13), C("B.Perriman", "rec", 1), C("B.Perriman", "rec_yds", 13))
        ),
    ),
    (
        "two-point pass succeeds",
        "ARI",
        "TWO-POINT CONVERSION ATTEMPT. 1-K.Murray pass to 19-K.Johnson is complete. ATTEMPT SUCCEEDS.",
        PlayLabel(False, (C("K.Murray", "two_pt", 1), C("K.Johnson", "two_pt", 1))),
    ),
    (
        "two-point run fails",
        "WAS",
        "TWO-POINT CONVERSION ATTEMPT. 7-D.Haskins rushes left tackle. ATTEMPT FAILS.",
        PlayLabel(False, ()),
    ),
    (
        "defensive two-point return is not an offensive conversion",
        "ATL",
        "(Pass formation) TWO-POINT CONVERSION ATTEMPT. 2-M.Ryan pass to 81-A.Hooper is incomplete. ATTEMPT FAILS. "
        "DEFENSIVE TWO-POINT ATTEMPT. 29-E.Berry intercepted the try attempt. ATTEMPT SUCCEEDS.",
        PlayLabel(False, ()),
    ),
    (
        "suffix and St. names",
        "DET",
        "(2:00) (Shotgun) 16-J.Goff pass short right to 14-A.St. Brown to DET 40 for 9 yards (22-W.Gallman Jr.).",
        PlayLabel(
            False, (C("J.Goff", "pass_yds", 9), C("A.St. Brown", "rec", 1), C("A.St. Brown", "rec_yds", 9))
        ),
    ),
    (
        "direct snap preamble",
        "TEN",
        "(5:10) Direct snap to 22-D.Henry. 22-D.Henry up the middle to 50 for 6 yards (57-B.Scarlett).",
        PlayLabel(False, (C("D.Henry", "rush_yds", 6),)),
    ),
]


@pytest.mark.parametrize("name,posteam,desc,expected", CASES, ids=[c[0] for c in CASES])
def test_parse_desc(name, posteam, desc, expected):
    got = parse_desc(desc, posteam)
    assert got.matches(expected), f"{name}: got {got.to_json()}"


def test_without_posteam_falls_back_to_stated_yards():
    desc = CASES[16][2]  # offensive holding case
    assert parse_desc(desc, None).matches(PlayLabel(False, (C("L.McCoy", "rush_yds", 6),)))


def test_predictor_emits_schema_json():
    recs = [{"desc": c[2], "posteam": c[1]} for c in CASES[:3]]
    outs = RegexPredictor().predict_batch(recs)
    assert all(PlayLabel.from_json(o) for o in outs)


def test_never_crashes_on_junk():
    for junk in ["", "Timeout #2 by NYJ at 00:38.", "END QUARTER 1", "(15:00) 9-X.Y kneels."]:
        PlayLabel.from_json(RegexPredictor().predict_batch([{"desc": junk, "posteam": None}])[0])
