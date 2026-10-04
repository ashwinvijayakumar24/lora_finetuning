"""Template-generated plays for smoke tests, in the real dataset record format.

These are not training data for the real model. They exist so the loop can be
exercised end to end on the real 1B model before the P1 dataset exists, and so
tests have realistic-looking prompts. Labels follow the PRD scoring rules for the
handful of play shapes generated here (passes, runs, sacks, touchdowns, an
interception, and a nullified play).
"""
from __future__ import annotations

import random

from playparse.ffscore.schema import Credit, PlayLabel

TEAMS = {
    "PHI": (["J.Hurts"], ["A.Brown", "D.Smith", "D.Goedert"], ["S.Barkley", "K.Gainwell"]),
    "KC": (["P.Mahomes"], ["T.Kelce", "R.Rice", "X.Worthy"], ["I.Pacheco", "K.Hunt"]),
    "BUF": (["J.Allen"], ["K.Coleman", "D.Kincaid", "K.Shakir"], ["J.Cook", "R.Davis"]),
    "DET": (["J.Goff"], ["A.St. Brown", "J.Williams", "S.LaPorta"], ["J.Gibbs", "D.Montgomery"]),
}
DIRS = ["short left", "short middle", "short right", "deep left", "deep middle", "deep right"]
RUN_DIRS = ["left end", "left tackle", "up the middle", "right guard", "right end"]
DEFENDERS = ["J.Smith", "M.Jones", "T.Watt", "N.Bosa", "F.Warner", "R.Smith"]


def _num(rng: random.Random) -> int:
    return rng.randint(1, 99)


def _clock(rng: random.Random) -> str:
    return f"({rng.randint(0, 14)}:{rng.randint(0, 59):02d})"


def synth_play(rng: random.Random) -> tuple[str, str, PlayLabel, str]:
    """Return (posteam, desc, label, bucket) for one random play."""
    team = rng.choice(sorted(TEAMS))
    qbs, wrs, rbs = TEAMS[team]
    qb, wr, rb = rng.choice(qbs), rng.choice(wrs), rng.choice(rbs)
    d = rng.choice(DEFENDERS)
    kind = rng.choices(["pass", "incomplete", "run", "pass_td", "run_td", "sack", "int", "noplay"],
                       weights=[30, 10, 25, 8, 6, 8, 5, 8])[0]
    c = _clock(rng)
    if kind == "pass":
        y = rng.randint(1, 45)
        desc = (f"{c} {_num(rng)}-{qb} pass {rng.choice(DIRS)} to {_num(rng)}-{wr} to {team} "
                f"{rng.randint(10, 50)} for {y} yards ({_num(rng)}-{d}).")
        label = PlayLabel(False, (Credit(qb, "pass_yds", y), Credit(wr, "rec", 1), Credit(wr, "rec_yds", y)))
        return team, desc, label, "pass"
    if kind == "incomplete":
        desc = f"{c} {_num(rng)}-{qb} pass incomplete {rng.choice(DIRS)} to {_num(rng)}-{wr}."
        return team, desc, PlayLabel(False, ()), "pass"
    if kind == "run":
        y = rng.randint(1, 30)
        desc = (f"{c} {_num(rng)}-{rb} {rng.choice(RUN_DIRS)} to {team} {rng.randint(10, 50)} "
                f"for {y} yards ({_num(rng)}-{d}).")
        return team, desc, PlayLabel(False, (Credit(rb, "rush_yds", y),)), "run"
    if kind == "pass_td":
        y = rng.randint(1, 60)
        desc = f"{c} {_num(rng)}-{qb} pass {rng.choice(DIRS)} to {_num(rng)}-{wr} for {y} yards, TOUCHDOWN."
        label = PlayLabel(False, (Credit(qb, "pass_yds", y), Credit(qb, "pass_td", 1), Credit(wr, "rec", 1),
                                  Credit(wr, "rec_yds", y), Credit(wr, "rec_td", 1)))
        return team, desc, label, "td"
    if kind == "run_td":
        y = rng.randint(1, 40)
        desc = f"{c} {_num(rng)}-{rb} {rng.choice(RUN_DIRS)} for {y} yards, TOUCHDOWN."
        return team, desc, PlayLabel(False, (Credit(rb, "rush_yds", y), Credit(rb, "rush_td", 1))), "td"
    if kind == "sack":
        y = rng.randint(1, 12)
        desc = f"{c} {_num(rng)}-{qb} sacked at {team} {rng.randint(10, 50)} for -{y} yards ({_num(rng)}-{d})."
        return team, desc, PlayLabel(False, ()), "sack"
    if kind == "int":
        desc = (f"{c} {_num(rng)}-{qb} pass {rng.choice(DIRS)} intended for {_num(rng)}-{wr} INTERCEPTED by "
                f"{_num(rng)}-{d} at {team} {rng.randint(10, 50)}.")
        return team, desc, PlayLabel(False, (Credit(qb, "int", 1),)), "turnover"
    y = rng.randint(3, 30)
    desc = (f"{c} {_num(rng)}-{qb} pass {rng.choice(DIRS)} to {_num(rng)}-{wr} for {y} yards. PENALTY on "
            f"{team}-{_num(rng)}-L.Johnson, Offensive Holding, 10 yards, enforced at {team} "
            f"{rng.randint(10, 50)} - No Play.")
    return team, desc, PlayLabel(True, ()), "nullified"


def synth_records(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        posteam, desc, label, bucket = synth_play(rng)
        out.append({
            "game_id": f"SYNTH_{seed}", "play_id": i, "season": 2024, "week": 1, "season_type": "REG",
            "posteam": posteam, "desc": desc, "bucket": bucket, "label": label.to_json(),
        })
    return out
