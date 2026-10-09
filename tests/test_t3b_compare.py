"""scripts/t3b_compare.py: the pre-registered T3b verdict, applied mechanically."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("t3b_compare", REPO / "scripts" / "t3b_compare.py")
cmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp)

GOOD = '{"nullified":false,"credits":[{"player":"A","stat":"rush_yds","value":5}]}'
BAD = '{"nullified":false,"credits":[{"player":"A","stat":"rush_yds","value":4}]}'


def _records(n_games: int) -> list[dict]:
    return [{"game_id": f"g{i}", "play_id": 1, "bucket": "fumble" if i % 2 else "normal", "label": GOOD}
            for i in range(n_games)]


def _arm(tmp: Path, name: str, records: list[dict], wrong: set[int]) -> Path:
    d = tmp / name
    d.mkdir()
    with open(d / "predictions.jsonl", "w") as f:
        for i, r in enumerate(records):
            f.write(json.dumps({"key": f"{r['game_id']}#1", "raw": BAD if i in wrong else GOOD}) + "\n")
    return d


def test_verdict(tmp_path):
    recs = _records(40)
    best = _arm(tmp_path, "best", recs, set())
    worse = _arm(tmp_path, "worse", recs, set(range(0, 40, 2)))  # wrong on every normal play
    res = cmp.compare(recs, {"t3b": best, "r0": worse}, n_boot=300)
    assert res["verdict"]["earned"] and res["exact_match"]["r0"]["OVERALL"] == 0.5
    # the other way round: not earned, and the regressing bucket is named
    res = cmp.compare(recs, {"t3b": worse, "r0": best}, n_boot=300)
    assert not res["verdict"]["earned"]
    assert any(r.startswith("normal vs r0: regresses") for r in res["verdict"]["reasons"])
    # a tie is not a win
    res = cmp.compare(recs, {"t3b": best, "r0": _arm(tmp_path, "same", recs, set())}, n_boot=300)
    assert not res["verdict"]["earned"]
