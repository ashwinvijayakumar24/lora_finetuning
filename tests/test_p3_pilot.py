"""The P3 local pilot's data selection and paired comparison (no model needed)."""
from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pilot():
    spec = importlib.util.spec_from_file_location("p3_pilot", REPO / "scripts" / "p3_pilot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _train_pool():
    sizes = {"normal": 3000, "lateral": 7, "fumble": 400, "challenge": 300, "penalty_stands": 300,
             "two_point": 50, "interception": 300, "td": 300, "penalty_nullified": 500}
    recs = []
    for b, k in sizes.items():
        recs += [{"game_id": f"G{i % 50}", "play_id": len(recs) + i, "bucket": b} for i in range(k)]
    return recs


def test_enriched_train_takes_all_laterals_and_balances(pilot):
    pool = _train_pool()
    out = pilot.enriched_train(pool, n_hard=400, seed=1)
    c = Counter(r["bucket"] for r in out)
    assert c["lateral"] == 7  # every lateral
    assert c["two_point"] == 50  # capped by availability ...
    others = [c[b] for b in ("fumble", "challenge", "penalty_stands", "interception", "td", "penalty_nullified")]
    assert max(others) - min(others) <= 1  # ... and the shortfall is spread evenly over the rest
    assert sum(c[b] for b in pilot.HARD) == 400
    assert c["normal"] == 400  # half normal, half hard
    assert len({r["play_id"] for r in out}) == len(out)
    assert out == pilot.enriched_train(pool, n_hard=400, seed=1)
    assert out != pilot.enriched_train(pool, n_hard=400, seed=2)


def _write_preds(d: Path, exact: dict[str, int], buckets: dict[str, str], games: dict[str, str]):
    d.mkdir(parents=True)
    with open(d / "predictions.jsonl", "w") as f:
        for k, e in exact.items():
            f.write(json.dumps({"key": k, "exact": bool(e), "bucket": buckets[k], "game_id": games[k]}) + "\n")


def test_paired_bucket_diffs(pilot, tmp_path):
    keys = [f"k{i}" for i in range(6)]
    buckets = dict(zip(keys, ["fumble"] * 4 + ["normal"] * 2))
    games = dict(zip(keys, ["A", "A", "B", "C", "A", "B"]))
    _write_preds(tmp_path / "a", dict(zip(keys, [1, 1, 1, 0, 1, 1])), buckets, games)
    _write_preds(tmp_path / "b", dict(zip(keys, [0, 1, 0, 0, 1, 1])), buckets, games)
    d = pilot.paired_bucket_diffs(tmp_path / "a", tmp_path / "b", n_boot=200)
    assert d["fumble"]["diff"] == pytest.approx(2 / 4)
    assert d["normal"]["diff"] == 0 and d["normal"]["lo"] == 0 and d["normal"]["hi"] == 0
    assert d["OVERALL"]["n"] == 6 and d["OVERALL"]["diff"] == pytest.approx(2 / 6)
    assert d["fumble"]["lo"] <= d["fumble"]["diff"] <= d["fumble"]["hi"]
    assert d["fumble"]["n_games"] == 3
