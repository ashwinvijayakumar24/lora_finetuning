"""The P3 prompt ablation's rescoring and paired comparison (no model needed)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from playparse.eval.harness import load_records

REPO = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).parent / "fixtures" / "eval_mini.jsonl"


@pytest.fixture(scope="module")
def abl():
    spec = importlib.util.spec_from_file_location("p3_prompt_ablation", REPO / "scripts" / "p3_prompt_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_preds(path: Path, records: list[dict], wrong: set[int]) -> None:
    with open(path, "w") as f:
        for i, r in enumerate(records):
            raw = '{"nullified":false,"credits":[]}' if i in wrong else r["label"]
            f.write(json.dumps({"key": f"{r['game_id']}#{r['play_id']}", "raw": raw}) + "\n")


def test_rescore_and_paired_diff(abl, tmp_path):
    records = load_records(FIXTURE)
    # Make the gold of record 0 non-empty so the "wrong" prediction really is wrong.
    assert json.loads(records[0]["label"])["credits"]
    _write_preds(tmp_path / "a.jsonl", records, wrong=set())
    _write_preds(tmp_path / "b.jsonl", records, wrong={0})
    ta, sa, _ = abl.stat_table(tmp_path / "a.jsonl", records)
    tb, sb, _ = abl.stat_table(tmp_path / "b.jsonl", records)
    assert all(s.exact for s in sa) and not sb[0].exact
    same = abl.paired(ta, ta, n_boot=50, seed=0)
    assert same["OVERALL"]["exact_match"]["point"] == 0.0
    d = abl.paired(ta, tb, n_boot=200, seed=0)
    assert d["OVERALL"]["exact_match"]["point"] == pytest.approx(1 / len(records))
    assert d["OVERALL"]["exact_match"]["lo"] >= 0.0


def test_rescore_rejects_missing_plays(abl, tmp_path):
    records = load_records(FIXTURE)
    _write_preds(tmp_path / "a.jsonl", records[1:], wrong=set())
    with pytest.raises(SystemExit):
        abl.stat_table(tmp_path / "a.jsonl", records)
