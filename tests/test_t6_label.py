import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import t6_label_openai as t6  # noqa: E402

from playparse.distill.teacher import Budget, label_dataset  # noqa: E402
from playparse.ffscore.schema import Credit, PlayLabel  # noqa: E402


def _rec(i, yds):
    lab = PlayLabel(False, (Credit("J.Conner", "rush_yds", yds),))
    return {"game_id": "G", "play_id": i, "posteam": "ARI", "season": 2020, "bucket": "normal",
            "desc": f"(15:00) 6-J.Conner up the middle to ARI 33 for {yds} yards.", "label": lab.to_json()}


class FakeTeacher:
    """Sample 0 always right; sample 2 disagrees on odd plays (so R9 drops them)."""

    def __call__(self, records, n_samples):
        out = []
        for r in records:
            yds = int(r["desc"].split("for ")[1].split()[0])
            good = PlayLabel(False, (Credit("J.Conner", "rush_yds", yds),)).to_json()
            bad = PlayLabel(False, (Credit("J.Conner", "rush_yds", yds + 1),)).to_json()
            out.append([good, good, bad if r["play_id"] % 2 else good])
        from playparse.distill.teacher import TeacherBatch, Usage
        return TeacherBatch(out, Usage(calls=len(records), cost_usd=0.001 * len(records)))


def test_build_sets_r8_keeps_all_r9_drops_disagreements(tmp_path, monkeypatch):
    monkeypatch.setattr(t6, "SIZES", (4, 8))
    recs = [_rec(i, 3 + i) for i in range(8)]
    tid = "fake"
    label_dataset(recs, FakeTeacher(), tmp_path / "cache", n_samples=3, teacher_id=tid,
                  budget=Budget(max_usd=1.0, est_usd_per_call=0.001))
    rep = t6.build_sets(recs, tmp_path / "cache", tmp_path / "sets", tid, 3)
    assert rep["labeled_plays"] == 8
    assert rep["sizes"][8]["r8"] == 8 and rep["sizes"][8]["r9"] == 4
    assert rep["sizes"][8]["r8_precision"] == 1.0
    rows = [json.loads(l) for l in open(tmp_path / "sets" / "r9_n8.jsonl")]
    assert all(r["label_source"] == "teacher_r9" for r in rows) and len(rows) == 4
