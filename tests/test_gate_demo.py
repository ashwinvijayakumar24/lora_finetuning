import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gate_demo  # noqa: E402

R5 = ROOT / "results" / "eval_lite" / "r5_full" / "result.json"


def _adapter(d: Path) -> str:
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text("{}")
    (d / "adapter_model.safetensors").write_bytes(b"x")
    return str(d)


@pytest.mark.skipif(not R5.is_file(), reason="committed R5 eval artifact not present")
def test_broken_candidate_is_refused_and_rollback_works(tmp_path):
    broken = json.loads(R5.read_text())
    broken["overall"]["exact_match"]["point"] = 0.80
    broken["overall"]["valid_rate"]["point"] = 0.70
    broken["buckets"]["fumble"]["exact_match"]["point"] = 0.40
    bpath = tmp_path / "broken.json"
    bpath.write_text(json.dumps(broken))
    out = gate_demo.run((_adapter(tmp_path / "a1"), str(R5)), (_adapter(tmp_path / "a2"), str(bpath)),
                        str(tmp_path / "reg"), min_exact_match=0.95)
    assert out["promote_current"]["promote"] is True
    assert out["promote_candidate"]["promote"] is False
    failed = {c["name"] for c in out["promote_candidate"]["checks"] if not c["passed"]}
    assert {"valid_rate", "overall_exact_match", "bucket_regression"} <= failed
    assert out["serving_after_candidate"] == "v0001"
    assert out["rollback"] == {"from": "v0003", "to": "v0001"}
