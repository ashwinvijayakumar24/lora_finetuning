import json
from collections import Counter

from playparse.data.audit import draw_sample, summarize


def test_draw_sample_stratified_and_seeded(tmp_path):
    path = tmp_path / "test.jsonl"
    rows = []
    for i in range(300):
        bucket = "normal" if i < 250 else ("td" if i < 290 else "lateral")
        rows.append({"game_id": "g", "play_id": i, "bucket": bucket})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    s = draw_sample(path, size=30, seed=1)
    counts = Counter(r["bucket"] for r in s)
    assert len(s) == 30
    assert counts["lateral"] == 10 and counts["td"] == 10 and counts["normal"] == 10
    assert [r["audit_id"] for r in s] == list(range(30))
    assert s == draw_sample(path, size=30, seed=1)

    # a bucket with too few plays gives all it has; the rest is spread over the others
    s2 = draw_sample(path, size=60, seed=1)
    c2 = Counter(r["bucket"] for r in s2)
    assert c2["lateral"] == 10 and len(s2) == 60


def test_summarize():
    v = [{"bucket": "lateral", "verdict": "wrong"}, {"bucket": "lateral", "verdict": "correct"},
         {"bucket": "td", "verdict": "ambiguous"}]
    s = summarize(v)
    assert s["per_bucket"]["lateral"]["error_rate"] == 0.5
    assert s["per_bucket"]["td"]["error_rate"] == 0.0
    assert s["per_bucket"]["normal"]["error_rate"] is None
    assert s["total"]["n"] == 3 and s["total"]["wrong"] == 1
