"""Draw the stratified audit sample and summarize audit verdicts.

The PRD asks for a hand audit: 200 test plays, stratified by bucket, each label
checked against its `desc`. `draw_sample` picks the plays (seeded, so the sample
is reproducible); `summarize` turns per-play verdicts into per-bucket error rates.

Usage::

    python -m playparse.data.audit sample      # writes results/p1/audit_sample.jsonl
    python -m playparse.data.audit summarize   # re-summarizes results/p1/audit_results.json
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from playparse import paths
from playparse.data.buckets import PRECEDENCE

AUDIT_SEED = 20240
AUDIT_SIZE = 200
OWNER_SPOT_CHECK = 20


def draw_sample(test_path: Path, size: int = AUDIT_SIZE, seed: int = AUDIT_SEED) -> list[dict]:
    """Equal allocation per bucket (rarest buckets first get any remainder);
    a bucket with too few plays contributes all of them and the rest is spread."""
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_bucket[rec["bucket"]].append(rec)
    rng = random.Random(seed)
    quota = {b: 0 for b in PRECEDENCE}
    remaining = size
    open_buckets = [b for b in PRECEDENCE if by_bucket[b]]
    while remaining > 0 and open_buckets:
        share = max(remaining // len(open_buckets), 1)
        for b in list(open_buckets):
            take = min(share, len(by_bucket[b]) - quota[b], remaining)
            quota[b] += take
            remaining -= take
            if quota[b] >= len(by_bucket[b]):
                open_buckets.remove(b)
            if remaining == 0:
                break
    sample = []
    for b in PRECEDENCE:
        sample.extend(rng.sample(by_bucket[b], quota[b]))
    for i, rec in enumerate(sample):
        rec["audit_id"] = i
    return sample


def summarize(verdicts: list[dict]) -> dict:
    """verdicts: [{audit_id, bucket, verdict in {correct, wrong, ambiguous}, ...}]"""
    per = {b: {"n": 0, "correct": 0, "wrong": 0, "ambiguous": 0} for b in PRECEDENCE}
    for v in verdicts:
        per[v["bucket"]]["n"] += 1
        per[v["bucket"]][v["verdict"]] += 1
    for d in per.values():
        d["error_rate"] = d["wrong"] / d["n"] if d["n"] else None
    total = {k: sum(d[k] for d in per.values()) for k in ("n", "correct", "wrong", "ambiguous")}
    total["error_rate"] = total["wrong"] / total["n"] if total["n"] else None
    return {"per_bucket": per, "total": total}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out_dir = paths.RESULTS / "p1"
    if argv and argv[0] == "sample":
        sample = draw_sample(paths.DATA_PROCESSED / "test.jsonl")
        with open(out_dir / "audit_sample.jsonl", "w", encoding="utf-8") as f:
            for rec in sample:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"wrote {len(sample)} plays")
        return 0
    if argv and argv[0] == "summarize":
        verdicts = json.loads((out_dir / "audit_results.json").read_text())["plays"]
        print(json.dumps(summarize(verdicts), indent=1))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
