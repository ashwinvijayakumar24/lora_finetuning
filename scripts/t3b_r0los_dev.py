#!/usr/bin/env python
"""Development loop for the T3b fairness arm "R0 + LOS", on train and val only.

    python scripts/t3b_r0los_dev.py [--split val] [--bucket fumble] [--show 20] [--rung r0los]

Scores R0 and R0+LOS against the frozen labels in data/processed_v2/<split>.jsonl
(the v1 labels plus the line of scrimmage), prints exact match per bucket for both,
and shows R0+LOS's misses in one bucket. The 2024 test split is refused: the arm
is developed on train and val, and scored on test once, with the other rungs.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playparse.eval.baselines.regex_los import parse_desc_los  # noqa: E402
from playparse.eval.baselines.regex_parser import parse_desc  # noqa: E402
from playparse.ffscore.schema import PlayLabel  # noqa: E402
from playparse.paths import REPO_ROOT  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["train", "val", "train_50k"])
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "processed_v2")
    ap.add_argument("--bucket", default=None, help="show R0+LOS misses in this bucket")
    ap.add_argument("--show", type=int, default=10)
    ap.add_argument("--regressions", action="store_true", help="only show plays R0 gets right and R0+LOS wrong")
    ap.add_argument("--out", type=Path, default=None, help="write the per-bucket table as JSON")
    args = ap.parse_args(argv)

    n, ok0, ok1 = Counter(), Counter(), Counter()
    misses = []
    with open(args.data_dir / f"{args.split}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gold = PlayLabel.from_json(r["label"])
            p0 = parse_desc(r["desc"], r["posteam"])
            p1 = parse_desc_los(r["desc"], r["posteam"], r["los"])
            b = r["bucket"]
            n[b] += 1
            ok0[b] += p0.matches(gold)
            ok1[b] += p1.matches(gold)
            if not p1.matches(gold) and (args.bucket in (None, b)) and not (args.regressions and not p0.matches(gold)):
                misses.append((b, r["los"], r["posteam"], r["desc"], gold.canonical().to_json(),
                               p1.canonical().to_json(), p0.matches(gold)))
    table = {}
    print(f"{'bucket':18s} {'n':>7} {'R0':>7} {'R0+LOS':>7}")
    for b in sorted(n, key=lambda k: -n[k]) + ["OVERALL"]:
        nn = sum(n.values()) if b == "OVERALL" else n[b]
        a0 = sum(ok0.values()) if b == "OVERALL" else ok0[b]
        a1 = sum(ok1.values()) if b == "OVERALL" else ok1[b]
        table[b] = {"n": nn, "r0": a0 / nn, "r0los": a1 / nn}
        print(f"{b:18s} {nn:>7} {100 * a0 / nn:7.2f} {100 * a1 / nn:7.2f}")
    for m in misses[: args.show]:
        print(f"\n[{m[0]}] los={m[1]} posteam={m[2]} (R0 {'right' if m[6] else 'wrong'})\n  {m[3]}\n  gold {m[4]}\n  pred {m[5]}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"split": args.split, "buckets": table}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
