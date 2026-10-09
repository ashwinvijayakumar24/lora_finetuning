#!/usr/bin/env python
"""Paired comparison for the pre-registered T3b test (PRD section 17).

    python scripts/t3b_compare.py --data data/processed_v2/test.jsonl \\
        --arm t3b=results/t3b_test --arm r0=results/r0_test --arm r0los=results/r0los_test \\
        [--arm r5=results/r5_test] --out results/t3b/test_comparison.json

Each --arm NAME=DIR points at an eval-harness output directory (predictions.jsonl +
result.json) scored on the same plays. Every arm is rescored from its raw outputs
with the unchanged v1 scorer (a schema-v2 arm's `raw` is already the converted v1
text), and the first arm is compared with each other arm by a paired, game-clustered
bootstrap of exact match, overall and per bucket.

The verdict applies the pre-registered rule mechanically:

* earned: for every baseline arm, the overall difference's 95% CI is above 0, and
  no bucket's difference has its whole 95% CI below 0 ("no bucket regresses beyond
  its CI");
* otherwise: not earned. The per-bucket table is reported either way.

A v2 data file and a v1 data file with the same plays give the same verdict: only
`label`, `bucket`, `game_id` and `play_id` are read.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from playparse.eval.bootstrap import paired_diff_ci  # noqa: E402
from playparse.eval.harness import file_sha256, load_records  # noqa: E402
from playparse.eval.metrics import build_stat_table, metrics_from_sums, score_example  # noqa: E402
from playparse.ffscore.schema import PlayLabel  # noqa: E402


def rescore(pred_dir: Path, records: list[dict]):
    with open(pred_dir / "predictions.jsonl", encoding="utf-8") as f:
        preds = {json.loads(l)["key"]: json.loads(l) for l in f if l.strip()}
    keys = [f"{r['game_id']}#{r['play_id']}" for r in records]
    missing = [k for k in keys if k not in preds]
    if missing:
        raise SystemExit(f"{pred_dir} lacks {len(missing)} plays (first {missing[0]})")
    golds = [PlayLabel.from_json(r["label"]) for r in records]
    scores = [score_example(g, preds[k]["raw"]) for g, k in zip(golds, keys)]
    return build_stat_table([str(r["game_id"]) for r in records], [r["bucket"] for r in records], golds, scores)


def compare(records: list[dict], arms: dict[str, Path], n_boot: int = 2000, seed: int = 0) -> dict:
    names = list(arms)
    tables = {n: rescore(d, records) for n, d in arms.items()}
    buckets = sorted({r["bucket"] for r in records})
    out = {"n": len(records), "n_games": len({r["game_id"] for r in records}), "n_boot": n_boot, "seed": seed,
           "exact_match": {}, "paired": {}, "verdict": {}}
    for n, t in tables.items():
        em = {"OVERALL": t.overall[:, 3].sum() / t.overall[:, 0].sum()}
        for b in buckets:
            em[b] = t.by_bucket[b][:, 3].sum() / t.by_bucket[b][:, 0].sum()
        out["exact_match"][n] = em
    target = names[0]
    earned = True
    reasons = []
    for base in names[1:]:
        ta, tb = tables[target], tables[base]
        d = {"OVERALL": paired_diff_ci(ta.overall, tb.overall, metrics_from_sums, n_boot=n_boot,
                                       seed=seed)["exact_match"]}
        for b in buckets:
            d[b] = paired_diff_ci(ta.by_bucket[b], tb.by_bucket[b], metrics_from_sums, n_boot=n_boot,
                                  seed=seed)["exact_match"]
        out["paired"][f"{target}-minus-{base}"] = d
        if not d["OVERALL"]["lo"] > 0:
            earned = False
            reasons.append(f"overall vs {base}: CI [{d['OVERALL']['lo']:+.4f}, {d['OVERALL']['hi']:+.4f}] includes 0 "
                           "or is below it")
        for b in buckets:
            if d[b]["hi"] < 0:
                earned = False
                reasons.append(f"{b} vs {base}: regresses, CI [{d[b]['lo']:+.4f}, {d[b]['hi']:+.4f}]")
    out["verdict"] = {"target": target, "baselines": names[1:], "earned": earned, "reasons": reasons}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=DIR",
                    help="first arm is the one tested; the rest are baselines")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    arms = dict(a.split("=", 1) for a in args.arm)
    records = load_records(args.data)
    res = compare(records, {k: Path(v) for k, v in arms.items()}, args.n_boot, args.seed)
    res["data"] = {"path": args.data, "sha256": file_sha256(args.data)}
    names = list(arms)
    print(f"{'bucket':18s} " + " ".join(f"{n:>9s}" for n in names))
    for b in res["exact_match"][names[0]]:
        print(f"{b:18s} " + " ".join(f"{100 * res['exact_match'][n][b]:9.2f}" for n in names))
    for key, d in res["paired"].items():
        o = d["OVERALL"]
        print(f"{key}: {100 * o['point']:+.2f} [{100 * o['lo']:+.2f}, {100 * o['hi']:+.2f}]")
    print("T3b earned" if res["verdict"]["earned"] else "T3b not earned: " + "; ".join(res["verdict"]["reasons"]))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
