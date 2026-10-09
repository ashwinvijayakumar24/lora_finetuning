#!/usr/bin/env python
"""T3b local pilot (Apple M4, MPS): the P3 prompt-ablation pilot, retrained with schema v2.

A smoke signal, not the pre-registered test: same 2,400 enriched train plays, same
recipe (configs/p3_prompt_ablation.yaml), same 1,014-play pilot eval set, but the
v2 files (line of scrimmage in the prompt, field spots in the completion).

    python scripts/t3b_pilot.py data      # rebuild the P3 pilot files from their manifest, hash-check, add v2 fields
    python scripts/train.py --config configs/t3b_pilot.yaml \\
        --set data.val="\\"data/processed_v2/val.jsonl\\"" [--resume latest]
    python -m playparse.eval.run --rung lora --adapter runs/t3b_pilot/train/checkpoints/step_0000300/adapter \\
        --prompt-style minimal_v2 --schema v2 --data runs/t3b_pilot/data/eval_pilot_v2.jsonl \\
        --out runs/t3b_pilot/eval/lora_v2 --dtype bfloat16 --batch-size 16
    python -m playparse.eval.run --rung r0los --data runs/t3b_pilot/data/eval_pilot_v2.jsonl --out runs/t3b_pilot/eval/r0los
    python scripts/t3b_pilot.py report    # per-bucket table vs the v1 minimal pilot and R0, paired CIs

The pilot files are rebuilt from the key lists in results/p3_local_pilot/data_manifest.json
and must reproduce its sha256 hashes byte for byte, which proves the plays and their
order are the ones the v1 pilot used. The eval set holds 2024 plays (eval_lite);
like the P3 pilots, it is only scored, never inspected play by play.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from playparse.paths import DATA_PROCESSED  # noqa: E402

RUN = REPO / "runs" / "t3b_pilot"
DATA = RUN / "data"
RESULTS = REPO / "results" / "t3b" / "pilot"
V2_DIR = REPO / "data" / "processed_v2"
P3_MANIFEST = REPO / "results" / "p3_local_pilot" / "data_manifest.json"
BUCKETS = ["fumble", "lateral", "challenge", "penalty_stands", "two_point", "interception", "td",
           "penalty_nullified", "normal"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")  # the P3 pilot's serialization


def key(r: dict) -> str:
    return f"{r['game_id']}#{r['play_id']}"


def cmd_data(args: argparse.Namespace) -> None:
    man = json.loads(P3_MANIFEST.read_text())
    sources = {"train_pilot.jsonl": "train.jsonl", "eval_pilot.jsonl": "eval_lite.jsonl"}
    out = {"purpose": "T3b local pilot: the P3 pilot plays with schema-v2 fields", "files": {}}
    for name, src in sources.items():
        v1 = {key(r): r for r in read_jsonl(Path(args.processed) / src)}
        v2 = {key(r): r for r in read_jsonl(V2_DIR / src)}
        keys = man["files"][name]["keys"]
        v1_recs = [v1[k] for k in keys]
        tmp = DATA / name
        write_jsonl(tmp, v1_recs)
        got = sha256(tmp)
        if got != man["files"][name]["sha256"]:
            raise SystemExit(f"{name}: rebuilt sha256 {got[:12]} != P3 manifest {man['files'][name]['sha256'][:12]}")
        v2_recs = []
        for r in v1_recs:
            w = v2[key(r)]
            assert w["label"] == r["label"] and w["desc"] == r["desc"]
            v2_recs.append({**r, "los": w["los"], "label_v2": w["label_v2"]})
        v2_name = name.replace(".jsonl", "_v2.jsonl")
        write_jsonl(DATA / v2_name, v2_recs)
        out["files"][name] = {"n": len(v1_recs), "sha256": got, "matches_p3_manifest": True}
        out["files"][v2_name] = {"n": len(v2_recs), "sha256": sha256(DATA / v2_name)}
        print(f"{name}: {len(v1_recs)} plays, sha256 matches the P3 pilot; wrote {v2_name}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "data_manifest.json").write_text(json.dumps(out, indent=1) + "\n")


def stat_table(pred_path: Path, records: list[dict]):
    from playparse.eval.metrics import build_stat_table, score_example
    from playparse.ffscore.schema import PlayLabel

    with open(pred_path, encoding="utf-8") as f:
        preds = {json.loads(l)["key"]: json.loads(l) for l in f if l.strip()}
    keys = [key(r) for r in records]
    golds = [PlayLabel.from_json(r["label"]) for r in records]
    scores = [score_example(g, preds[k]["raw"]) for g, k in zip(golds, keys)]
    table = build_stat_table([str(r["game_id"]) for r in records], [r["bucket"] for r in records], golds, scores)
    return table, scores


def cmd_report(args: argparse.Namespace) -> None:
    from playparse.eval.bootstrap import paired_diff_ci
    from playparse.eval.metrics import metrics_from_sums

    records = read_jsonl(DATA / "eval_pilot_v2.jsonl")
    arms = {
        "v2_spots": RUN / "eval" / "lora_v2",
        "v1_minimal": REPO / "results" / "p3_prompt_ablation" / "eval" / "lora_minimal",
        "r0": REPO / "results" / "p3_local_pilot" / "eval" / "r0",
        "r0los": RUN / "eval" / "r0los",
    }
    tables, scores, results = {}, {}, {}
    for arm, d in arms.items():
        tables[arm], scores[arm] = stat_table(d / "predictions.jsonl", records)
        results[arm] = json.loads((d / "result.json").read_text())
        got = tables[arm].overall[:, 3].sum() / len(records)
        assert abs(got - results[arm]["overall"]["exact_match"]["point"]) < 1e-12, arm

    def paired(a: str, b: str) -> dict:
        ta, tb = tables[a], tables[b]
        out = {"OVERALL": paired_diff_ci(ta.overall, tb.overall, metrics_from_sums, n_boot=args.n_boot,
                                         seed=args.seed)["exact_match"]}
        for bk in BUCKETS:
            out[bk] = paired_diff_ci(ta.by_bucket[bk], tb.by_bucket[bk], metrics_from_sums, n_boot=args.n_boot,
                                     seed=args.seed)["exact_match"]
        return out

    summary = {"n_eval": len(records), "n_boot": args.n_boot, "seed": args.seed,
               "note": "smoke signal on the P3 pilot scale; not the pre-registered T3b test",
               "exact_match": {}, "valid_rate": {}, "paired_exact_match": {}, "configs": {}}
    for arm in arms:
        res = results[arm]
        summary["exact_match"][arm] = {"OVERALL": res["overall"]["exact_match"],
                                       **{b: res["buckets"][b]["exact_match"] for b in BUCKETS}}
        summary["valid_rate"][arm] = res["overall"]["valid_rate"]["point"]
        summary["configs"][arm] = {"dir": str(arms[arm].relative_to(REPO)), "config_hash": res["meta"]["config_hash"]}
    for a, b in [("v2_spots", "v1_minimal"), ("v2_spots", "r0"), ("v2_spots", "r0los"), ("r0los", "r0")]:
        summary["paired_exact_match"][f"{a}-minus-{b}"] = paired(a, b)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for arm in ("v2_spots", "r0los"):
        dst = RESULTS / "eval" / arms[arm].name
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("result.json", "predictions.jsonl"):
            shutil.copy2(arms[arm] / f, dst / f)
    for src in ("run_spec.json", "run_meta.json", "metrics.jsonl", "result.json"):
        p = RUN / "train" / src
        if p.exists():
            (RESULTS / "train").mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, RESULTS / "train" / src)

    def pct(ci: dict) -> str:
        return f"{100 * ci['point']:5.1f}"

    print(f"{'bucket':18s} {'n':>5} " + " ".join(f"{a:>10s}" for a in arms) + "   v2-v1 [95% CI]")
    for b in ["OVERALL"] + BUCKETS:
        n = len(records) if b == "OVERALL" else sum(r["bucket"] == b for r in records)
        d = summary["paired_exact_match"]["v2_spots-minus-v1_minimal"][b]
        print(f"{b:18s} {n:>5} " + " ".join(f"{pct(summary['exact_match'][a][b]):>10s}" for a in arms)
              + f"   {100 * d['point']:+5.1f} [{100 * d['lo']:+5.1f}, {100 * d['hi']:+5.1f}]")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--processed", default=str(DATA_PROCESSED))
    r = sub.add_parser("report")
    r.add_argument("--n-boot", type=int, default=2000)
    r.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    {"data": cmd_data, "report": cmd_report}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
