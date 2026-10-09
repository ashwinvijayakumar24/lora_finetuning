"""Build the T3b (schema v2) dataset from the frozen v1 files plus the raw spots.

Usage::

    python -m playparse.data.build_dataset_v2 [--v1-dir PROCESSED] [--out-dir data/processed_v2]

Reads the v1 JSONL files (``PLAYPARSE_PROCESSED_DIR``; never modified), joins each
play to its raw nflverse row on (game_id, play_id) for `yrdln`, `defteam` and the
lateral columns (``PLAYPARSE_DATA_DIR``), and writes, for every v1 file:

    data/processed_v2/<name>.jsonl   v1 record keys + "los" + "label_v2" (+ eval_lite_part)
    data/processed_v2/manifest.json  hashes of the v1 inputs and v2 outputs, the
                                     round-trip rate per split, and how often the
                                     ground-truth spots are printed in desc

A committed copy of the manifest goes to results/t3b/dataset_v2_manifest.json.

Record keys, in order::

    {"game_id", "play_id", "season", "week", "season_type", "posteam", "desc",
     "bucket", "label", "los", "label_v2"}

``label`` is the unchanged v1 label (the harness scores against it); ``label_v2``
is `PlayLabelV2.to_json()`, the training completion for `data.schema: v2`.

Literal-spot statistics are computed on train and val only: the test split is
round-trip checked (a mechanical identity) but not otherwise analysed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from playparse import paths
from playparse.data.build_dataset import RECORD_KEYS, dumps, sha256_file
from playparse.data.ground_truth_v2 import V2_COLUMNS, build_label_v2, literal_kind
from playparse.data.load_pbp import ALL_SEASONS, pbp_path
from playparse.ffscore.schema import PlayLabel
from playparse.ffscore.schema_v2 import PlayLabelV2

V2_RECORD_KEYS = RECORD_KEYS + ("los", "label_v2")
FILES = ("train", "val", "test", "train_50k", "eval_lite")
DEV_SPLITS = ("train", "val")  # literal-spot analysis only here; test is not inspected
SOURCES = ("ground_truth_v2.py", "build_dataset_v2.py")


def builder_hash() -> str:
    h = hashlib.sha256()
    here = Path(__file__).parent
    for name in SOURCES:
        h.update((here / name).read_bytes())
    for name in ("schema_v2.py", "spots.py", "schema.py"):
        h.update((here.parent / "ffscore" / name).read_bytes())
    return h.hexdigest()


def load_raw_index(data_dir: Path | None, seasons=ALL_SEASONS) -> dict[tuple[str, int], dict]:
    """(game_id, play_id) -> the raw columns v2 needs, for every season."""
    index: dict[tuple[str, int], dict] = {}
    for season in seasons:
        df = pd.read_parquet(pbp_path(season, data_dir), columns=list(V2_COLUMNS))
        for row in df.to_dict("records"):
            index[(row["game_id"], int(row["play_id"]))] = row
    return index


class V2Stats:
    def __init__(self) -> None:
        self.plays: Counter = Counter()
        self.round_trip_ok: Counter = Counter()
        self.exceptions: dict[str, list[dict]] = defaultdict(list)
        self.with_from: Counter = Counter()
        self.los_missing: Counter = Counter()
        # per split:bucket, counts of yardage-credit `to` spots by kind, and plays
        self.to_kind: dict[str, Counter] = defaultdict(Counter)
        self.from_kind: dict[str, Counter] = defaultdict(Counter)
        self.plays_all_literal: Counter = Counter()
        self.plays_with_yardage: Counter = Counter()

    def add(self, split: str, rec: dict, ok: bool, err: str | None, lab2: PlayLabelV2 | None) -> None:
        self.plays[split] += 1
        self.round_trip_ok[split] += int(ok)
        if rec["los"] is None:
            self.los_missing[split] += 1
        if not ok:
            ex = {"key": f"{rec['game_id']}#{rec['play_id']}", "bucket": rec["bucket"], "error": err}
            if split in DEV_SPLITS:
                ex["desc"] = rec["desc"]
            self.exceptions[split].append(ex)
        if lab2 is None:
            return
        yard = [c for c in lab2.credits if c.is_yardage]
        self.with_from[split] += int(any(c.from_ is not None for c in yard))
        if split not in DEV_SPLITS or not yard:
            return
        key = f"{split}:{rec['bucket']}"
        kinds = [literal_kind(c.to, rec["desc"]) for c in yard]
        self.to_kind[key].update(kinds)
        for c in yard:
            if c.from_ is not None:
                self.from_kind[key][literal_kind(c.from_, rec["desc"])] += 1
        self.plays_with_yardage[key] += 1
        self.plays_all_literal[key] += int(all(k != "computed" for k in kinds))

    def to_dict(self) -> dict:
        rt = {s: {"plays": self.plays[s], "round_trip_ok": self.round_trip_ok[s],
                  "rate": self.round_trip_ok[s] / max(self.plays[s], 1),
                  "plays_with_from": self.with_from[s], "los_missing": self.los_missing[s],
                  "exceptions": self.exceptions[s][:50], "n_exceptions": len(self.exceptions[s])}
              for s in self.plays}
        lit = {}
        for key in sorted(self.to_kind):
            k = self.to_kind[key]
            n = sum(k.values())
            lit[key] = {
                "yardage_credits": n,
                "to_literal": k["literal"], "to_goal_line": k["goal_line"], "to_computed": k["computed"],
                "to_literal_rate": k["literal"] / n,
                "to_literal_or_goal_rate": (k["literal"] + k["goal_line"]) / n,
                "plays": self.plays_with_yardage[key],
                "plays_no_computed_spot_rate": self.plays_all_literal[key] / self.plays_with_yardage[key],
                "from_spots": dict(self.from_kind[key]),
            }
        return {"round_trip": rt, "literal_spots": lit}


def build(v1_dir: Path, out_dir: Path, data_dir: Path | None = None, log=print, files=FILES) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    log("loading raw spots ...")
    raw = load_raw_index(data_dir)
    stats = V2Stats()
    manifest_files: dict = {}
    inputs: dict = {}
    for name in files:
        src = v1_dir / f"{name}.jsonl"
        dst = out_dir / f"{name}.jsonl"
        # Subsets (train_50k, eval_lite) repeat plays from the splits; count them separately.
        split = name if name in ("train", "val", "test") else f"subset:{name}"
        inputs[src.name] = sha256_file(src)
        n = 0
        with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
            for line in fin:
                rec = json.loads(line)
                row = raw.get((rec["game_id"], int(rec["play_id"])))
                if row is None:
                    raise KeyError(f"{name}: no raw row for {rec['game_id']}#{rec['play_id']}")
                rp = row.get("posteam")
                if (None if rp is None or rp != rp else rp) != rec["posteam"]:  # NaN != NaN
                    raise ValueError(f"posteam mismatch for {rec['game_id']}#{rec['play_id']}")
                v1 = PlayLabel.from_json(rec["label"])
                res = build_label_v2({**row, "desc": rec["desc"]}, v1)
                out = {k: rec[k] for k in RECORD_KEYS}
                out["los"] = res.los
                # A play whose spots cannot be written keeps a None label_v2 and is
                # listed in the manifest (there are none in the 2015-2024 data).
                out["label_v2"] = res.label_v2.to_json() if res.label_v2 is not None else None
                if "eval_lite_part" in rec:
                    out["eval_lite_part"] = rec["eval_lite_part"]
                fout.write(dumps(out) + "\n")
                stats.add(split, {**rec, "los": res.los}, res.round_trip, res.error, res.label_v2)
                n += 1
        manifest_files[dst.name] = {"rows": n, "bytes": dst.stat().st_size, "sha256": sha256_file(dst)}
        log(f"{name}: {n:,} plays, round trip {stats.round_trip_ok[split]:,}/{stats.plays[split]:,}")
    st = stats.to_dict()
    manifest = {
        "dataset": "playparse-v2-spots (T3b)",
        "builder_hash": builder_hash(),
        "record_keys": list(V2_RECORD_KEYS),
        "v1_inputs_sha256": inputs,
        "frozen_eval_v1": {"file": "test.jsonl", "sha256": inputs.get("test.jsonl"),
                           "note": "v1 test.jsonl, read only; v2 test.jsonl carries the same plays and labels"},
        "files": manifest_files,
        **st,
        "literal_spots_note": "train and val only. to_literal: the ground-truth `to` spot is printed in desc; "
                              "to_goal_line: an unprinted goal line (touchdown/safety); to_computed: neither "
                              "(the model must compute it from the line of scrimmage and a stated gain).",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--v1-dir", type=Path, default=paths.DATA_PROCESSED)
    ap.add_argument("--out-dir", type=Path, default=paths.REPO_ROOT / "data" / "processed_v2")
    ap.add_argument("--data-dir", type=Path, default=None, help="raw parquet dir (default PLAYPARSE_DATA_DIR)")
    ap.add_argument("--copy-manifest-to", type=Path, default=paths.RESULTS / "t3b" / "dataset_v2_manifest.json")
    args = ap.parse_args(argv)
    manifest = build(args.v1_dir, args.out_dir, args.data_dir)
    if str(args.copy_manifest_to):
        args.copy_manifest_to.parent.mkdir(parents=True, exist_ok=True)
        args.copy_manifest_to.write_text(json.dumps(manifest, indent=2) + "\n")
    for s, r in manifest["round_trip"].items():
        print(f"{s:18s} round trip {r['round_trip_ok']:>8,}/{r['plays']:>8,} = {100 * r['rate']:.4f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
