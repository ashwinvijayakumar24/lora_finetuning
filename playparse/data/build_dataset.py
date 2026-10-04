"""Build the frozen PlayParse dataset: label, bucket, split, and hash every play.

Usage::

    python -m playparse.data.build_dataset [--data-dir RAW] [--out-dir PROCESSED]

Writes to ``PLAYPARSE_PROCESSED_DIR`` (default ``data/processed``):

* ``train.jsonl`` (2015-2022), ``val.jsonl`` (2023), ``test.jsonl`` (2024)
* ``train_50k.jsonl`` - the seeded sweep subset of train
* ``eval_lite.jsonl`` - ~3k test plays for local runs (see `splits.eval_lite`)
* ``manifest.json`` - row counts per split and bucket, label checks, and the
  sha256 of every file. ``test.jsonl``'s hash is the frozen-eval identity.

Every line is one play, keys in this fixed order::

    {"game_id", "play_id", "season", "week", "season_type", "posteam", "desc",
     "bucket", "label"}

where ``label`` is the `PlayLabel.to_json()` string (eval_lite lines also carry
``eval_lite_part``). Lines are sorted by (season, game_id, play_id) and serialized
deterministically, so rebuilding from the same raw files reproduces the same hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from playparse import paths
from playparse.data.buckets import PRECEDENCE, assign_bucket
from playparse.data.ground_truth import build_label
from playparse.data.load_pbp import ALL_SEASONS, load_pbp_season
from playparse.data.splits import (
    EVAL_LITE_CORE_PLAYS,
    EVAL_LITE_MIN_PER_BUCKET,
    EVAL_LITE_SEED,
    SPLIT_SEASONS,
    TRAIN_SUBSET_SEED,
    TRAIN_SUBSET_SIZE,
    eval_lite,
    sample_indices,
    split_of,
)
from playparse.ffscore.schema import PlayLabel

RECORD_KEYS = ("game_id", "play_id", "season", "week", "season_type", "posteam", "desc", "bucket", "label")
BUILDER_SOURCES = ("ground_truth.py", "buckets.py", "splits.py", "load_pbp.py", "build_dataset.py")

_GAIN = re.compile(r"\bfor (-?\d+) yards?\b|\bfor no gain\b")


def desc_gain(desc: str) -> int | None:
    """Yards in the final ruling's first "for N yards" phrase, or None if absent."""
    final = re.split(r"play was REVERSED\.", desc)[-1]
    m = _GAIN.search(final)
    if not m:
        return None
    return int(m.group(1)) if m.group(1) is not None else 0


def _clean(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float):
        if v != v:  # NaN
            return None
        if v.is_integer():
            return int(v)
    if hasattr(v, "item"):
        return _clean(v.item())
    return v


def make_record(row: dict) -> tuple[dict, Any]:
    res = build_label(row)
    rec = {
        "game_id": row["game_id"],
        "play_id": int(row["play_id"]),
        "season": int(row["season"]),
        "week": int(row["week"]),
        "season_type": row["season_type"],
        "posteam": _clean(row.get("posteam")),
        "desc": row["desc"],
        "bucket": assign_bucket(row),
        "label": res.label.to_json(),
    }
    return rec, res


def dumps(rec: dict) -> str:
    return json.dumps(rec, ensure_ascii=False, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def builder_hash() -> str:
    h = hashlib.sha256()
    here = Path(__file__).parent
    for name in BUILDER_SOURCES:
        h.update((here / name).read_bytes())
    for name in ("schema.py", "scorer.py"):
        h.update((here.parent / "ffscore" / name).read_bytes())
    return h.hexdigest()


class Stats:
    """Label-quality checks accumulated during the build, reported in the manifest."""

    def __init__(self) -> None:
        self.buckets: dict[str, Counter] = defaultdict(Counter)
        self.plays: Counter = Counter()
        self.nullified: Counter = Counter()
        self.credits: Counter = Counter()
        self.credits_name_in_desc: Counter = Counter()
        self.dropped: Counter = Counter()
        self.noisy: Counter = Counter()
        self.season_type: dict[str, Counter] = defaultdict(Counter)
        # text-vs-label yardage check on single-ball-carrier plays, per bucket
        self.gain_checked: Counter = Counter()
        self.gain_mismatch: Counter = Counter()
        self.gain_examples: list[dict] = []

    def add(self, split: str, rec: dict, row: dict, res: Any) -> None:
        b = rec["bucket"]
        self.buckets[split][b] += 1
        self.plays[split] += 1
        self.season_type[split][rec["season_type"]] += 1
        self.nullified[split] += int(res.label.nullified)
        n = len(res.label.credits)
        self.credits[split] += n
        self.credits_name_in_desc[split] += n - sum(
            1 for c in res.label.credits if c.player in res.names_missing_from_desc
        )
        self.dropped[split] += len(res.dropped)
        self.noisy[split] += int(res.noisy)

        # Does the text's "for N yards" match the credited yards? Only checked where
        # one player carries the ball and no lateral is involved.
        if res.label.nullified or b == "lateral" or row.get("two_point_attempt") == 1:
            return
        primary = None
        if row.get("play_type") == "run" and row.get("rushing_yards") == row.get("rushing_yards"):
            primary = row.get("rushing_yards")
        elif row.get("complete_pass") == 1 and row.get("receiving_yards") == row.get("receiving_yards"):
            primary = row.get("receiving_yards")
        if primary is None:
            return
        gain = desc_gain(row["desc"])
        if gain is None:
            return
        key = f"{split}:{b}"
        self.gain_checked[key] += 1
        if int(primary) != gain:
            self.gain_mismatch[key] += 1
            if len(self.gain_examples) < 40 and split == "test":
                self.gain_examples.append({"bucket": b, "desc": row["desc"], "label": rec["label"]})

    def to_dict(self) -> dict:
        out: dict = {}
        for split in self.plays:
            out[split] = {
                "plays": self.plays[split],
                "by_bucket": {b: self.buckets[split][b] for b in PRECEDENCE},
                "by_season_type": dict(self.season_type[split]),
                "nullified": self.nullified[split],
                "credits": self.credits[split],
                "name_in_desc_rate": self.credits_name_in_desc[split] / max(self.credits[split], 1),
                "dropped_credits": self.dropped[split],
                "noisy_plays": self.noisy[split],
            }
        gain = {}
        for key in sorted(self.gain_checked):
            gain[key] = {
                "checked": self.gain_checked[key],
                "text_disagrees": self.gain_mismatch[key],
                "rate": self.gain_mismatch[key] / self.gain_checked[key],
            }
        out["text_yards_check"] = gain
        return out


def build(data_dir: Path | None, out_dir: Path, seasons=ALL_SEASONS, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = Stats()
    files = {name: open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") for name in SPLIT_SEASONS}
    test_records: list[dict] = []
    try:
        for season in seasons:
            split = split_of(season)
            df = load_pbp_season(season, data_dir)
            df = df.sort_values(["game_id", "play_id"], kind="stable")
            rows = df.to_dict("records")
            for row in rows:
                rec, res = make_record(row)
                # every label must survive the schema's own validator
                assert PlayLabel.from_json(rec["label"]) == res.label.canonical()
                files[split].write(dumps(rec) + "\n")
                stats.add(split, rec, row, res)
                if split == "test":
                    test_records.append(rec)
            log(f"{season}: {len(rows):,} plays -> {split}")
            del df, rows
    finally:
        for f in files.values():
            f.close()

    # train_50k: second streaming pass over train.jsonl
    n_train = stats.plays["train"]
    keep = set(sample_indices(n_train, TRAIN_SUBSET_SIZE, TRAIN_SUBSET_SEED))
    sub_buckets: Counter = Counter()
    with open(out_dir / "train.jsonl", encoding="utf-8") as src, open(
        out_dir / "train_50k.jsonl", "w", encoding="utf-8"
    ) as dst:
        for i, line in enumerate(src):
            if i in keep:
                dst.write(line)
                sub_buckets[json.loads(line)["bucket"]] += 1

    lite = eval_lite(test_records)
    with open(out_dir / "eval_lite.jsonl", "w", encoding="utf-8") as f:
        for r in lite:
            f.write(dumps({**{k: r[k] for k in RECORD_KEYS}, "eval_lite_part": r["eval_lite_part"]}) + "\n")
    lite_buckets = Counter(r["bucket"] for r in lite)
    lite_parts = Counter(r["eval_lite_part"] for r in lite)
    lite_game_buckets = Counter(r["bucket"] for r in lite if r["eval_lite_part"] == "game")

    st = stats.to_dict()
    manifest = {
        "dataset": "playparse-v1",
        "builder_hash": builder_hash(),
        "seasons": {k: list(v) for k, v in SPLIT_SEASONS.items()},
        "record_keys": list(RECORD_KEYS),
        "bucket_precedence": list(PRECEDENCE),
        "seeds": {"train_50k": TRAIN_SUBSET_SEED, "eval_lite": EVAL_LITE_SEED},
        "eval_lite_params": {"core_plays": EVAL_LITE_CORE_PLAYS, "min_per_bucket": EVAL_LITE_MIN_PER_BUCKET},
        "splits": {k: st[k] for k in SPLIT_SEASONS},
        "subsets": {
            "train_50k": {"plays": len(keep), "by_bucket": {b: sub_buckets[b] for b in PRECEDENCE}},
            "eval_lite": {
                "plays": len(lite),
                "games": len({r["game_id"] for r in lite if r["eval_lite_part"] == "game"}),
                "by_part": dict(lite_parts),
                "by_bucket": {b: lite_buckets[b] for b in PRECEDENCE},
                "game_part_by_bucket": {b: lite_game_buckets[b] for b in PRECEDENCE},
            },
        },
        "text_yards_check": st["text_yards_check"],
        "files": {},
    }
    for name in list(SPLIT_SEASONS) + ["train_50k", "eval_lite"]:
        path = out_dir / f"{name}.jsonl"
        with open(path, "rb") as f:
            rows = sum(1 for _ in f)
        manifest["files"][path.name] = {"rows": rows, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest["frozen_eval"] = {
        "file": "test.jsonl",
        "sha256": manifest["files"]["test.jsonl"]["sha256"],
        "note": "Touched only for final ladder numbers. Every result file must carry this hash.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / "text_yards_examples.json").write_text(json.dumps(stats.gain_examples, indent=1, ensure_ascii=False))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", type=Path, default=None, help="raw parquet dir (default PLAYPARSE_DATA_DIR)")
    ap.add_argument("--out-dir", type=Path, default=paths.DATA_PROCESSED)
    ap.add_argument("--copy-manifest-to", type=Path, default=paths.RESULTS / "p1" / "dataset_manifest.json",
                    help="where to put the committed copy of the manifest ('' to skip)")
    args = ap.parse_args(argv)
    manifest = build(args.data_dir, args.out_dir)
    if str(args.copy_manifest_to):
        args.copy_manifest_to.parent.mkdir(parents=True, exist_ok=True)
        args.copy_manifest_to.write_text(json.dumps(manifest, indent=2) + "\n")
    for name, f in manifest["files"].items():
        print(f"{name:18s} {f['rows']:>8,} rows  sha256={f['sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
