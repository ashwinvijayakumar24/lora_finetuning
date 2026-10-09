"""The v2 dataset files: the round-trip invariant on every play, and the manifest.

The slow tests need `python -m playparse.data.build_dataset_v2` to have run
(data/processed_v2/, git-ignored). The fast test builds a tiny v2 set from a
synthetic v1 file and raw table.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from playparse import paths
from playparse.data import build_dataset_v2 as b2
from playparse.data.build_dataset import RECORD_KEYS, sha256_file
from playparse.ffscore.schema import PlayLabel
from playparse.ffscore.schema_v2 import PlayLabelV2, to_v1

V2_DIR = paths.REPO_ROOT / "data" / "processed_v2"


def test_build_small(tmp_path, monkeypatch):
    v1 = tmp_path / "v1"
    v1.mkdir()
    rec = {"game_id": "2019_01_X_Y", "play_id": 7, "season": 2019, "week": 1, "season_type": "REG",
           "posteam": "PHI", "desc": "1-A.Run up the middle to DAL 22 for 48 yards.", "bucket": "normal",
           "label": '{"nullified":false,"credits":[{"player":"A.Run","stat":"rush_yds","value":48}]}'}
    for name in b2.FILES:
        extra = {"eval_lite_part": "game"} if name == "eval_lite" else {}
        (v1 / f"{name}.jsonl").write_text(json.dumps({**rec, **extra}) + "\n")
    raw = {"game_id": "2019_01_X_Y", "play_id": 7.0, "posteam": "PHI", "defteam": "DAL", "yrdln": "PHI 30",
           "yardline_100": 70.0, **{c: None for c in b2.V2_COLUMNS[6:]}}
    monkeypatch.setattr(b2, "load_raw_index", lambda data_dir, seasons=None: {("2019_01_X_Y", 7): raw})
    m = b2.build(v1, tmp_path / "v2", log=lambda *_: None)
    out = json.loads((tmp_path / "v2" / "train.jsonl").read_text())
    assert tuple(out)[: len(RECORD_KEYS)] == RECORD_KEYS and out["los"] == "PHI 30"
    assert out["label_v2"] == '{"nullified":false,"credits":[{"player":"A.Run","stat":"rush_yds","to":"DAL 22"}]}'
    assert m["round_trip"]["train"]["rate"] == 1.0
    assert json.loads((tmp_path / "v2" / "eval_lite.jsonl").read_text())["eval_lite_part"] == "game"
    assert m["literal_spots"]["train:normal"]["to_literal_rate"] == 1.0


def _need_v2():
    if not (V2_DIR / "manifest.json").exists():
        pytest.skip("v2 dataset not built locally (python -m playparse.data.build_dataset_v2)")


@pytest.mark.slow
@pytest.mark.parametrize("split", ["train", "val", "test", "train_50k", "eval_lite"])
def test_round_trip_every_play(split):
    """to_v1(label_v2) == the frozen v1 label for every play (the T3b invariant)."""
    _need_v2()
    n = 0
    with open(V2_DIR / f"{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            v2 = PlayLabelV2.from_json(rec["label_v2"])
            assert v2.to_json() == rec["label_v2"]  # stored canonically
            back = to_v1(v2, rec["los"], rec["posteam"])
            assert back.canonical() == PlayLabel.from_json(rec["label"]).canonical(), rec["play_id"]
            n += 1
    assert n > 0


@pytest.mark.slow
def test_v2_files_match_committed_manifest_and_v1_inputs():
    _need_v2()
    committed = json.loads((paths.RESULTS / "t3b" / "dataset_v2_manifest.json").read_text())
    for name, meta in committed["files"].items():
        assert sha256_file(V2_DIR / name) == meta["sha256"], name
    v1 = json.loads((paths.RESULTS / "p1" / "dataset_manifest.json").read_text())
    for name, sha in committed["v1_inputs_sha256"].items():
        assert v1["files"][name]["sha256"] == sha, name  # built from the frozen v1 files
    assert committed["frozen_eval_v1"]["sha256"].startswith("13d0d714")


@pytest.mark.slow
def test_v2_eval_lite_has_the_v1_plays_and_labels():
    _need_v2()
    v1 = pd.read_json(paths.DATA_PROCESSED / "eval_lite.jsonl", lines=True, dtype=False)
    v2 = pd.read_json(V2_DIR / "eval_lite.jsonl", lines=True, dtype=False)
    assert list(v1["label"]) == list(v2["label"]) and list(v1["play_id"]) == list(v2["play_id"])
