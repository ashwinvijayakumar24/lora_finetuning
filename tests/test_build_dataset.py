"""Dataset build on a tiny synthetic raw directory made from the real fixture rows."""
import json
from pathlib import Path

import pandas as pd
import pytest

from playparse import paths
from playparse.data.build_dataset import RECORD_KEYS, build, desc_gain
from playparse.data.load_pbp import PBP_COLUMNS
from playparse.ffscore.schema import PlayLabel

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "gt_cases.json").read_text())


@pytest.mark.parametrize(
    "desc, expected",
    [
        ("27-J.Dobbins right guard to LAC 32 for 2 yards (95-C.Jones).", 2),
        ("4-J.Cook left guard to ARI 4 for no gain (2-Ma.Wilson).", 0),
        ("14-S.Darnold sacked at MIN 35 for -12 yards", -12),
        ("pass short right to 1-J.Downs for 1 yard, TOUCHDOWN.", 1),
        ("... for 2 yards. The Replay Official reviewed ... and the play was REVERSED. ... for 3 yards.", 3),
        ("pass incomplete short middle to 24-S.Moore.", None),
    ],
)
def test_desc_gain(desc, expected):
    assert desc_gain(desc) == expected


def _write_raw(raw: Path) -> None:
    rows = [c["row"] for c in FIXTURES]
    df = pd.DataFrame(rows)[list(PBP_COLUMNS)]
    # spread the fixture rows over one train, one val, and one test season, one game each
    for season in (2015, 2023, 2024):
        part = df.copy()
        part["season"] = season
        part["game_id"] = f"{season}_01_AAA_BBB"
        part["play_id"] = range(1, len(part) + 1)
        part.to_parquet(raw / f"pbp_{season}.parquet")


def test_build_is_deterministic_and_well_formed(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_raw(raw)
    seasons = (2015, 2023, 2024)
    m1 = build(raw, tmp_path / "out1", seasons=seasons, log=lambda *_: None)
    m2 = build(raw, tmp_path / "out2", seasons=seasons, log=lambda *_: None)
    assert {k: v["sha256"] for k, v in m1["files"].items()} == {k: v["sha256"] for k, v in m2["files"].items()}
    n = len(FIXTURES)
    assert m1["files"]["train.jsonl"]["rows"] == n
    assert m1["files"]["test.jsonl"]["rows"] == n
    assert m1["frozen_eval"]["sha256"] == m1["files"]["test.jsonl"]["sha256"]
    assert sum(m1["splits"]["test"]["by_bucket"].values()) == n
    assert m1["splits"]["test"]["name_in_desc_rate"] == 1.0

    lines = (tmp_path / "out1" / "test.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    assert tuple(first) == RECORD_KEYS
    for line in lines:
        rec = json.loads(line)
        PlayLabel.from_json(rec["label"])  # every label validates
    lite = [json.loads(x) for x in (tmp_path / "out1" / "eval_lite.jsonl").read_text().splitlines()]
    assert all(r["eval_lite_part"] in {"game", "topup"} for r in lite)


@pytest.mark.slow
def test_committed_manifest_matches_processed_files():
    """After a full build, the processed files hash to what the committed manifest says."""
    committed = json.loads((paths.RESULTS / "p1" / "dataset_manifest.json").read_text())
    local = paths.DATA_PROCESSED / "manifest.json"
    if not local.exists():
        pytest.skip("dataset not built locally")
    from playparse.data.build_dataset import sha256_file

    for name, meta in committed["files"].items():
        assert sha256_file(paths.DATA_PROCESSED / name) == meta["sha256"], name
