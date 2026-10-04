"""Every train and val example fits the default config's max_len (slow: tokenizes ~330k plays).

configs/train_default.yaml uses on_overlength: raise, so a single over-length
example would crash a full run at startup. See docs/issues/p3-default-max-len-too-short.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from playparse.paths import DATA_PROCESSED, WEIGHTS

REPO = Path(__file__).resolve().parent.parent


@pytest.mark.slow
@pytest.mark.parametrize("split", ["train.jsonl", "val.jsonl"])
def test_default_max_len_fits_every_example(split):
    path = DATA_PROCESSED / split
    if not path.exists() or not (WEIGHTS / "tokenizer.json").exists():
        pytest.skip("processed data or tokenizer not present")
    from transformers import AutoTokenizer

    from playparse.train.build import RunSpec
    from playparse.train.collate import encode_record

    max_len = RunSpec.from_file(REPO / "configs" / "train_default.yaml").data.max_len
    tok = AutoTokenizer.from_pretrained(str(WEIGHTS))
    longest = 0
    with open(path) as f:
        for line in f:
            longest = max(longest, len(encode_record(json.loads(line), tok)))
    assert longest <= max_len, f"{split}: longest example {longest} > max_len {max_len}"
