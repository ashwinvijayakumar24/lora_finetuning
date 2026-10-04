"""Prompt rendering with the real Llama 3.2 tokenizer, plus a slow end-to-end R1 smoke run.

The tokenizer-only tests are fast (no weights loaded) and skip when the checkpoint
directory is absent. The smoke test loads the 1B model and is marked slow.
"""
import json
from pathlib import Path

import pytest

from playparse.eval.baselines.hf_predictor import CHAT_DATE_STRING, build_fewshot_messages, load_fewshot, render_chat
from playparse.eval.harness import load_records, run_eval
from playparse.paths import WEIGHTS

FIXTURE = Path(__file__).parent / "fixtures" / "eval_mini.jsonl"
needs_weights = pytest.mark.skipif(not (WEIGHTS / "tokenizer.json").exists(), reason="Llama weights not present")


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(WEIGHTS))


@needs_weights
def test_single_bos_and_assistant_header(tokenizer):
    rec = {"posteam": "PHI", "desc": "1-J.Hurts up the middle for 3 yards."}
    text = render_chat(tokenizer, build_fewshot_messages(rec, load_fewshot()))
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert ids.count(tokenizer.bos_token_id) == 1 and ids[0] == tokenizer.bos_token_id
    assert text.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
    # The guard that matters: add_special_tokens=True would have doubled BOS.
    assert tokenizer(text)["input_ids"][:2] == [tokenizer.bos_token_id] * 2


@needs_weights
def test_chat_date_is_pinned(tokenizer):
    rec = {"posteam": "PHI", "desc": "x"}
    text = render_chat(tokenizer, build_fewshot_messages(rec, []))
    assert f"Today Date: {CHAT_DATE_STRING}" in text
    # Without the pin the template uses today's date, so the prompt drifts daily.
    unpinned = tokenizer.apply_chat_template(build_fewshot_messages(rec, []), tokenize=False, add_generation_prompt=True)
    assert "Today Date:" in unpinned


@needs_weights
def test_fewshot_prompt_length_is_modest(tokenizer):
    rec = load_records(FIXTURE)[0]
    n = len(tokenizer(render_chat(tokenizer, build_fewshot_messages(rec, load_fewshot())), add_special_tokens=False)["input_ids"])
    assert 600 < n < 2500


@pytest.mark.slow
@needs_weights
def test_r1_smoke_on_mps(tmp_path):
    """R1 on 5 hand-made plays with the real 1B model: runs, never crashes, writes artifacts."""
    from playparse.eval.baselines.hf_predictor import make_r1

    records = load_records(FIXTURE)[:5]
    pred = make_r1(WEIGHTS, batch_size=5, max_new_tokens=128)
    res = run_eval(pred, records, rung="r1", out_dir=tmp_path, n_boot=50)
    assert res["overall"]["n"] == 5
    assert 0.0 <= res["overall"]["valid_rate"]["point"] <= 1.0
    rows = [json.loads(l) for l in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 5 and all(isinstance(r["raw"], str) and r["raw"] for r in rows)
    assert all(r["usage"]["output_tokens"] > 0 for r in rows)
    # Greedy decoding is deterministic: a second call on the same batch gives identical text.
    again = pred.predict_batch(records)
    assert [p.text for p in again] == [r["raw"] for r in rows]
