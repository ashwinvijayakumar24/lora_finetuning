"""The training prompt and the eval prompt must be the same token ids.

An adapter is only meaningful on the prompt it was trained on. Training encodes
prompts with `playparse.train.collate.encode_prompt`; the eval harness (R1-R3 and
the LoRA rung) renders them with `playparse.eval.baselines.hf_predictor.render_record`
and tokenizes a left-padded batch. These tests pin that both paths produce
byte-identical ids for the same record, including after batch padding is removed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from p2_fixtures import load_tokenizer, sample_records
from playparse import prompt as prompt_mod
from playparse.eval.baselines import hf_predictor
from playparse.eval.harness import load_records
from playparse.train import collate, generate

FIXTURE = Path(__file__).parent / "fixtures" / "eval_mini.jsonl"


def _records() -> list[dict]:
    recs = sample_records() + load_records(FIXTURE)
    recs.append({**recs[0], "posteam": None, "play_id": 999})  # missing posteam renders as UNK
    return recs


def test_single_date_constant_shared_by_train_and_eval():
    # One definition, imported (not copied) by both sides.
    assert hf_predictor.CHAT_DATE_STRING is prompt_mod.CHAT_DATE_STRING
    assert not hasattr(collate, "DEFAULT_DATE_STRING")
    import inspect

    for fn in (collate.render_prompt, collate.encode_prompt, collate.encode_record, generate.generate_for_records):
        assert inspect.signature(fn).parameters["date_string"].default is prompt_mod.CHAT_DATE_STRING


def test_train_and_eval_prompt_token_ids_identical():
    tok = load_tokenizer()
    for rec in _records():
        train_ids = collate.encode_prompt(tok, rec.get("posteam"), rec["desc"])
        eval_text = hf_predictor.render_record(tok, rec)
        eval_ids = tok(eval_text, add_special_tokens=False)["input_ids"]
        assert eval_ids == train_ids, rec["desc"][:60]
        # The training example is the prompt ids followed by the completion ids.
        ex = collate.encode_record(rec, tok)
        assert ex.input_ids[: ex.n_prompt] == eval_ids


def test_batched_left_padded_eval_ids_match_training_ids():
    """HFPredictor tokenizes a padded batch; stripping pads must give the training ids."""
    tok = load_tokenizer()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = "<|finetune_right_pad_id|>"
    recs = _records()
    enc = tok([hf_predictor.render_record(tok, r) for r in recs], padding=True, add_special_tokens=False)
    for rec, ids, mask in zip(recs, enc["input_ids"], enc["attention_mask"]):
        real = [t for t, m in zip(ids, mask) if m]
        assert real == collate.encode_prompt(tok, rec.get("posteam"), rec["desc"])


def test_fewshot_rendering_still_differs():
    """Sanity: the parity is not vacuous; R2's prompt is a different, longer sequence."""
    tok = load_tokenizer()
    rec = _records()[0]
    zero = hf_predictor.render_record(tok, rec)
    few = hf_predictor.render_record(tok, rec, hf_predictor.load_fewshot())
    assert few != zero and len(few) > 2 * len(zero)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
