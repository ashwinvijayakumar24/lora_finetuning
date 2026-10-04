"""Tokenization and loss-masking tests, using the real Llama 3.2 tokenizer."""
from __future__ import annotations

import pytest
import torch

from p2_fixtures import load_tokenizer, sample_records
from playparse.prompt import CHAT_DATE_STRING, build_messages
from playparse.train.collate import (
    IGNORE_INDEX,
    OverLengthError,
    collate,
    count_target_tokens,
    encode_prompt,
    encode_record,
    encode_records,
    end_of_turn_id,
    pad_token_id,
    render_prompt,
    stop_token_ids,
)


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer()


def test_end_of_turn_is_eot_not_end_of_text(tok):
    eot = end_of_turn_id(tok)
    assert tok.convert_ids_to_tokens(eot) == "<|eot_id|>"
    assert eot == tok.eos_token_id == 128009
    # The template closes the assistant turn with exactly this token.
    full = tok.apply_chat_template(
        build_messages("PHI", "x") + [{"role": "assistant", "content": "{}"}],
        tokenize=False,
        date_string=CHAT_DATE_STRING,
    )
    assert full.endswith("{}<|eot_id|>")
    assert stop_token_ids(tok)[0] == eot and 128001 in stop_token_ids(tok)


def test_bos_not_duplicated(tok):
    text = render_prompt(tok, "PHI", "26-S.Barkley up the middle for 3 yards.")
    assert text.startswith("<|begin_of_text|>")
    # The trap: the template already wrote BOS, and add_special_tokens=True adds another.
    trap = tok(text, add_special_tokens=True)["input_ids"]
    assert trap[:2] == [tok.bos_token_id, tok.bos_token_id]
    # What we actually use.
    ids = encode_prompt(tok, "PHI", "26-S.Barkley up the middle for 3 yards.")
    assert ids[0] == tok.bos_token_id and ids.count(tok.bos_token_id) == 1
    for rec in sample_records():
        ex = encode_record(rec, tok)
        assert ex.input_ids.count(tok.bos_token_id) == 1


def test_prompt_is_date_independent(tok):
    """Without date_string the template writes today's date into the system header."""
    msgs = build_messages("PHI", "x")
    a = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, date_string="01 Jan 2025")
    b = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, date_string="02 Jan 2025")
    assert a != b  # the date really is part of the prompt
    assert render_prompt(tok, "PHI", "x") == render_prompt(tok, "PHI", "x")
    assert f"Today Date: {CHAT_DATE_STRING}" in render_prompt(tok, "PHI", "x")


def test_rendering_is_stable_regardless_of_system_date(tok, monkeypatch):
    """Fake the clock the template reads ("strftime_now") and check our rendering ignores it."""
    import datetime as dt

    import transformers.utils.chat_template_utils as ctu

    def fake_clock(day):
        class FakeDatetime(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return dt.datetime(2031, 1, day, 12, 0, 0)

        return FakeDatetime

    rendered, unpinned = [], []
    for day in (1, 2):
        monkeypatch.setattr(ctu, "datetime", fake_clock(day))
        rendered.append(render_prompt(tok, "PHI", "x"))
        unpinned.append(tok.apply_chat_template(build_messages("PHI", "x"), tokenize=False, add_generation_prompt=True))
    assert unpinned[0] != unpinned[1] and "Today Date: 01 Jan 2031" in unpinned[0]  # the hazard is real
    assert rendered[0] == rendered[1]
    assert encode_prompt(tok, "PHI", "x") == encode_prompt(tok, "PHI", "x")


def test_prompt_matches_template_tokenization(tok):
    """Our prompt ids equal the template's own tokenization (with the pinned date)."""
    rec = sample_records()[0]
    ref = tok.apply_chat_template(
        build_messages(rec["posteam"], rec["desc"]),
        tokenize=True,
        add_generation_prompt=True,
        date_string=CHAT_DATE_STRING,
    )
    ref_ids = ref["input_ids"] if hasattr(ref, "keys") else ref
    assert encode_prompt(tok, rec["posteam"], rec["desc"]) == list(ref_ids)


def test_full_sequence_matches_template_with_assistant_turn(tok):
    """prompt ids + completion ids == template applied to the whole conversation.

    We tokenize the two halves separately. This test shows no BPE merge crosses the
    '\\n\\n' / '{' boundary, so training sequences equal the 'natural' rendering too.
    """
    for rec in sample_records():
        ex = encode_record(rec, tok)
        full = tok.apply_chat_template(
            build_messages(rec["posteam"], rec["desc"]) + [{"role": "assistant", "content": rec["label"]}],
            tokenize=False,
            date_string=CHAT_DATE_STRING,
        )
        assert ex.input_ids == tok(full, add_special_tokens=False)["input_ids"]


def test_mask_covers_exactly_the_completion(tok):
    for rec in sample_records():
        ex = encode_record(rec, tok)
        targets = [t for t, l in zip(ex.input_ids, ex.labels) if l != IGNORE_INDEX]
        assert ex.labels[: ex.n_prompt] == [IGNORE_INDEX] * ex.n_prompt
        assert ex.labels[ex.n_prompt :] == ex.input_ids[ex.n_prompt :]
        # The supervised tokens decode to exactly the label JSON plus <|eot_id|>.
        assert tok.decode(targets) == rec["label"] + "<|eot_id|>"
        assert targets[-1] == end_of_turn_id(tok)


def test_collate_right_pads_and_masks_padding(tok):
    exs = [encode_record(r, tok) for r in sample_records()]
    pad = pad_token_id(tok)
    batch = collate(exs, pad)
    lens = [len(e) for e in exs]
    assert batch["input_ids"].shape == (3, max(lens))
    for i, e in enumerate(exs):
        n = len(e)
        assert batch["attention_mask"][i].tolist() == [1] * n + [0] * (max(lens) - n)
        assert (batch["labels"][i, n:] == IGNORE_INDEX).all()
        assert (batch["input_ids"][i, n:] == pad).all()
        assert batch["input_ids"][i, :n].tolist() == e.input_ids
        # eot is a target in every row
        assert end_of_turn_id(tok) in batch["labels"][i].tolist()
    # Targets after the one-position shift = completion tokens (none is at position 0).
    assert count_target_tokens(batch["labels"]) == sum(e.n_completion for e in exs)
    assert collate(exs, pad, pad_to_multiple_of=64)["input_ids"].shape[1] % 64 == 0


def test_over_length_raises_and_drop_reports(tok):
    recs = sample_records()
    n = len(encode_record(recs[2], tok))
    with pytest.raises(OverLengthError):
        encode_record(recs[2], tok, max_len=n - 1)
    assert len(encode_record(recs[2], tok, max_len=n)) == n
    with pytest.raises(OverLengthError):
        encode_records(recs, tok, max_len=n - 1)
    kept, report = encode_records(recs, tok, max_len=n - 1, on_overlength="drop")
    assert report.n_records == 3 and report.n_kept == len(kept) < 3
    assert report.over_length and all(o["n_tokens"] > n - 1 for o in report.over_length)


def test_mask_prompt_false_supervises_everything(tok):
    rec = sample_records()[0]
    masked = encode_record(rec, tok, mask_prompt=True)
    unmasked = encode_record(rec, tok, mask_prompt=False)
    assert masked.input_ids == unmasked.input_ids
    assert unmasked.labels == unmasked.input_ids
    b = collate([unmasked], pad_token_id(tok))
    assert count_target_tokens(b["labels"]) == len(unmasked) - 1
    assert torch.equal(collate([masked], 0)["input_ids"], b["input_ids"])
