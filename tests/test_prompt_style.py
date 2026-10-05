"""prompt_style: "full" (system prompt, the default) vs "minimal" (no system message).

Three things are pinned here:

1. Train/eval parity holds for both styles: the training ids
   (`collate.encode_prompt`) and the eval harness ids (`hf_predictor.render_record`,
   tokenized as a left-padded batch) are identical.
2. The default is "full" everywhere, and a "full" predictor config hashes exactly
   as before the option existed (checked against the committed P3 pilot result).
3. A minimal prompt is much shorter: exactly the system prompt's tokens fewer,
   with the date still pinned in the template's own system header.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import pytest

from p2_fixtures import load_tokenizer, sample_records
from playparse import prompt as prompt_mod
from playparse.eval.baselines import hf_predictor
from playparse.eval.harness import config_hash, load_records
from playparse.prompt import CHAT_DATE_STRING, SYSTEM_PROMPT, build_messages
from playparse.train import collate, generate, val_eval
from playparse.train.build import DataSpec, RunSpec

REPO = Path(__file__).resolve().parent.parent
FIXTURE = Path(__file__).parent / "fixtures" / "eval_mini.jsonl"
STYLES = ("full", "minimal")


def _records() -> list[dict]:
    recs = sample_records() + load_records(FIXTURE)
    recs.append({**recs[0], "posteam": None, "play_id": 999})
    return recs


# ----------------------------------------------------------------- messages


def test_build_messages_styles():
    full = build_messages("PHI", "x")
    assert full == build_messages("PHI", "x", style="full")
    assert [m["role"] for m in full] == ["system", "user"] and full[0]["content"] == SYSTEM_PROMPT
    minimal = build_messages("PHI", "x", style="minimal")
    assert minimal == [full[1]]  # same user turn, no system message
    with pytest.raises(ValueError):
        build_messages("PHI", "x", style="short")
    with pytest.raises(ValueError):  # a custom system prompt would be silently dropped
        build_messages("PHI", "x", system="other", style="minimal")


def test_default_style_is_full_everywhere():
    assert prompt_mod.DEFAULT_PROMPT_STYLE == "full"
    for fn, name in [(collate.render_prompt, "prompt_style"), (collate.encode_prompt, "prompt_style"),
                     (collate.encode_record, "prompt_style"), (collate.encode_records, "prompt_style"),
                     (generate.generate_for_records, "prompt_style"),
                     (val_eval.harness_val_callback, "prompt_style"),
                     (hf_predictor.render_record, "style"), (hf_predictor.build_fewshot_messages, "style"),
                     (hf_predictor.HFPredictor.__init__, "prompt_style")]:
        assert inspect.signature(fn).parameters[name].default == "full", fn.__name__
    assert DataSpec().prompt_style == "full"
    assert RunSpec.from_dict({"data": {"train": "x"}}).data.prompt_style == "full"  # old run_spec.json files


def test_dataspec_rejects_unknown_style():
    with pytest.raises(ValueError):
        DataSpec(prompt_style="tiny")
    assert RunSpec.from_dict({"data": {"prompt_style": "minimal"}}).data.prompt_style == "minimal"


# ----------------------------------------------------------------- parity


@pytest.mark.parametrize("style", STYLES)
def test_train_and_eval_prompt_ids_identical(style):
    tok = load_tokenizer()
    for rec in _records():
        train_ids = collate.encode_prompt(tok, rec.get("posteam"), rec["desc"], prompt_style=style)
        eval_ids = tok(hf_predictor.render_record(tok, rec, style=style), add_special_tokens=False)["input_ids"]
        assert eval_ids == train_ids
        ex = collate.encode_record(rec, tok, prompt_style=style)
        assert ex.input_ids[: ex.n_prompt] == eval_ids


@pytest.mark.parametrize("style", STYLES)
def test_batched_left_padded_eval_ids_match_training(style):
    tok = load_tokenizer()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = "<|finetune_right_pad_id|>"
    recs = _records()
    enc = tok([hf_predictor.render_record(tok, r, style=style) for r in recs], padding=True,
              add_special_tokens=False)
    for rec, ids, mask in zip(recs, enc["input_ids"], enc["attention_mask"]):
        real = [t for t, m in zip(ids, mask) if m]
        assert real == collate.encode_prompt(tok, rec.get("posteam"), rec["desc"], prompt_style=style)


def test_default_ids_unchanged():
    tok = load_tokenizer()
    for rec in _records():
        ref = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": prompt_mod.render_user(rec.get("posteam"), rec["desc"])}],
            tokenize=True, add_generation_prompt=True, date_string=CHAT_DATE_STRING)
        ref = list(ref["input_ids"]) if hasattr(ref, "keys") else list(ref)
        assert collate.encode_prompt(tok, rec.get("posteam"), rec["desc"]) == ref


# ----------------------------------------------------------------- lengths


def test_minimal_prompt_is_much_shorter_and_still_dated():
    tok = load_tokenizer()
    sys_tokens = len(tok(SYSTEM_PROMPT, add_special_tokens=False)["input_ids"])
    assert sys_tokens == 186
    for rec in _records():
        full = collate.encode_prompt(tok, rec.get("posteam"), rec["desc"])
        minimal = collate.encode_prompt(tok, rec.get("posteam"), rec["desc"], prompt_style="minimal")
        # The system message's content is the only difference.
        assert len(full) - len(minimal) == sys_tokens
        user = len(tok(prompt_mod.render_user(rec.get("posteam"), rec["desc"]), add_special_tokens=False)["input_ids"])
        assert len(minimal) - user == 35  # fixed template without a system prompt (221 with it)
    text = collate.render_prompt(tok, "PHI", "x", prompt_style="minimal")
    # The Llama 3.2 template writes its own system header even with no system message.
    assert text.startswith("<|begin_of_text|><|start_header_id|>system<|end_header_id|>")
    assert f"Today Date: {CHAT_DATE_STRING}" in text and "You extract" not in text


def test_minimal_training_example_keeps_completion():
    tok = load_tokenizer()
    rec = _records()[0]
    full = collate.encode_record(rec, tok)
    minimal = collate.encode_record(rec, tok, prompt_style="minimal")
    assert full.input_ids[full.n_prompt:] == minimal.input_ids[minimal.n_prompt:]
    assert full.labels[full.n_prompt:] == minimal.labels[minimal.n_prompt:]
    assert minimal.n_completion == full.n_completion


# ----------------------------------------------------------------- config hash


def _pilot_predictor(style: str):
    """A LoRAPredictor with the committed P3 pilot's settings, without loading a model."""
    from playparse.eval.baselines.lora_predictor import LoRAPredictor

    res = json.loads((REPO / "results" / "p3_local_pilot" / "eval" / "lora" / "result.json").read_text())
    cfg = res["meta"]["config"]
    p = object.__new__(LoRAPredictor)
    p.name, p.weights, p.examples_id = "lora", "/somewhere/weights", None
    p.max_new_tokens, p.batch_size, p.dtype, p.device = 256, 16, "bfloat16", "mps"
    p.system, p.prompt_style = SYSTEM_PROMPT, style
    p.adapter_dir, p.merge = Path("x/adapter"), False
    p.fingerprint = {k: cfg[k] for k in ("adapter_sha256", "adapter_r", "adapter_alpha", "adapter_targets")}
    return p, res["meta"]["config_hash"]


def test_full_config_hash_matches_committed_pilot():
    p, committed = _pilot_predictor("full")
    cfg = {"rung": "lora", "predictor": p.name, **p.config()}
    assert "prompt_style" not in cfg
    assert config_hash(cfg) == committed


def test_minimal_style_changes_config_hash():
    p, committed = _pilot_predictor("minimal")
    cfg = {"rung": "lora", "predictor": p.name, **p.config()}
    assert cfg["prompt_style"] == "minimal" and cfg["system_prompt_sha"] is None
    assert config_hash(cfg) != committed


# ----------------------------------------------------------------- plumbing


def test_eval_cli_passes_prompt_style(monkeypatch, tmp_path):
    from playparse.eval import run

    seen = {}

    def fake_make_lora(weights, adapter, merge=False, **kw):
        seen.update(kw)
        raise SystemExit(0)

    import playparse.eval.baselines.lora_predictor as lp

    monkeypatch.setattr(lp, "make_lora", fake_make_lora)
    with pytest.raises(SystemExit):
        run.main(["--rung", "lora", "--adapter", "a", "--data", str(FIXTURE), "--out", str(tmp_path / "o"),
                  "--prompt-style", "minimal"])
    assert seen["prompt_style"] == "minimal"
    seen.clear()
    with pytest.raises(SystemExit):
        run.main(["--rung", "lora", "--adapter", "a", "--data", str(FIXTURE), "--out", str(tmp_path / "o")])
    assert seen["prompt_style"] == "full"
    with pytest.raises(SystemExit):
        run.main(["--rung", "r1", "--data", str(FIXTURE), "--out", str(tmp_path / "o"), "--prompt-style", "x"])


def test_val_callback_decodes_with_training_style(monkeypatch):
    seen = {}

    def fake_generate(model, tokenizer, records, **kw):
        seen.update(kw)
        return [r["label"] for r in records]

    monkeypatch.setattr(generate, "generate_for_records", fake_generate)
    recs = sample_records()
    cb = val_eval.harness_val_callback(None, recs, prompt_style="minimal")
    out = cb(None, 1)
    assert seen["prompt_style"] == "minimal" and out["val_exact_match"] == 1.0


def test_train_cli_encodes_with_spec_style(tmp_path):
    load_tokenizer()
    spec = importlib.util.spec_from_file_location("train_script_ps", REPO / "scripts" / "train.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    with open(tmp_path / "train.jsonl", "w") as f:
        for r in sample_records():
            f.write(json.dumps(r) + "\n")
    means = {}
    for style in STYLES:
        out = tmp_path / style
        assert script.main(["--set", f"data.train={json.dumps(str(tmp_path / 'train.jsonl'))}",
                            f"train.output_dir={json.dumps(str(out))}", f"data.prompt_style={style}",
                            "--dry-run"]) == 0
        assert json.loads((out / "run_spec.json").read_text())["data"]["prompt_style"] == style
        means[style] = json.loads((out / "run_meta.json").read_text())["train_report"]["mean_tokens"]
    assert means["full"] - means["minimal"] == pytest.approx(186)


def test_lora_predictor_refuses_mismatched_style(tmp_path):
    from playparse.eval.baselines.lora_predictor import LoRAPredictor, trained_prompt_style

    adapter = tmp_path / "run" / "checkpoints" / "step_0000010" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8, "target_modules": ["q_proj"]}))
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    assert trained_prompt_style(adapter) is None
    (tmp_path / "run" / "run_spec.json").write_text(json.dumps({"data": {"train": "t"}}))
    assert trained_prompt_style(adapter) == "full"  # a spec from before the option
    (tmp_path / "run" / "run_spec.json").write_text(json.dumps({"data": {"prompt_style": "minimal"}}))
    assert trained_prompt_style(adapter) == "minimal"
    with pytest.raises(ValueError, match="prompt_style='minimal'"):
        LoRAPredictor("/nonexistent/weights", adapter)  # raises before any model load
