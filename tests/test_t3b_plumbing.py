"""T3b plumbing: the minimal_v2 prompt style, schema guards, and v2 -> v1 scoring.

Pinned here:
1. The v1 styles render exactly as before, even for records that carry `los`.
2. Train/eval prompt parity holds for minimal_v2 (training ids == harness ids,
   including a left-padded batch), and the completion is the v2 label.
3. Style and schema cannot disagree anywhere (RunSpec, eval CLI, LoRA run_spec).
4. The val callback and the eval predictor score v2 outputs as the v1 labels they
   convert to; an output that is not valid v2 is invalid.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from p2_fixtures import load_tokenizer, sample_records
from playparse.data.ground_truth import build_label
from playparse.data.ground_truth_v2 import build_label_v2
from playparse.eval.baselines import hf_predictor
from playparse.eval.harness import config_hash
from playparse.prompt import (
    PROMPT_STYLES,
    build_messages,
    check_style_schema,
    render_user,
    render_user_v2,
    schema_of_style,
)
from playparse.train import collate, generate, val_eval
from playparse.train.build import DataSpec, RunSpec

CASES = json.loads((Path(__file__).parent / "fixtures" / "t3b_cases.json").read_text())


def v2_records() -> list[dict]:
    """Dataset-shaped v2 records from the real 2019 fixture plays."""
    out = []
    for i, c in enumerate(CASES):
        row = c["row"]
        res = build_label_v2(row, build_label(row).label)
        out.append({"game_id": row["game_id"], "play_id": int(row["play_id"]), "posteam": row["posteam"],
                    "desc": row["desc"], "bucket": "lateral" if "lateral" in c["case"] else "normal",
                    "label": c["label"], "los": res.los, "label_v2": res.label_v2.to_json()})
    return out


# ----------------------------------------------------------------- styles and schemas


def test_style_schema_table():
    assert PROMPT_STYLES == ("full", "minimal", "minimal_v2")
    assert [schema_of_style(s) for s in PROMPT_STYLES] == ["v1", "v1", "v2"]
    assert check_style_schema("minimal_v2", None) == "v2" and check_style_schema("minimal", "v1") == "v1"
    for style, schema in [("minimal", "v2"), ("full", "v2"), ("minimal_v2", "v1"), ("minimal", "v3")]:
        with pytest.raises(ValueError):
            check_style_schema(style, schema)


def test_v1_styles_ignore_los():
    for style in ("full", "minimal"):
        assert build_messages("PHI", "x", style=style, los="PHI 30") == build_messages("PHI", "x", style=style)
    assert build_messages("PHI", "x", style="minimal")[0]["content"] == render_user("PHI", "x")


def test_minimal_v2_messages():
    msgs = build_messages("PHI", "x", style="minimal_v2", los="PHI 30")
    assert msgs == [{"role": "user", "content": "posteam: PHI\nlos: PHI 30\ndesc: x"}]
    assert render_user_v2(None, None, "x") == "posteam: UNK\nlos: UNK\ndesc: x"
    with pytest.raises(ValueError):
        build_messages("PHI", "x", system="other", style="minimal_v2")


def test_dataspec_guard():
    assert DataSpec().schema == "v1"
    assert DataSpec(prompt_style="minimal_v2").schema == "v2"
    assert DataSpec(prompt_style="minimal_v2", schema="v2").schema == "v2"
    for kw in ({"prompt_style": "minimal", "schema": "v2"}, {"prompt_style": "minimal_v2", "schema": "v1"},
               {"prompt_style": "full", "schema": "v2"}):
        with pytest.raises(ValueError):
            DataSpec(**kw)
    # old run_spec.json files have no schema key
    assert RunSpec.from_dict({"data": {"train": "x", "prompt_style": "minimal"}}).data.schema == "v1"
    assert RunSpec.from_dict({"data": {"schema": "v2", "prompt_style": "minimal_v2"}}).to_dict()["data"]["schema"] == "v2"


def test_t3b_config_is_r5_plus_v2():
    import yaml

    repo = Path(__file__).resolve().parent.parent
    base = yaml.safe_load((repo / "configs" / "train_default.yaml").read_text())
    t3b = yaml.safe_load((repo / "configs" / "t3b_spots.yaml").read_text())
    assert t3b["data"].pop("schema") == "v2" and t3b["data"].pop("prompt_style") == "minimal_v2"
    assert t3b["data"].pop("train").startswith("data/processed_v2/")
    assert t3b["data"].pop("val").startswith("data/processed_v2/")
    for k in ("schema", "prompt_style", "train", "val"):
        base["data"].pop(k, None)
    base["train"].pop("output_dir"), t3b["train"].pop("output_dir")
    assert t3b == base  # everything else is R5's recipe, unchanged
    spec = RunSpec.from_file(repo / "configs" / "t3b_spots.yaml")
    assert spec.data.schema == "v2" and spec.lora.r == 16 and spec.train.epochs == 2


# ----------------------------------------------------------------- collate and parity


def test_v2_completion_and_missing_fields():
    tok = load_tokenizer()
    rec = v2_records()[0]
    ex = collate.encode_record(rec, tok, prompt_style="minimal_v2")
    completion = tok.decode(ex.input_ids[ex.n_prompt:], skip_special_tokens=False)
    assert completion == rec["label_v2"] + collate.EOT_TOKEN
    assert '"to":' in completion and '"value":' in completion  # spots for yards, values for counts
    with pytest.raises(KeyError):  # a v1 file fed to a v2 run
        collate.encode_record({k: v for k, v in rec.items() if k != "los"}, tok, prompt_style="minimal_v2")
    with pytest.raises(KeyError):
        collate.encode_record({k: v for k, v in rec.items() if k != "label_v2"}, tok, prompt_style="minimal_v2")
    # the v1 styles still train on the v1 label, even on a v2 record
    v1 = collate.encode_record(rec, tok, prompt_style="minimal")
    assert tok.decode(v1.input_ids[v1.n_prompt:]) == rec["label"] + collate.EOT_TOKEN


def test_v1_styles_unchanged_on_v2_records():
    tok = load_tokenizer()
    for rec in v2_records():
        v1rec = {k: v for k, v in rec.items() if k not in ("los", "label_v2")}
        for style in ("full", "minimal"):
            assert collate.encode_record(rec, tok, prompt_style=style).input_ids == \
                collate.encode_record(v1rec, tok, prompt_style=style).input_ids


def test_minimal_v2_train_eval_parity():
    tok = load_tokenizer()
    recs = v2_records()
    for rec in recs:
        train_ids = collate.encode_prompt(tok, rec["posteam"], rec["desc"], prompt_style="minimal_v2",
                                          los=rec["los"])
        eval_ids = tok(hf_predictor.render_record(tok, rec, style="minimal_v2"), add_special_tokens=False)["input_ids"]
        assert eval_ids == train_ids
        ex = collate.encode_record(rec, tok, prompt_style="minimal_v2")
        assert ex.input_ids[: ex.n_prompt] == eval_ids
        assert f"los: {rec['los']}" in tok.decode(train_ids)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = "<|finetune_right_pad_id|>"
    enc = tok([hf_predictor.render_record(tok, r, style="minimal_v2") for r in recs], padding=True,
              add_special_tokens=False)
    for rec, ids, mask in zip(recs, enc["input_ids"], enc["attention_mask"]):
        real = [t for t, m in zip(ids, mask) if m]
        assert real == collate.encode_prompt(tok, rec["posteam"], rec["desc"], prompt_style="minimal_v2",
                                             los=rec["los"])


def test_harness_refuses_v1_records_for_v2():
    tok = load_tokenizer()
    rec = {k: v for k, v in v2_records()[0].items() if k != "los"}
    with pytest.raises(KeyError):
        hf_predictor.render_record(tok, rec, style="minimal_v2")


# ----------------------------------------------------------------- scoring v2 outputs


def test_val_callback_scores_v2_as_v1(monkeypatch, tmp_path):
    recs = v2_records()
    outputs = {r["play_id"]: r["label_v2"] for r in recs}
    # one play answered with a valid *v1* label: must count as invalid for a v2 run
    outputs[recs[0]["play_id"]] = recs[0]["label"]

    def fake_generate(model, tokenizer, records, **kw):
        assert kw["prompt_style"] == "minimal_v2"
        return [outputs[r["play_id"]] for r in records]

    monkeypatch.setattr(generate, "generate_for_records", fake_generate)
    cb = val_eval.harness_val_callback(None, recs, prompt_style="minimal_v2", predictions_dir=tmp_path)
    out = cb(None, 3)
    n = len(recs)
    assert out["val_exact_match"] == pytest.approx((n - 1) / n)
    assert out["val_valid_rate"] == pytest.approx((n - 1) / n)
    rows = [json.loads(x) for x in (tmp_path / "step_0000003.jsonl").read_text().splitlines()]
    assert all("as_v1" in r for r in rows) and rows[1]["raw"] == recs[1]["label_v2"]


def test_as_v1_texts_is_identity_for_v1():
    recs = sample_records()
    texts = [r["label"] for r in recs]
    assert val_eval.as_v1_texts(recs, texts, "minimal") == texts


def test_predictor_converts_v2_output():
    rec = v2_records()[0]
    text, usage = hf_predictor.HFPredictor._to_v1(rec, rec["label_v2"], {"output_tokens": 5})
    assert json.loads(text) == json.loads(rec["label"]) and usage["raw_v2"] == rec["label_v2"]
    bad, usage = hf_predictor.HFPredictor._to_v1(rec, rec["label"], {})
    assert "{" not in bad and "v2_error" in usage


def test_predictor_config_records_schema():
    p = object.__new__(hf_predictor.HFPredictor)
    p.name, p.weights, p.examples_id, p.max_new_tokens, p.batch_size = "x", "w", None, 256, 16
    p.dtype, p.device, p.system = "bfloat16", "cuda", hf_predictor.SYSTEM_PROMPT
    p.prompt_style = "minimal"
    v1cfg = p.config()
    p.prompt_style = "minimal_v2"
    v2cfg = p.config()
    assert "schema" not in v1cfg and v2cfg["schema"] == "v2" and v2cfg["prompt_style"] == "minimal_v2"
    assert config_hash(v1cfg) != config_hash(v2cfg)


def test_lora_guard_reads_schema_from_run_spec(tmp_path):
    from playparse.eval.baselines.lora_predictor import LoRAPredictor, trained_schema

    run = tmp_path / "run"
    adapter = run / "best"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    (run / "run_spec.json").write_text(json.dumps({"data": {"prompt_style": "minimal_v2", "schema": "v2"}}))
    assert trained_schema(adapter) == "v2"
    with pytest.raises(ValueError, match="prompt_style"):
        LoRAPredictor("w", adapter, prompt_style="minimal")
    (run / "run_spec.json").write_text(json.dumps({"data": {"prompt_style": "minimal"}}))
    assert trained_schema(adapter) == "v1"  # pre-T3b run spec: the style decides


def test_eval_cli_schema_guards(tmp_path, monkeypatch):
    from playparse.eval import run

    v1_file = Path(__file__).parent / "fixtures" / "eval_mini.jsonl"
    with pytest.raises(SystemExit, match="produces schema"):
        run.main(["--rung", "lora", "--adapter", "a", "--data", str(v1_file), "--out", str(tmp_path / "o"),
                  "--prompt-style", "minimal", "--schema", "v2"])
    with pytest.raises(SystemExit, match="no 'los'"):
        run.main(["--rung", "lora", "--adapter", "a", "--data", str(v1_file), "--out", str(tmp_path / "o"),
                  "--prompt-style", "minimal_v2", "--schema", "v2"])
    with pytest.raises(SystemExit, match="no 'los'"):
        run.main(["--rung", "r0los", "--data", str(v1_file), "--out", str(tmp_path / "o")])
    seen = {}

    def fake_make_lora(weights, adapter, merge=False, **kw):
        seen.update(kw)
        raise SystemExit(0)

    import playparse.eval.baselines.lora_predictor as lp

    monkeypatch.setattr(lp, "make_lora", fake_make_lora)
    v2_file = tmp_path / "v2.jsonl"
    v2_file.write_text("".join(json.dumps(r) + "\n" for r in v2_records()))
    with pytest.raises(SystemExit):
        run.main(["--rung", "lora", "--adapter", "a", "--data", str(v2_file), "--out", str(tmp_path / "o"),
                  "--prompt-style", "minimal_v2", "--schema", "v2"])
    assert seen["prompt_style"] == "minimal_v2"


def test_train_sbatch_post_eval_handles_v2():
    text = (Path(__file__).resolve().parent.parent / "scripts" / "slurm" / "train_h100.sbatch").read_text()
    assert "data/processed_v2/eval_lite.jsonl" in text
    assert text.count('--schema "$SCHEMA"') == 2  # both the LoRA and the full-FT branch
    assert "d.get('schema') or 'v1'" in text      # pre-T3b run specs stay on v1
