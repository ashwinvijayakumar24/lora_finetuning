"""scripts/train.py end to end on a tiny model saved next to the real tokenizer."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from p2_fixtures import load_tokenizer, tiny_llama
from playparse.ffscore.schema import PlayLabel
from playparse.train.build import RunSpec
from playparse.train.synthetic import synth_records

REPO = Path(__file__).resolve().parent.parent


def _load_script():
    spec = importlib.util.spec_from_file_location("train_script", REPO / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_config_parses():
    spec = RunSpec.from_file(REPO / "configs" / "train_default.yaml")
    assert spec.lora.r == 16 and spec.lora.alpha == 32
    assert spec.train.examples_per_step == 64


def test_synthetic_records_are_valid():
    recs = synth_records(50, seed=1)
    assert len({r["bucket"] for r in recs}) >= 4
    for r in recs:
        PlayLabel.from_json(r["label"])  # schema-valid


@pytest.fixture(scope="module")
def tiny_weights(tmp_path_factory):
    tok = load_tokenizer()
    d = tmp_path_factory.mktemp("tiny_weights")
    tiny_llama(vocab=len(tok), hidden=16, layers=1).save_pretrained(d)
    tok.save_pretrained(d)
    return d


def test_cli_trains_and_resumes(tmp_path, tiny_weights):
    for name, n, seed in [("train", 12, 0), ("val", 4, 1)]:
        with open(tmp_path / f"{name}.jsonl", "w") as f:
            for r in synth_records(n, seed):
                f.write(json.dumps(r) + "\n")
    out = tmp_path / "run"
    common = [
        f"model.weights={json.dumps(str(tiny_weights))}", "model.dtype=fp32", "model.attn_implementation=eager",
        "lora.r=4", f"data.train={json.dumps(str(tmp_path / 'train.jsonl'))}",
        f"data.val={json.dumps(str(tmp_path / 'val.jsonl'))}", f"train.output_dir={json.dumps(str(out))}",
        "train.device=cpu", "train.micro_batch_size=2", "train.grad_accum_steps=2", "train.eval_every=2",
        "train.save_every=2", "train.gen_every=4", "gen_eval_examples=2", "gen_max_new_tokens=4",
        "data.val_loss_examples=3",
    ]
    script = _load_script()
    assert script.main(["--config", str(REPO / "configs" / "train_default.yaml"), "--set", *common,
                        "train.max_steps=4", "train.epochs=null"]) == 0
    result = json.loads((out / "result.json").read_text())
    assert result["step"] == 4
    assert (out / "checkpoints" / "step_0000004" / "adapter" / "adapter_model.safetensors").exists()
    assert (out / "checkpoints" / "step_0000004" / "adapter" / "adapter_config.json").exists()  # PEFT format
    recs = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines()]
    assert any(r["event"] == "val_callback" and "val_exact_match" in r and "val_exact_match_natural" in r
               and any(k.startswith("val_exact_match/") for k in r) for r in recs)
    assert (out / "val_predictions" / "step_0000004.jsonl").exists()
    meta = json.loads((out / "run_meta.json").read_text())
    assert len(meta["val_loss_keys"]) == 3 and len(meta["gen_eval_keys"]) == 2

    # Resume from the saved spec and extend the run to 6 steps.
    assert script.main(["--config", str(out / "run_spec.json"), "--resume", "latest",
                        "--set", "train.max_steps=6"]) == 0
    assert json.loads((out / "result.json").read_text())["step"] == 6
