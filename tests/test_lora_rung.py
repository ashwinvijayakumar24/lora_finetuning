"""The LoRA rung in the eval harness, on a tiny Llama saved next to the real tokenizer.

Checks: the rung runs end to end through the CLI; the adapter changes outputs;
merged and unmerged give the same greedy tokens; and the harness's HF-generate
path agrees token for token with playparse.train.generate (the path the training
val callback uses), so in-training and post-training scores are comparable.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from p2_fixtures import load_tokenizer, tiny_llama
from playparse.eval.harness import load_records
from playparse.lora import LoRAConfig, inject_lora, save_adapter

FIXTURE = Path(__file__).parent / "fixtures" / "eval_mini.jsonl"
STOP_TOKENS = ["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"]


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    """A tiny random Llama with the real tokenizer and Llama 3 stop tokens."""
    tok = load_tokenizer()
    d = tmp_path_factory.mktemp("tiny_base")
    model = tiny_llama(vocab=len(tok), hidden=32, layers=2, seed=0, embed_std=1.0)
    stop = [tok.convert_tokens_to_ids(t) for t in STOP_TOKENS]
    model.config.eos_token_id = stop
    model.generation_config.eos_token_id = stop
    model.generation_config.bos_token_id = tok.bos_token_id
    model.save_pretrained(d)
    tok.save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def adapter_dir(tiny_dir, tmp_path_factory):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(tiny_dir, dtype=torch.float32)
    cfg = LoRAConfig(r=4, alpha=8, dropout=0.0)
    inject_lora(model, cfg)
    torch.manual_seed(3)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.5)  # large enough that the adapter visibly changes greedy outputs
    return save_adapter(model, cfg, tmp_path_factory.mktemp("adapter") / "adapter")


def _predict(pred, records):
    return [p.text for p in pred.predict_batch(records)]


def test_adapter_changes_outputs_and_merge_is_identical(tiny_dir, adapter_dir):
    from playparse.eval.baselines.hf_predictor import make_r1
    from playparse.eval.baselines.lora_predictor import make_lora

    recs = load_records(FIXTURE)[:4]
    kw = dict(device="cpu", dtype="float32", batch_size=4, max_new_tokens=12)
    base = _predict(make_r1(tiny_dir, **kw), recs)
    unmerged_pred = make_lora(tiny_dir, adapter_dir, **kw)
    merged_pred = make_lora(tiny_dir, adapter_dir, merge=True, **kw)
    unmerged, merged = _predict(unmerged_pred, recs), _predict(merged_pred, recs)
    assert unmerged != base
    assert merged == unmerged
    from playparse.lora import LoRALinear

    assert not any(isinstance(m, LoRALinear) for m in merged_pred.model.modules())
    cu, cm = unmerged_pred.config(), merged_pred.config()
    assert cu["merged"] is False and cm["merged"] is True
    assert cu["adapter_sha256"] == cm["adapter_sha256"] and cu["adapter_r"] == 4


def test_harness_decoding_matches_training_generate(tiny_dir, adapter_dir):
    from playparse.eval.baselines.lora_predictor import make_lora
    from playparse.train.collate import encode_prompt
    from playparse.train.generate import greedy_generate

    recs = load_records(FIXTURE)[:4]
    pred = make_lora(tiny_dir, adapter_dir, device="cpu", dtype="float32", batch_size=4, max_new_tokens=12)
    harness = _predict(pred, recs)
    tok = pred.tokenizer
    prompts = [encode_prompt(tok, r.get("posteam"), r["desc"]) for r in recs]
    gens = greedy_generate(pred.model, tok, prompts, max_new_tokens=12, batch_size=4, autocast="none")
    assert harness == [tok.decode(g.token_ids, skip_special_tokens=True) for g in gens]


def test_cli_lora_rung_end_to_end(tiny_dir, adapter_dir, tmp_path):
    from playparse.eval.run import main

    out = tmp_path / "res"
    main(["--rung", "lora", "--adapter", str(adapter_dir), "--weights", str(tiny_dir), "--data", str(FIXTURE),
          "--out", str(out), "--device", "cpu", "--dtype", "float32", "--max-new-tokens", "6", "--batch-size", "4",
          "--n-boot", "50"])
    res = json.loads((out / "result.json").read_text())
    assert res["meta"]["rung"] == "lora"
    assert res["meta"]["config"]["adapter_sha256"]
    assert res["meta"]["adapter_path"] == str(adapter_dir)
    assert res["overall"]["n"] == len(load_records(FIXTURE))


def test_cli_lora_requires_adapter(tmp_path):
    from playparse.eval.run import main

    with pytest.raises(SystemExit):
        main(["--rung", "lora", "--data", str(FIXTURE), "--out", str(tmp_path)])
