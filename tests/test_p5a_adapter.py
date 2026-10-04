"""P5a: PEFT adapter loading, name mapping and validation (fast, no engine forward)."""
import json

import numpy as np
import pytest
import torch

from playparse.serving.adapter import (
    AdapterError,
    adapter_from_tensors,
    engine_to_peft_key,
    engine_weight_name,
    expected_linear_shapes,
    load_peft_adapter,
    lora_scale,
    parse_peft_key,
    peft_to_engine_key,
    save_peft_adapter,
    validate_peft_config,
)
from playparse.serving._testing import TINY_CONFIG, build_tiny_checkpoint, make_peft_adapter

BASE_CFG = {"peft_type": "LORA", "r": 2, "lora_alpha": 4, "bias": "none", "use_rslora": False}


def _pair(out_dim, in_dim, r=2, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((r, in_dim)).astype(np.float32), rng.standard_normal((out_dim, r)).astype(np.float32)


def _tensors(modules=("q_proj",), layers=(0,), r=2, shapes=None):
    shapes = shapes or expected_linear_shapes(TINY_CONFIG)
    out = {}
    for layer in layers:
        for m in modules:
            A, B = _pair(*shapes[m], r=r, seed=layer)
            key = engine_weight_name(layer, m)
            out[engine_to_peft_key(key, "A")] = A
            out[engine_to_peft_key(key, "B")] = B
    return out


# ---------------------------------------------------------------- name mapping

@pytest.mark.parametrize("module,block", [
    ("q_proj", "self_attn"), ("k_proj", "self_attn"), ("v_proj", "self_attn"), ("o_proj", "self_attn"),
    ("gate_proj", "mlp"), ("up_proj", "mlp"), ("down_proj", "mlp"),
])
def test_peft_engine_key_roundtrip(module, block):
    peft_key = f"base_model.model.model.layers.11.{block}.{module}.lora_A.weight"
    engine_key, which = peft_to_engine_key(peft_key)
    assert engine_key == f"model.layers.11.{block}.{module}.weight"
    assert which == "A"
    assert engine_to_peft_key(engine_key, which) == peft_key


def test_key_with_adapter_name_segment_and_without_prefix():
    # PEFT's in-memory state_dict keeps the adapter name; some exporters drop the prefix.
    assert parse_peft_key("base_model.model.model.layers.3.mlp.up_proj.lora_B.default.weight") == (3, "up_proj", "B")
    assert parse_peft_key("model.layers.0.self_attn.v_proj.lora_A.weight") == (0, "v_proj", "A")


@pytest.mark.parametrize("key", [
    "base_model.model.lm_head.lora_A.weight",
    "base_model.model.model.embed_tokens.lora_embedding_A",
    "base_model.model.model.layers.0.self_attn.q_proj.lora_magnitude_vector",   # DoRA
    "base_model.model.model.layers.0.self_attn.q_proj.lora_B.bias",             # lora_bias
    "base_model.model.model.layers.0.mlp.q_proj.lora_A.weight",                 # wrong block
    "base_model.model.model.layers.0.self_attn.rotary.lora_A.weight",
])
def test_unservable_keys_rejected(key):
    with pytest.raises(AdapterError):
        parse_peft_key(key)


# ---------------------------------------------------------------- config validation

@pytest.mark.parametrize("override", [
    {"use_dora": True},
    {"fan_in_fan_out": True},
    {"bias": "lora_only"},
    {"lora_bias": True},
    {"modules_to_save": ["lm_head"]},
    {"rank_pattern": {"q_proj": 4}},
    {"alpha_pattern": {"q_proj": 8}},
    {"layer_replication": [[0, 4]]},
    {"init_lora_weights": "pissa"},
    {"init_lora_weights": "olora"},
    {"loftq_config": {"loftq_bits": 4}},
    {"peft_type": "IA3"},
    {"some_future_option": True},
    {"r": 0},
])
def test_unsupported_options_rejected(override):
    with pytest.raises(AdapterError) as exc:
        validate_peft_config({**BASE_CFG, **override})
    assert next(iter(override)) in str(exc.value) or "r=" in str(exc.value)


def test_benign_and_default_options_accepted():
    validate_peft_config({
        **BASE_CFG, "lora_dropout": 0.1, "target_modules": ["q_proj"], "task_type": "CAUSAL_LM",
        "use_dora": False, "modules_to_save": None, "rank_pattern": {}, "loftq_config": {},
        "init_lora_weights": "gaussian", "layers_to_transform": [0], "some_future_option": None,
    })


def test_scale_plain_and_rslora():
    assert lora_scale(8, 16) == 2.0
    assert lora_scale(16, 16, use_rslora=True) == 4.0


# ---------------------------------------------------------------- building adapters

def test_adapter_from_tensors_shapes_and_scale():
    ad = adapter_from_tensors({**BASE_CFG, "use_rslora": True}, _tensors(("q_proj", "down_proj"), (0, 1)),
                              base_config=TINY_CONFIG)
    assert ad.scale == pytest.approx(4 / np.sqrt(2))
    assert len(ad.layers) == 4
    assert ad.target_modules == ("q_proj", "down_proj")
    t = ad.layers["model.layers.1.mlp.down_proj.weight"]
    assert t.shape == (64, 128) and t.r == 2
    np.testing.assert_allclose(t.delta(), ad.scale * (t.B.astype(np.float64) @ t.A), rtol=1e-5, atol=1e-5)


def test_missing_half_rejected():
    tensors = _tensors()
    del tensors[next(k for k in tensors if "lora_B" in k)]
    with pytest.raises(AdapterError, match="lora_B is missing"):
        adapter_from_tensors(BASE_CFG, tensors)


def test_rank_mismatch_rejected():
    with pytest.raises(AdapterError, match="rank mismatch"):
        adapter_from_tensors({**BASE_CFG, "r": 4}, _tensors(r=2))


def test_shape_mismatch_with_base_rejected():
    wrong = expected_linear_shapes({**TINY_CONFIG, "hidden_size": 32, "head_dim": 8})
    with pytest.raises(AdapterError, match="do not fit base weight"):
        adapter_from_tensors(BASE_CFG, _tensors(shapes=wrong), base_config=TINY_CONFIG)


def test_layer_out_of_range_rejected():
    with pytest.raises(AdapterError, match="num_hidden_layers"):
        adapter_from_tensors(BASE_CFG, _tensors(layers=(5,)), base_config=TINY_CONFIG)


def test_lm_head_target_rejected():
    tensors = _tensors()
    tensors["base_model.model.lm_head.lora_A.weight"] = np.zeros((2, 64), np.float32)
    with pytest.raises(AdapterError, match="lm_head"):
        adapter_from_tensors(BASE_CFG, tensors)


def test_bf16_tensors_become_fp32():
    tensors = {k: torch.from_numpy(v).to(torch.bfloat16) for k, v in _tensors().items()}
    ad = adapter_from_tensors(BASE_CFG, tensors)
    t = next(iter(ad.layers.values()))
    assert t.A.dtype == np.float32 and t.B.dtype == np.float32


def test_zero_B_copy():
    ad = adapter_from_tensors(BASE_CFG, _tensors())
    z = ad.with_zero_B()
    assert all(not t.B.any() for t in z.layers.values())
    assert all(t.A.any() for t in z.layers.values())


# ---------------------------------------------------------------- real PEFT files

def test_load_adapter_written_by_peft(tmp_path):
    hf = build_tiny_checkpoint(tmp_path / "base")
    peft_model = make_peft_adapter(hf, tmp_path / "ad", r=4, lora_alpha=8, target_modules=("q_proj", "v_proj"))
    cfg = json.loads((tmp_path / "base" / "config.json").read_text())
    ad = load_peft_adapter(tmp_path / "ad", base_config=cfg)
    assert ad.r == 4 and ad.scale == 2.0
    assert set(ad.target_modules) == {"q_proj", "v_proj"}
    assert len(ad.layers) == 2 * TINY_CONFIG["num_hidden_layers"]
    # Same numbers PEFT holds in memory.
    sd = {k: v for k, v in peft_model.state_dict().items() if "lora_" in k}
    A_mem = sd["base_model.model.model.layers.1.self_attn.v_proj.lora_A.default.weight"].numpy()
    np.testing.assert_array_equal(ad.layers["model.layers.1.self_attn.v_proj.weight"].A, A_mem)


def test_save_load_roundtrip(tmp_path):
    ad = adapter_from_tensors(BASE_CFG, _tensors(("k_proj", "gate_proj"), (0, 1)), name="x")
    save_peft_adapter(ad, tmp_path / "x")
    back = load_peft_adapter(tmp_path / "x", base_config=TINY_CONFIG)
    assert set(back.layers) == set(ad.layers)
    for k in ad.layers:
        np.testing.assert_array_equal(back.layers[k].A, ad.layers[k].A)
        np.testing.assert_array_equal(back.layers[k].B, ad.layers[k].B)


def test_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_peft_adapter(tmp_path)
    (tmp_path / "adapter_config.json").write_text(json.dumps(BASE_CFG))
    with pytest.raises(FileNotFoundError, match="safetensors"):
        load_peft_adapter(tmp_path)
