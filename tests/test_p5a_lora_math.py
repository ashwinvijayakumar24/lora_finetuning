"""P5a: merge and unmerged LoRA math on small tensors (fast, CPU)."""
import threading

import numpy as np
import pytest
import torch

from playparse.serving.adapter import AdapterError, LoRATensors, adapter_from_tensors, engine_to_peft_key
from playparse.serving.lora_engine import (
    AdapterSelection,
    LoRALinear,
    current_selection,
    install_lora_linear,
    lora_linear,
    uninstall_lora_linear,
    use_adapter,
)
from playparse.serving.merge import merge_adapter, merge_then_quantize, quantize_engine_weights

from engine import components_gpu
from engine.quant import QuantWeight, quantize_int8_perchannel

OUT, IN, R = 24, 32, 4
KEY = "model.layers.0.self_attn.q_proj.weight"


def _rand(*shape, seed=0, scale=1.0):
    return (np.random.default_rng(seed).standard_normal(shape) * scale).astype(np.float32)


def _adapter(seed=1, name="a", key=KEY, out=OUT, inp=IN, r=R, alpha=8):
    A, B = _rand(r, inp, seed=seed), _rand(out, r, seed=seed + 100, scale=0.1)
    cfg = {"peft_type": "LORA", "r": r, "lora_alpha": alpha, "bias": "none"}
    return adapter_from_tensors(cfg, {engine_to_peft_key(key, "A"): A, engine_to_peft_key(key, "B"): B}, name=name)


def _reference(x, W, t: LoRATensors):
    x64 = x.astype(np.float64)
    return x64 @ W.astype(np.float64).T + t.scale * (x64 @ t.A.T.astype(np.float64)) @ t.B.T.astype(np.float64)


# ---------------------------------------------------------------- merge

def test_merge_numpy_matches_formula_and_keeps_dtype():
    W = _rand(OUT, IN)
    ad = _adapter()
    t = ad.layers[KEY]
    merged = merge_adapter({KEY: W}, ad)[KEY]
    assert merged.dtype == np.float32 and merged.shape == W.shape
    np.testing.assert_allclose(merged, W + t.scale * t.B @ t.A, rtol=1e-6, atol=1e-6)


def test_merge_does_not_mutate_input_unless_inplace():
    W = _rand(OUT, IN)
    weights = {KEY: W}
    merge_adapter(weights, _adapter())
    assert weights[KEY] is W
    merge_adapter(weights, _adapter(), inplace=True)
    assert weights[KEY] is not W


def test_merge_torch_fp16_rounds_once():
    W = torch.from_numpy(_rand(OUT, IN)).half()
    ad = _adapter()
    t = ad.layers[KEY]
    merged = merge_adapter({KEY: W}, ad)[KEY]
    assert merged.dtype == torch.float16
    exact = W.float() + t.scale * torch.from_numpy(t.B) @ torch.from_numpy(t.A)
    assert torch.equal(merged, exact.half())


def test_merge_preserves_tied_embedding_alias():
    emb = _rand(10, IN)
    weights = {KEY: _rand(OUT, IN), "model.embed_tokens.weight": emb, "lm_head.weight": emb}
    merged = merge_adapter(weights, _adapter())
    assert merged["lm_head.weight"] is merged["model.embed_tokens.weight"] is emb


def test_merge_zero_B_is_bit_identical():
    W = torch.from_numpy(_rand(OUT, IN)).half()
    merged = merge_adapter({KEY: W}, _adapter().with_zero_B())[KEY]
    assert torch.equal(merged, W)


def test_merge_refuses_quantized_base():
    q, s = quantize_int8_perchannel(torch.from_numpy(_rand(OUT, IN)))
    with pytest.raises(AdapterError, match="quantize afterwards"):
        merge_adapter({KEY: QuantWeight(q, s, "int8")}, _adapter())


def test_merge_then_quantize_equals_quantizing_merged_weight():
    W = torch.from_numpy(_rand(OUT, IN)).half()
    weights = {KEY: W, "model.norm.weight": torch.ones(IN).half()}
    out = merge_then_quantize(weights, _adapter(), mode="int8")
    assert isinstance(out[KEY], QuantWeight)
    assert not isinstance(out["model.norm.weight"], QuantWeight)
    expected_q, _ = quantize_int8_perchannel(merge_adapter(weights, _adapter())[KEY])
    assert torch.equal(out[KEY].q, expected_q)


def test_quantize_engine_weights_int4():
    weights = {KEY: torch.from_numpy(_rand(OUT, IN)).half()}
    out = quantize_engine_weights(weights, mode="int4", group_size=16)
    assert out[KEY].mode == "int4"


# ---------------------------------------------------------------- unmerged, NumPy path

def test_numpy_unmerged_matches_formula():
    W, x = _rand(OUT, IN), _rand(5, IN, seed=7)
    ad = _adapter()
    w = LoRALinear(KEY, W)
    w.add("a", ad.layers[KEY].A, ad.layers[KEY].B, ad.scale)
    with use_adapter("a"):
        y = x @ w.T          # exactly how engine/components.py spells a projection
    np.testing.assert_allclose(y, _reference(x, W, ad.layers[KEY]), rtol=1e-5, atol=1e-5)


def test_numpy_unselected_is_bit_identical_to_base():
    W, x = _rand(OUT, IN), _rand(5, IN, seed=7)
    w = LoRALinear(KEY, W)
    ad = _adapter()
    w.add("a", ad.layers[KEY].A, ad.layers[KEY].B, ad.scale)
    assert np.array_equal(x @ w.T, x @ W.T)


def test_numpy_per_row_matches_single():
    W, x = _rand(OUT, IN), _rand(6, IN, seed=7)
    a, b = _adapter(seed=1, name="a"), _adapter(seed=2, name="b")
    w = LoRALinear(KEY, W)
    for ad in (a, b):
        w.add(ad.name, ad.layers[KEY].A, ad.layers[KEY].B, ad.scale)
    slots = np.array([0, 1, -1, 1, 0, -1])
    with use_adapter(AdapterSelection.per_row(["a", "b"], slots)):
        y = x @ w.T
    for i, s in enumerate(slots):
        sel = None if s < 0 else ["a", "b"][s]
        with use_adapter(sel):
            np.testing.assert_allclose(y[i], (x[i:i + 1] @ w.T)[0], rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------- unmerged, torch path (the linear() chokepoint)

@pytest.fixture
def installed():
    install_lora_linear()
    yield
    # leave installed: LoRAModelGPU instances in other tests rely on it; idempotent anyway


def _torch_lora(W, ads, quant=False):
    base = torch.from_numpy(W).half()
    if quant:
        q, s = quantize_int8_perchannel(base)
        base = QuantWeight(q, s, "int8")
    w = LoRALinear(KEY, base)
    for ad in ads:
        w.add(ad.name, ad.layers[KEY].A, ad.layers[KEY].B, ad.scale)
    return w


def test_install_is_idempotent_and_reversible():
    install_lora_linear()
    install_lora_linear()
    assert components_gpu.linear is lora_linear
    uninstall_lora_linear()
    assert components_gpu.linear is not lora_linear
    assert components_gpu.linear.__module__ == "engine.components_gpu"
    install_lora_linear()


def test_plain_weights_bit_identical_through_replacement(installed):
    W = torch.from_numpy(_rand(OUT, IN)).half()
    x = torch.from_numpy(_rand(3, IN, seed=5)).half()
    assert torch.equal(components_gpu.linear(x, W), x @ W.T)
    q, s = quantize_int8_perchannel(W)
    qw = QuantWeight(q, s, "int8")
    assert torch.equal(components_gpu.linear(x, qw), x @ qw.dequantize().T)


def test_torch_unmerged_matches_formula(installed):
    W, x = _rand(OUT, IN), _rand(5, IN, seed=7)
    ad = _adapter()
    w = _torch_lora(W, [ad])
    with use_adapter("a"):
        y = components_gpu.linear(torch.from_numpy(x).half(), w).float().numpy()
    np.testing.assert_allclose(y, _reference(x, W, ad.layers[KEY]), atol=2e-2, rtol=1e-2)


def test_torch_unmerged_on_int8_base(installed):
    """Quantized base + fp16 adapter: y = x @ dequant(Wq).T + low-rank path."""
    W, x = _rand(OUT, IN), torch.from_numpy(_rand(5, IN, seed=7)).half()
    ad = _adapter()
    w = _torch_lora(W, [ad], quant=True)
    with use_adapter("a"):
        y = components_gpu.linear(x, w)
    A, B = w.adapters["a"]
    expected = x @ w.base.dequantize().T + (x @ A.T) @ B.T
    assert torch.equal(y, expected)


def test_torch_no_selection_and_untargeted_adapter_are_base(installed):
    W, x = _rand(OUT, IN), torch.from_numpy(_rand(5, IN, seed=7)).half()
    w = _torch_lora(W, [_adapter()])
    base = x @ w.base.T
    assert torch.equal(components_gpu.linear(x, w), base)
    with use_adapter("not-on-this-projection"):
        assert torch.equal(components_gpu.linear(x, w), base)


def test_torch_per_row_matches_single(installed):
    W, x = _rand(OUT, IN), torch.from_numpy(_rand(6, IN, seed=7)).half()
    w = _torch_lora(W, [_adapter(seed=1, name="a"), _adapter(seed=2, name="b")])
    slots = torch.tensor([1, 1, -1, 0, 0, 1])
    with use_adapter(AdapterSelection.per_row(["a", "b"], slots)):
        y = components_gpu.linear(x, w)
    for i, s in enumerate(slots.tolist()):
        with use_adapter(None if s < 0 else ["a", "b"][s]):
            torch.testing.assert_close(y[i], components_gpu.linear(x[i:i + 1], w)[0], atol=1e-3, rtol=1e-3)


def test_torch_transpose_without_install_fails_loudly():
    w = _torch_lora(_rand(OUT, IN), [_adapter()])
    with pytest.raises(RuntimeError, match="install_lora_linear"):
        _ = w.T


def test_selection_is_per_thread():
    seen = {}

    def worker():
        seen["thread"] = current_selection()

    with use_adapter("a"):
        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert current_selection().name == "a"
    assert seen["thread"] is None
    assert current_selection() is None
