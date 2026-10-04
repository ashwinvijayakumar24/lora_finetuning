"""P5a oracle on a tiny random Llama: engine + adapter vs HF + PEFT (fast, CPU).

The same comparisons as the slow real-model suite, on a 2-layer model saved in
HF format, so a regression in the merge / unmerged paths fails in seconds.
"""
import numpy as np
import pytest
import torch

from playparse.serving._testing import build_tiny_checkpoint, make_peft_adapter
from playparse.serving.adapter import load_peft_adapter
from playparse.serving.lora_engine import (
    LoRALlamaModel,
    LoRAModelGPU,
    generate_with_adapter,
)
from playparse.serving.merge import merge_adapter, merge_then_quantize

from engine.loader import load_config, load_weights, load_weights_gpu
from engine.model import LlamaModel
from engine.model_gpu import LlamaModelGPU
from engine.sampler import greedy
from engine.scheduler import generate

PROMPT = [1, 17, 42, 99, 3, 250, 7, 64, 128, 5]
FP32_ATOL = 1e-4      # engine fp32 NumPy vs HF fp32: observed ~2e-6
FP16_ATOL = 5e-2      # engine fp16 torch vs HF fp32: observed ~7e-3


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("p5a_tiny")
    hf = build_tiny_checkpoint(root / "base", seed=0)
    ids = torch.tensor([PROMPT])
    refs = {}
    with torch.no_grad():
        refs["base"] = hf(ids).logits[0].numpy()
        for name, kw in {
            "all": dict(r=8, lora_alpha=16, b_std=0.1, seed=1),
            "qv": dict(r=4, lora_alpha=8, b_std=0.2, seed=2, target_modules=("q_proj", "v_proj")),
            "rslora": dict(r=16, lora_alpha=8, b_std=0.1, seed=3, use_rslora=True),
        }.items():
            pm = make_peft_adapter(hf, root / name, **kw)
            refs[name] = pm(ids).logits[0].numpy()
            refs[name + "_greedy"] = pm.generate(
                ids, max_new_tokens=8, do_sample=False, pad_token_id=0,
            )[0, len(PROMPT):].tolist()
            hf = pm.unload()
    cfg = load_config(root / "base")
    adapters = {n: load_peft_adapter(root / n, name=n, base_config=cfg) for n in ("all", "qv", "rslora")}
    return dict(root=root, cfg=cfg, refs=refs, adapters=adapters,
                np_weights=load_weights(root / "base", cfg),
                t_weights=load_weights_gpu(root / "base", cfg, device="cpu"))


def test_adapter_changes_output(tiny):
    # Guards against a vacuous oracle: B != 0 must move the logits noticeably.
    for n in ("all", "qv", "rslora"):
        assert np.abs(tiny["refs"][n] - tiny["refs"]["base"]).max() > 0.1


@pytest.mark.parametrize("name", ["all", "qv", "rslora"])
def test_numpy_merged_matches_peft(tiny, name):
    model = LlamaModel(merge_adapter(tiny["np_weights"], tiny["adapters"][name]), tiny["cfg"])
    np.testing.assert_allclose(model.forward(PROMPT), tiny["refs"][name], atol=FP32_ATOL, rtol=0)


@pytest.mark.parametrize("name", ["all", "qv", "rslora"])
def test_numpy_unmerged_matches_peft(tiny, name):
    model = LoRALlamaModel(tiny["np_weights"], tiny["cfg"])
    model.load_adapter(tiny["adapters"][name])
    np.testing.assert_allclose(model.forward(PROMPT, adapter=name), tiny["refs"][name], atol=FP32_ATOL, rtol=0)


def test_numpy_one_model_switches_adapters_per_call(tiny):
    model = LoRALlamaModel(tiny["np_weights"], tiny["cfg"])
    for n in ("all", "qv"):
        model.load_adapter(tiny["adapters"][n])
    for n in ("qv", "all", None, "qv"):
        ref = tiny["refs"][n or "base"]
        np.testing.assert_allclose(model.forward(PROMPT, adapter=n), ref, atol=FP32_ATOL, rtol=0)


def test_base_equals_zero_adapter_exactly(tiny):
    base = LlamaModel(tiny["np_weights"], tiny["cfg"]).forward(PROMPT)
    zero = tiny["adapters"]["all"].with_zero_B("zero")
    unmerged = LoRALlamaModel(tiny["np_weights"], tiny["cfg"])
    unmerged.load_adapter(zero)
    assert np.array_equal(unmerged.forward(PROMPT, adapter="zero"), base)
    merged = LlamaModel(merge_adapter(tiny["np_weights"], zero), tiny["cfg"])
    assert np.array_equal(merged.forward(PROMPT), base)


def test_unload_restores_plain_weights(tiny):
    model = LoRALlamaModel(tiny["np_weights"], tiny["cfg"])
    model.load_adapter(tiny["adapters"]["qv"])
    model.unload_adapter("qv")
    assert all(model.weights[k] is tiny["np_weights"][k] for k in tiny["np_weights"])


def test_load_adapter_does_not_touch_callers_dict(tiny):
    weights = dict(tiny["np_weights"])
    LoRALlamaModel(weights, tiny["cfg"]).load_adapter(tiny["adapters"]["all"])
    assert all(type(v) is np.ndarray for v in weights.values())


def test_torch_fp16_merged_and_unmerged_match_peft(tiny):
    for name in ("all", "qv", "rslora"):
        merged = LlamaModelGPU(merge_adapter(tiny["t_weights"], tiny["adapters"][name]), tiny["cfg"], device="cpu")
        unmerged = LoRAModelGPU(tiny["t_weights"], tiny["cfg"], device="cpu")
        unmerged.load_adapter(tiny["adapters"][name])
        lm = merged.forward_all(PROMPT)
        lu = unmerged.forward_all(PROMPT, adapter=name)
        np.testing.assert_allclose(lm, tiny["refs"][name], atol=FP16_ATOL, rtol=0)
        np.testing.assert_allclose(lu, tiny["refs"][name], atol=FP16_ATOL, rtol=0)
        np.testing.assert_allclose(lu, lm, atol=FP16_ATOL, rtol=0)


@pytest.mark.parametrize("name", ["all", "qv", "rslora"])
def test_greedy_tokens_merged_equal_unmerged_equal_peft(tiny, name):
    """Claim L2 in miniature: same greedy tokens from merged, unmerged and HF+PEFT."""
    ad = tiny["adapters"][name]
    np_merged = LlamaModel(merge_adapter(tiny["np_weights"], ad), tiny["cfg"])
    np_unmerged = LoRALlamaModel(tiny["np_weights"], tiny["cfg"])
    np_unmerged.load_adapter(ad)
    t_merged = LlamaModelGPU(merge_adapter(tiny["t_weights"], ad), tiny["cfg"], device="cpu")
    t_unmerged = LoRAModelGPU(tiny["t_weights"], tiny["cfg"], device="cpu")
    t_unmerged.load_adapter(ad)

    kw = dict(max_tokens=8, max_seq=64)
    expected = tiny["refs"][name + "_greedy"]
    assert list(generate(np_merged, PROMPT, greedy, **kw)) == expected
    assert list(generate_with_adapter(np_unmerged, PROMPT, greedy, name, **kw)) == expected
    assert list(generate(t_merged, PROMPT, greedy, **kw)) == expected
    with t_unmerged.use_adapter(name):
        assert list(generate(t_unmerged, PROMPT, greedy, **kw)) == expected


def test_unmerged_on_int8_base_close_to_merged_then_quantized(tiny):
    """Both quantized routes land near the fp16 adapter output; neither is exact."""
    from playparse.serving.merge import quantize_engine_weights

    ad = tiny["adapters"]["all"]
    mq = LlamaModelGPU(merge_then_quantize(tiny["t_weights"], ad, mode="int8"), tiny["cfg"], device="cpu")
    uq = LoRAModelGPU(quantize_engine_weights(tiny["t_weights"], mode="int8"), tiny["cfg"], device="cpu")
    uq.load_adapter(ad)
    ref = tiny["refs"]["all"]
    err_m = np.abs(mq.forward_all(PROMPT) - ref).max()
    err_u = np.abs(uq.forward_all(PROMPT, adapter="all") - ref).max()
    assert err_m < 0.5 and err_u < 0.5, (err_m, err_u)
    assert np.abs(uq.forward_all(PROMPT, adapter="all") - uq.forward_all(PROMPT, adapter=None)).max() > 0.1


def test_per_row_selection_through_forward_all(tiny):
    """One packed forward with token rows on different adapters == separate runs.

    Uses forward_all (no KV cache), where every row is one position of one
    sequence; mixing adapters across rows of the *same* sequence is not
    meaningful for generation, but it exercises the per-row linear path inside a
    full model, which is exactly what P5b's packed varlen batch will do.
    """
    from playparse.serving.lora_engine import AdapterSelection

    model = LoRAModelGPU(tiny["t_weights"], tiny["cfg"], device="cpu")
    model.load_adapter(tiny["adapters"]["all"])
    all_rows = AdapterSelection.per_row(["all"], torch.zeros(len(PROMPT), dtype=torch.long))
    np.testing.assert_allclose(model.forward_all(PROMPT, adapter=all_rows),
                               model.forward_all(PROMPT, adapter="all"), atol=1e-2, rtol=0)
    none_rows = AdapterSelection.per_row(["all"], -torch.ones(len(PROMPT), dtype=torch.long))
    assert np.array_equal(model.forward_all(PROMPT, adapter=none_rows), model.forward_all(PROMPT, adapter=None))


def test_generator_consumed_outside_block_falls_back_to_base(tiny):
    """Documents the pitfall generate_with_adapter exists for (docs/issues/p5a-generator-context.md)."""
    model = LoRAModelGPU(tiny["t_weights"], tiny["cfg"], device="cpu")
    model.load_adapter(tiny["adapters"]["all"])
    kw = dict(max_tokens=8, max_seq=64)
    with model.use_adapter(None):
        base_tokens = list(generate(model, PROMPT, greedy, **kw))
    with model.use_adapter("all"):
        adapter_tokens = list(generate(model, PROMPT, greedy, **kw))
        lazy = generate(model, PROMPT, greedy, **kw)            # created inside ...
        safe = generate_with_adapter(model, PROMPT, greedy, "all", **kw)
    assert adapter_tokens != base_tokens
    assert list(lazy) == base_tokens                             # ... consumed outside: base model
    assert list(safe) == adapter_tokens


def test_tiny_checkpoint_config_is_engine_readable(tiny):
    """transformers 5 drops rope_theta/rope_scaling from config.json; the engine needs them."""
    cfg = tiny["cfg"]
    assert cfg["rope_theta"] == 500000.0
    assert cfg["rope_scaling"]["rope_type"] == "llama3"


def test_unknown_adapter_name_rejected(tiny):
    from playparse.serving.adapter import AdapterError

    model = LoRAModelGPU(tiny["t_weights"], tiny["cfg"], device="cpu")
    with pytest.raises(AdapterError, match="not loaded"):
        model.forward_all(PROMPT, adapter="nope")
