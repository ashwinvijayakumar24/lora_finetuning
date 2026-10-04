"""Injection tests, with PEFT as the oracle for logits and gradients."""
import pytest
import torch
from torch import nn

from playparse.lora import (
    LoRAConfig,
    LoRALinear,
    count_trainable,
    inject_lora,
    load_lora_state_dict,
    lora_modules,
    lora_state_dict,
    merge_lora,
    unload_lora,
)
from tests._lora_util import (
    copy_ours_to_peft,
    input_ids,
    peft_config,
    peft_lora_params,
    randomize_lora,
    tiny_llama,
    tiny_pair,
)

peft = pytest.importorskip("peft")


def _wrapped(model) -> set[str]:
    return {n for n, _ in lora_modules(model)}


def test_all_linear_selects_seven_projections_per_layer_not_lm_head():
    model = inject_lora(tiny_llama(), LoRAConfig(r=4, alpha=8, dropout=0.0))
    names = _wrapped(model)
    assert len(names) == 2 * 7
    leaves = {n.rsplit(".", 1)[-1] for n in names}
    assert leaves == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert isinstance(model.lm_head, nn.Linear) and not isinstance(model.lm_head, LoRALinear)


def test_list_targets_select_only_q_and_v():
    model = inject_lora(tiny_llama(), LoRAConfig(r=4, alpha=8, target_modules=["q_proj", "v_proj"]))
    assert _wrapped(model) == {
        f"model.layers.{i}.self_attn.{p}" for i in range(2) for p in ("q_proj", "v_proj")
    }


def test_target_selection_matches_peft():
    for targets in ("all-linear", ["q_proj", "v_proj"], ["q_proj", "k_proj", "v_proj", "o_proj"]):
        cfg = LoRAConfig(r=4, alpha=8, dropout=0.0, target_modules=targets)
        ours, theirs = tiny_pair()
        inject_lora(ours, cfg)
        pm = peft.get_peft_model(theirs, peft_config(cfg))
        assert set(lora_state_dict(ours)) == set(peft_lora_params(pm)), targets


def test_only_lora_params_require_grad():
    model = inject_lora(tiny_llama(), LoRAConfig(r=4, alpha=8))
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.endswith(("lora_A.weight", "lora_B.weight")) for n in trainable)
    assert len(trainable) == 2 * 14


def test_count_trainable_matches_formula():
    model = inject_lora(tiny_llama(), LoRAConfig(r=4, alpha=8, target_modules=["q_proj", "v_proj"]))
    # q: 64->64, v: 64->32 (2 kv heads x head_dim 16); each adapter has r*(in+out) params
    per_layer = 4 * (64 + 64) + 4 * (64 + 32)
    assert count_trainable(model) == 2 * per_layer


def test_zero_init_output_equals_base_exactly():
    base, wrapped = tiny_pair()
    inject_lora(wrapped, LoRAConfig(r=8, alpha=16, dropout=0.0))
    ids = input_ids()
    with torch.no_grad():
        assert torch.equal(wrapped(ids).logits, base(ids).logits)


@pytest.mark.parametrize("targets", ["all-linear", ["q_proj", "v_proj"]])
def test_logits_match_peft_fp32(targets):
    cfg = LoRAConfig(r=4, alpha=8, dropout=0.0, target_modules=targets)
    ours, theirs = tiny_pair()
    inject_lora(ours, cfg)
    pm = peft.get_peft_model(theirs, peft_config(cfg)).eval()
    randomize_lora(ours)
    copy_ours_to_peft(ours, pm)
    ids = input_ids()
    with torch.no_grad():
        a, b = ours(ids).logits, pm(input_ids=ids).logits
    assert (a - b).abs().max().item() < 1e-5
    # guard against a vacuous pass: the adapter must actually change the output
    with torch.no_grad():
        assert (a - tiny_llama()(ids).logits).abs().max().item() > 1e-3


def test_gradients_match_peft_after_one_backward():
    cfg = LoRAConfig(r=4, alpha=8, dropout=0.0)
    ours, theirs = tiny_pair()
    inject_lora(ours, cfg)
    pm = peft.get_peft_model(theirs, peft_config(cfg))
    randomize_lora(ours)
    copy_ours_to_peft(ours, pm)
    ids = input_ids()
    ours(input_ids=ids, labels=ids).loss.backward()
    pm(input_ids=ids, labels=ids).loss.backward()
    mine = {f"{n}.lora_{ab}.weight": getattr(m, f"lora_{ab}").weight for n, m in lora_modules(ours) for ab in "AB"}
    theirs_p = peft_lora_params(pm)
    for k, p in mine.items():
        assert p.grad is not None and p.grad.abs().max() > 0, k
        torch.testing.assert_close(p.grad, theirs_p[k].grad, rtol=1e-5, atol=1e-6, msg=k)
    assert all(p.grad is None for n, p in ours.named_parameters() if "lora_" not in n)


def test_dropout_off_in_eval_mode():
    model = inject_lora(tiny_llama(), LoRAConfig(r=4, alpha=8, dropout=0.5))
    randomize_lora(model)
    ids = input_ids()
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(ids).logits, model(ids).logits)
        model.train()
        torch.manual_seed(0)
        y1 = model(ids).logits
        torch.manual_seed(1)
        y2 = model(ids).logits
    assert not torch.allclose(y1, y2)


def test_inject_into_eval_model_keeps_dropout_off():
    """Guard for docs/issues/p0-injected-layers-start-in-train-mode.md."""
    model = tiny_llama()  # already .eval()
    inject_lora(model, LoRAConfig(r=4, alpha=8, dropout=0.5))
    assert all(not m.training for _, m in lora_modules(model))
    randomize_lora(model)
    ids = input_ids()
    with torch.no_grad():
        assert torch.equal(model(ids).logits, model(ids).logits)


def test_merge_lora_equals_unmerged_and_returns_plain_model():
    model = inject_lora(tiny_llama(), LoRAConfig(r=4, alpha=8, dropout=0.0))
    randomize_lora(model)
    ids = input_ids()
    with torch.no_grad():
        before = model(ids).logits
        merged = merge_lora(model)
        after = merged(ids).logits
    assert not any(isinstance(m, LoRALinear) for m in merged.modules())
    assert type(merged.model.layers[0].self_attn.q_proj) is nn.Linear
    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-5)


def test_unload_lora_restores_base_exactly():
    base, model = tiny_pair()
    inject_lora(model, LoRAConfig(r=4, alpha=8, dropout=0.0))
    randomize_lora(model)
    model.model.layers[0].mlp.up_proj.merge()  # unload must undo merges too
    unload_lora(model)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    ids = input_ids()
    with torch.no_grad():
        torch.testing.assert_close(model(ids).logits, base(ids).logits, rtol=1e-6, atol=1e-6)


def test_inject_twice_or_no_match_raises():
    model = inject_lora(tiny_llama(), LoRAConfig(r=4, alpha=8))
    with pytest.raises(RuntimeError):
        inject_lora(model, LoRAConfig(r=4, alpha=8))
    with pytest.raises(ValueError):
        inject_lora(tiny_llama(), LoRAConfig(r=4, alpha=8, target_modules=["nope_proj"]))


def test_lora_state_dict_roundtrip():
    src = inject_lora(tiny_llama(), LoRAConfig(r=4, alpha=8))
    randomize_lora(src)
    dst = inject_lora(tiny_llama(), LoRAConfig(r=4, alpha=8))
    load_lora_state_dict(dst, lora_state_dict(src))
    ids = input_ids()
    with torch.no_grad():
        assert torch.equal(src(ids).logits, dst(ids).logits)
    with pytest.raises(KeyError):
        load_lora_state_dict(dst, {})


def test_inject_on_meta_device_counts_without_memory():
    with torch.device("meta"):
        model = tiny_llama()
    inject_lora(model, LoRAConfig(r=4, alpha=8))
    assert model.model.layers[0].self_attn.q_proj.lora_A.weight.is_meta
    assert count_trainable(model) > 0
