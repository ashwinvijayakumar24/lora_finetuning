"""PEFT on-disk format: our adapters load in PEFT, and PEFT's load in ours."""
import json

import pytest
import torch
from safetensors.torch import load_file

from playparse.lora import LoRAConfig, inject_lora, load_adapter, lora_state_dict, read_adapter, save_adapter
from tests._lora_util import input_ids, peft_config, randomize_lora, tiny_llama, tiny_pair

peft = pytest.importorskip("peft")


def _trained(cfg: LoRAConfig, seed: int = 0):
    model = inject_lora(tiny_llama(seed), cfg)
    randomize_lora(model)
    return model


@pytest.mark.parametrize("targets", ["all-linear", ["q_proj", "v_proj"]])
def test_saved_by_us_loads_in_peft_with_same_logits(tmp_path, targets):
    cfg = LoRAConfig(r=4, alpha=8, dropout=0.0, target_modules=targets)
    ours = _trained(cfg)
    save_adapter(ours, cfg, tmp_path)
    pm = peft.PeftModel.from_pretrained(tiny_llama(), tmp_path).eval()
    ids = input_ids()
    with torch.no_grad():
        diff = (ours(ids).logits - pm(input_ids=ids).logits).abs().max().item()
    assert diff < 1e-5


def test_saved_by_peft_loads_in_ours_with_same_logits(tmp_path):
    cfg = LoRAConfig(r=4, alpha=8, dropout=0.0)
    pm = peft.get_peft_model(tiny_llama(), peft_config(cfg))
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    pm.save_pretrained(tmp_path)
    # PEFT expands "all-linear" into full module paths in adapter_config.json
    saved_targets = json.loads((tmp_path / "adapter_config.json").read_text())["target_modules"]
    assert any("layers.0" in t for t in saved_targets)

    ours, loaded_cfg = load_adapter(tiny_llama(), tmp_path)
    assert (loaded_cfg.r, loaded_cfg.alpha) == (4, 8.0)
    pm.eval()
    ids = input_ids()
    with torch.no_grad():
        diff = (ours(ids).logits - pm(input_ids=ids).logits).abs().max().item()
    assert diff < 1e-5


def test_file_layout_and_key_names(tmp_path):
    cfg = LoRAConfig(r=4, alpha=8, target_modules=["q_proj", "v_proj"])
    save_adapter(_trained(cfg), cfg, tmp_path, base_model_name_or_path="meta-llama/Llama-3.2-1B-Instruct")
    assert {p.name for p in tmp_path.iterdir()} == {"adapter_config.json", "adapter_model.safetensors"}
    keys = set(load_file(str(tmp_path / "adapter_model.safetensors")))
    assert "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" in keys
    assert "base_model.model.model.layers.1.self_attn.v_proj.lora_B.weight" in keys
    assert len(keys) == 2 * 2 * 2
    conf = json.loads((tmp_path / "adapter_config.json").read_text())
    assert conf["peft_type"] == "LORA" and conf["r"] == 4 and conf["lora_alpha"] == 8
    assert conf["target_modules"] == ["q_proj", "v_proj"]
    assert conf["base_model_name_or_path"] == "meta-llama/Llama-3.2-1B-Instruct"


def test_our_roundtrip_is_exact_and_save_dtype(tmp_path):
    cfg = LoRAConfig(r=4, alpha=8, dropout=0.0)
    src = _trained(cfg)
    save_adapter(src, cfg, tmp_path / "fp32")
    dst, _ = load_adapter(tiny_llama(), tmp_path / "fp32")
    for k, v in lora_state_dict(src).items():
        assert torch.equal(v, lora_state_dict(dst)[k])
    save_adapter(src, cfg, tmp_path / "bf16", save_dtype=torch.bfloat16)
    _, state = read_adapter(tmp_path / "bf16")
    assert all(t.dtype == torch.bfloat16 for t in state.values())


def test_rejects_unsupported_peft_features(tmp_path):
    for flag in ("use_rslora", "use_dora"):
        cfg = peft_config(LoRAConfig(r=4, alpha=8, dropout=0.0))
        setattr(cfg, flag, True)
        pm = peft.get_peft_model(tiny_llama(), cfg)
        out = tmp_path / flag
        pm.save_pretrained(out)
        with pytest.raises(NotImplementedError):
            load_adapter(tiny_llama(), out)


def test_cannot_save_merged(tmp_path):
    cfg = LoRAConfig(r=4, alpha=8)
    model = _trained(cfg)
    model.model.layers[0].self_attn.q_proj.merge()
    with pytest.raises(RuntimeError):
        save_adapter(model, cfg, tmp_path)


def test_peft_and_ours_agree_on_base_for_pair(tmp_path):
    """Sanity: tiny_pair really yields identical bases (the oracle tests depend on it)."""
    a, b = tiny_pair()
    for (n1, p1), (n2, p2) in zip(a.state_dict().items(), b.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2)
