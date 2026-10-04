"""playparse.lora is the default training adapter, saved in PEFT format.

A run's saved adapter must load in every consumer downstream of training: the
hand-written loader (playparse.lora.load_adapter), the serving layer
(playparse.serving.adapter.load_peft_adapter, used by P5a/P5b), and PEFT itself
(the stand-in for vLLM, which reads the same files). Tiny CPU models only.
"""
from __future__ import annotations

import json

import pytest
import torch

from p2_fixtures import synthetic_examples, tiny_llama
from playparse.lora import LoRALinear, load_adapter
from playparse.train.build import LoRASpec, RunSpec, apply_lora
from playparse.train.loop import TrainConfig, list_checkpoints, train


def _cfg(out, **kw) -> TrainConfig:
    base = dict(output_dir=str(out), device="cpu", lr=5e-3, micro_batch_size=4, grad_accum_steps=1, max_steps=4,
                log_every=1, keep_last_checkpoints=None, save_final=True)
    base.update(kw)
    return TrainConfig(**base)


def _spec(**kw) -> LoRASpec:
    return LoRASpec(**{"r": 4, "alpha": 8, "dropout": 0.0, **kw})


def _logits(model, ids):
    model.eval()
    with torch.no_grad():
        return model(input_ids=ids).logits.float()


@pytest.fixture
def trained(tmp_path):
    model, save_fn, load_fn, impl = apply_lora(tiny_llama(seed=0), _spec())
    assert impl == "playparse"
    res = train(model, synthetic_examples(16, seed=1), None, _cfg(tmp_path), pad_id=0, save_fn=save_fn,
                load_fn=load_fn)
    adapter_dir = list_checkpoints(tmp_path)[-1] / "adapter"
    assert res.step == 4
    return model, adapter_dir


def test_default_impl_is_playparse():
    assert LoRASpec().impl == "playparse"
    assert RunSpec().lora.impl == "playparse"
    model, *_ , impl = apply_lora(tiny_llama(seed=0), LoRASpec(r=4, alpha=8))
    assert impl == "playparse"
    assert any(isinstance(m, LoRALinear) for m in model.modules())


def test_checkpoint_is_peft_format(trained):
    _, d = trained
    assert sorted(p.name for p in d.iterdir()) == ["adapter_config.json", "adapter_model.safetensors"]
    cfg = json.loads((d / "adapter_config.json").read_text())
    assert cfg["peft_type"] == "LORA" and cfg["r"] == 4 and cfg["lora_alpha"] == 8
    assert cfg["target_modules"] == sorted(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                            "down_proj"])
    from safetensors.torch import load_file

    keys = list(load_file(str(d / "adapter_model.safetensors")))
    assert all(k.startswith("base_model.model.model.layers.") for k in keys)
    assert len(keys) == 2 * 7 * 2  # A and B, 7 projections, 2 layers


def test_saved_adapter_loads_in_playparse_and_matches(trained):
    model, d = trained
    ids = torch.randint(2, 64, (2, 10), generator=torch.Generator().manual_seed(0))
    fresh, _ = load_adapter(tiny_llama(seed=0), d)
    torch.testing.assert_close(_logits(fresh, ids), _logits(model, ids), rtol=0, atol=1e-6)


def test_saved_adapter_loads_in_peft_and_matches(trained):
    from peft import PeftModel

    model, d = trained
    ids = torch.randint(2, 64, (2, 10), generator=torch.Generator().manual_seed(0))
    peft_model = PeftModel.from_pretrained(tiny_llama(seed=0), str(d))
    # The adapter really changed the function (B moved away from zero) ...
    assert not torch.allclose(_logits(tiny_llama(seed=0), ids), _logits(model, ids), atol=1e-4)
    # ... and PEFT reproduces it.
    torch.testing.assert_close(_logits(peft_model, ids), _logits(model, ids), rtol=1e-5, atol=1e-5)


def test_saved_adapter_loads_in_serving_layer(trained):
    from playparse.serving.adapter import load_peft_adapter

    model, d = trained
    adapter = load_peft_adapter(d)
    assert adapter.r == 4 and adapter.lora_alpha == 8
    assert len(adapter.layers) == 14
    # Same matrices as the live model, under engine weight names.
    q = model.model.layers[1].self_attn.q_proj
    lt = adapter.layers["model.layers.1.self_attn.q_proj.weight"]
    assert lt.scale == pytest.approx(2.0)
    assert torch.equal(torch.from_numpy(lt.A), q.lora_A.weight.detach().float())
    assert torch.equal(torch.from_numpy(lt.B), q.lora_B.weight.detach().float())


def test_load_fn_rejects_mismatched_rank(trained, tmp_path):
    _, d = trained
    model, _, load_fn, _ = apply_lora(tiny_llama(seed=0), _spec(r=8, alpha=16))
    with pytest.raises(ValueError, match="r=4"):
        load_fn(model, d)


def test_resume_with_playparse_impl_is_exact(tmp_path):
    exs, val = synthetic_examples(24, seed=7), synthetic_examples(6, seed=8)
    kw = dict(max_steps=6, micro_batch_size=3, grad_accum_steps=2, warmup_steps=1, eval_every=2, save_every=2)

    def fresh(scramble=False):
        m, s, l, _ = apply_lora(tiny_llama(seed=0), _spec(dropout=0.1))
        if scramble:
            with torch.no_grad():
                for p in m.parameters():
                    if p.requires_grad:
                        p.normal_()
        return m, s, l

    full, s, l = fresh()
    r_full = train(full, exs, val, _cfg(tmp_path / "full", **kw), pad_id=0, save_fn=s, load_fn=l)
    part, s, l = fresh()
    train(part, exs, val, _cfg(tmp_path / "split", **kw), pad_id=0, save_fn=s, load_fn=l, stop_after_step=2)
    resumed, s, l = fresh(scramble=True)
    torch.manual_seed(999)
    r_res = train(resumed, exs, val, _cfg(tmp_path / "split", **kw), pad_id=0, save_fn=s, load_fn=l,
                  resume_from="latest")
    assert r_res.step == 6 and r_res.val_loss == r_full.val_loss
    a = {n: p for n, p in full.named_parameters() if p.requires_grad}
    b = {n: p for n, p in resumed.named_parameters() if p.requires_grad}
    for k in a:
        assert torch.equal(a[k], b[k]), k
