import pytest
import torch

from playparse.train.build import LoRASpec, apply_lora
from tests._lora_util import tiny_llama


def test_full_mode_trains_every_weight_and_roundtrips(tmp_path):
    model, save_fn, load_fn, impl = apply_lora(tiny_llama(seed=0), LoRASpec(impl="full"))
    assert impl == "full"
    assert all(p.requires_grad for p in model.parameters())
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01)
    save_fn(model, tmp_path / "ckpt")
    fresh, *_ = apply_lora(tiny_llama(seed=1), LoRASpec(impl="full"))
    load_fn(fresh, tmp_path / "ckpt")
    x = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        assert torch.equal(model(x).logits, fresh(x).logits)


def test_full_mode_rejects_bf16_weights():
    with pytest.raises(ValueError, match="fp32"):
        apply_lora(tiny_llama(seed=0).to(torch.bfloat16), LoRASpec(impl="full"))
