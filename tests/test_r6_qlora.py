"""R6 (QLoRA): config plumbing on CPU, and a real 4-bit base on CUDA."""
import pytest
import torch
from torch import nn

from playparse.lora.lora_linear import LoRALinear, compute_dtype, is_quantized_linear
from playparse.train.build import (
    LoRASpec, ModelSpec, apply_lora, bnb_4bit_kwargs, normalize_quant, quantized_load_kwargs,
)
from tests._lora_util import tiny_llama


def _fake_4bit(in_f=8, out_f=4):
    """An nn.Linear dressed like a bitsandbytes Linear4bit (weight has quant_state)."""
    layer = nn.Linear(in_f, out_f, bias=False)
    layer.weight.quant_state = object()
    layer.compute_dtype = torch.bfloat16
    return layer


def test_normalize_quant():
    assert normalize_quant(None) is None and normalize_quant("none") is None
    assert normalize_quant("nf4") == "nf4"
    with pytest.raises(ValueError):
        normalize_quant("int3")
    assert ModelSpec(quant="null").quant is None


def test_bnb_kwargs_are_qlora_defaults():
    kw = bnb_4bit_kwargs("nf4")
    assert kw["load_in_4bit"] and kw["bnb_4bit_quant_type"] == "nf4"
    assert kw["bnb_4bit_use_double_quant"] and kw["bnb_4bit_compute_dtype"] == torch.bfloat16


def test_quantized_load_needs_cuda():
    assert quantized_load_kwargs(None, "cpu") == {}
    with pytest.raises(ValueError, match="CUDA"):
        quantized_load_kwargs("nf4", "cpu")


def test_adapter_dtype_follows_compute_dtype_of_4bit_base():
    base = _fake_4bit()
    assert is_quantized_linear(base) and compute_dtype(base) == torch.bfloat16
    layer = LoRALinear(base, r=2, alpha=4)
    assert layer.quantized and layer.lora_A.weight.dtype == torch.bfloat16
    assert "base=4bit" in repr(layer)
    assert not is_quantized_linear(nn.Linear(2, 2))


def test_full_finetune_refuses_a_4bit_base():
    model = tiny_llama(seed=0)
    model.model.layers[0].self_attn.q_proj.weight.quant_state = object()
    with pytest.raises(ValueError, match="4-bit"):
        apply_lora(model, LoRASpec(impl="full"))


@pytest.mark.gpu
def test_real_linear4bit_forward_merge_and_grad():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    bnb = pytest.importorskip("bitsandbytes")
    torch.manual_seed(0)
    dense = nn.Linear(64, 32, bias=False).to(torch.bfloat16)
    q = bnb.nn.Linear4bit(64, 32, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4")
    q.load_state_dict(dense.state_dict())
    q = q.cuda()  # quantizes on placement
    layer = LoRALinear(q, r=4, alpha=8, adapter_dtype=torch.float32).cuda()
    with torch.no_grad():
        layer.lora_B.weight.normal_(0, 0.05)
    x = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)
    y = layer(x)
    y.float().sum().backward()
    assert layer.lora_A.weight.grad is not None and layer.lora_A.weight.grad.abs().sum() > 0
    with torch.no_grad():
        before = layer(x).float()
        layer.merge()
        after = layer(x).float()
    assert not layer.quantized and layer.merged
    assert (after - before).abs().max() < 0.1  # bf16 merge of a dequantized NF4 base
