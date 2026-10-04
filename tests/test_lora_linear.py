import math

import pytest
import torch
from torch import nn

from playparse.lora import LoRALinear


def _layer(in_f=32, out_f=48, r=4, alpha=8.0, dropout=0.0, dtype=torch.float32, adapter_dtype=None, seed=0):
    torch.manual_seed(seed)
    base = nn.Linear(in_f, out_f, bias=True, dtype=dtype)
    return LoRALinear(base, r=r, alpha=alpha, dropout=dropout, adapter_dtype=adapter_dtype)


def test_init_b_zero_a_kaiming_bound():
    layer = _layer(in_f=32)
    assert torch.count_nonzero(layer.lora_B.weight) == 0
    # kaiming_uniform(a=sqrt(5)) on fan_in=32 gives U(-1/sqrt(32), 1/sqrt(32)).
    bound = 1 / math.sqrt(32)
    a = layer.lora_A.weight
    assert a.abs().max() <= bound
    assert a.abs().max() > 0.5 * bound  # not degenerate
    assert a.shape == (4, 32) and layer.lora_B.weight.shape == (48, 4)


def test_output_equals_base_at_init_exactly():
    layer = _layer()
    x = torch.randn(3, 5, 32)
    assert torch.equal(layer(x), layer.base_layer(x))


def test_formula_and_scaling():
    layer = _layer(r=4, alpha=8.0)
    nn.init.normal_(layer.lora_B.weight)
    x = torch.randn(7, 32)
    expected = layer.base_layer(x) + 2.0 * (x @ layer.lora_A.weight.T @ layer.lora_B.weight.T)
    assert layer.scaling == 2.0
    torch.testing.assert_close(layer(x), expected, rtol=1e-5, atol=1e-5)


def test_only_adapter_requires_grad():
    layer = _layer()
    names = {n for n, p in layer.named_parameters() if p.requires_grad}
    assert names == {"lora_A.weight", "lora_B.weight"}


def test_merge_unmerge_roundtrip_fp32():
    layer = _layer()
    nn.init.normal_(layer.lora_B.weight, std=0.1)
    w0 = layer.base_layer.weight.clone()
    x = torch.randn(4, 32)
    y_unmerged = layer(x)
    layer.merge()
    assert layer.merged
    torch.testing.assert_close(layer(x), y_unmerged, rtol=1e-5, atol=1e-5)
    # forward must not add the delta a second time once merged
    torch.testing.assert_close(layer(x), layer.base_layer(x), rtol=0, atol=0)
    layer.unmerge()
    torch.testing.assert_close(layer.base_layer.weight, w0, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(layer(x), y_unmerged, rtol=1e-5, atol=1e-5)


def test_double_merge_and_unmerge_without_merge_raise():
    layer = _layer()
    with pytest.raises(RuntimeError):
        layer.unmerge()
    layer.merge()
    with pytest.raises(RuntimeError):
        layer.merge()


def test_dropout_active_in_train_only():
    layer = _layer(dropout=0.5)
    nn.init.normal_(layer.lora_B.weight)
    x = torch.randn(16, 32)
    no_drop = layer.base_layer(x) + layer.scaling * layer.lora_B(layer.lora_A(x))
    layer.eval()
    torch.testing.assert_close(layer(x), no_drop)
    layer.train()
    torch.manual_seed(0)
    assert not torch.allclose(layer(x), no_drop)


def test_bf16_base_fp32_adapter_dtypes_and_grads():
    layer = _layer(dtype=torch.bfloat16, adapter_dtype=torch.float32)
    assert layer.base_layer.weight.dtype == torch.bfloat16
    assert layer.lora_A.weight.dtype == torch.float32
    nn.init.normal_(layer.lora_B.weight, std=0.1)
    x = torch.randn(4, 32, dtype=torch.bfloat16)
    y = layer(x)
    assert y.dtype == torch.bfloat16  # callers see the base dtype
    y.float().sum().backward()
    assert layer.lora_A.weight.grad.dtype == torch.float32
    assert layer.lora_B.weight.grad is not None
    assert layer.base_layer.weight.grad is None


def test_adapter_dtype_defaults_to_base_dtype():
    layer = _layer(dtype=torch.bfloat16)
    assert layer.lora_A.weight.dtype == torch.bfloat16


def test_bf16_unmerge_is_not_bit_exact():
    """Guard for docs/issues/p0-bf16-unmerge-drift.md: merge rounds into bf16.

    Merging into a bf16 weight and subtracting again does not return the
    original weight, so serving code must keep a pristine copy (or reload) rather
    than rely on unmerge() to switch adapters on a bf16 base.
    """
    layer = _layer(in_f=256, out_f=256, r=8, dtype=torch.bfloat16, adapter_dtype=torch.float32)
    nn.init.normal_(layer.lora_B.weight, std=0.02)
    w0 = layer.base_layer.weight.clone()
    layer.merge()
    layer.unmerge()
    drift = (layer.base_layer.weight.float() - w0.float()).abs()
    assert drift.max() > 0  # rounding happened
    assert drift.max() < 1e-2  # but it is only rounding-sized


def test_rejects_non_linear_and_bad_rank():
    with pytest.raises(TypeError):
        LoRALinear(nn.Conv1d(4, 4, 1), r=2, alpha=4)
    with pytest.raises(ValueError):
        LoRALinear(nn.Linear(4, 4), r=0, alpha=4)


def test_extra_repr():
    text = repr(_layer(r=4, alpha=8.0))
    assert "r=4" in text and "alpha=8.0" in text and "scaling=2" in text and "merged=False" in text
