"""A hand-written LoRA wrapper around a frozen `nn.Linear`.

    y = base(x) + (alpha / r) * B(A(dropout(x)))

`A` (r x in) starts Kaiming-uniform and `B` (out x r) starts at zero, so the
product `B @ A` is exactly zero at init: the wrapped model computes the same
function as the base model until the first optimizer step moves `B`. The
`alpha / r` factor keeps the size of the update roughly independent of the rank,
so a learning rate tuned at one rank stays reasonable at another.

The layout deliberately mirrors PEFT's `lora.Linear` (submodules named
`base_layer`, `lora_A`, `lora_B`; the scale applied after `B`; the input cast to
the adapter dtype). That keeps state-dict keys translatable to PEFT's on-disk
format by a fixed prefix, and keeps the floating-point operation order identical
so the oracle tests can demand agreement to 1e-5. PEFT is only the reference; no
PEFT code runs here.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Frozen `nn.Linear` plus a trainable rank-`r` update.

    The adapter parameters may live in a different dtype from the base layer
    (typically fp32 adapters on a bf16 base): the input is cast to the adapter
    dtype for the low-rank path, and the sum is cast back to the base output
    dtype, so callers see the dtype they would have seen without LoRA.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: float,
        dropout: float = 0.0,
        adapter_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear wraps nn.Linear, got {type(base).__name__}")
        if r <= 0:
            raise ValueError(f"rank r must be positive, got {r}")
        self.base_layer = base
        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        device = base.weight.device
        dtype = adapter_dtype if adapter_dtype is not None else base.weight.dtype
        self.lora_A = nn.Linear(self.in_features, r, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(r, self.out_features, bias=False, device=device, dtype=dtype)
        self.merged = False
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        # Same as nn.Linear's default init for A (bound = 1/sqrt(in_features)),
        # and B = 0 so the initial update B @ A is exactly zero.
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def delta_weight(self) -> torch.Tensor:
        """`scaling * B @ A`, shaped like the base weight, in fp32."""
        return (self.lora_B.weight.float() @ self.lora_A.weight.float()) * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        """Fold the update into the base weight so forward costs one matmul.

        The delta is computed in fp32 and added in the base dtype. On a bf16
        base this rounds, so `unmerge()` afterwards does not restore the
        original weight bit-for-bit (see docs/issues/p0-bf16-unmerge-drift.md).
        """
        if self.merged:
            raise RuntimeError("LoRALinear is already merged")
        w = self.base_layer.weight
        w.copy_((w.float() + self.delta_weight()).to(w.dtype))
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            raise RuntimeError("LoRALinear is not merged")
        w = self.base_layer.weight
        w.copy_((w.float() - self.delta_weight()).to(w.dtype))
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        if self.merged:
            return result
        out_dtype = result.dtype
        x = x.to(self.lora_A.weight.dtype)
        result = result + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return result.to(out_dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:g}, merged={self.merged}"
        )
