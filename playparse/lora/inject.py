"""Put LoRA adapters into a model, read them out, and fold them back in.

`inject_lora` freezes every existing parameter and then swaps each targeted
`nn.Linear` for a `LoRALinear` that wraps it. After injection the only
parameters with `requires_grad=True` are the `lora_A` / `lora_B` weights, so an
optimizer built from `trainable_parameters(model)` holds states for the adapter
alone. That is where LoRA's memory saving comes from: AdamW keeps two fp32
moments per trainable parameter, and the frozen base has none.
"""
from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn

from playparse.lora.config import LoRAConfig
from playparse.lora.lora_linear import LoRALinear


def _set_submodule(model: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, new)


def lora_modules(model: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def inject_lora(
    model: nn.Module,
    cfg: LoRAConfig,
    adapter_dtype: torch.dtype | None = torch.float32,
) -> nn.Module:
    """Wrap every `nn.Linear` whose path matches `cfg` and freeze the rest.

    Modifies `model` in place and returns it. `adapter_dtype=None` keeps each
    adapter in its base layer's dtype; the default fp32 is what training wants
    on a bf16 base, because tiny updates to bf16 weights round away.
    """
    if any(True for _ in lora_modules(model)):
        raise RuntimeError("model already has LoRA layers; refusing to inject twice")
    for p in model.parameters():
        p.requires_grad_(False)

    # Collect first: replacing modules while iterating named_modules() is unsafe.
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and cfg.matches(name)
    ]
    if not targets:
        raise ValueError(f"no nn.Linear matched target_modules={cfg.target_modules!r}")
    for name in targets:
        base = model.get_submodule(name)
        wrapped = LoRALinear(base, cfg.r, cfg.alpha, cfg.dropout, adapter_dtype=adapter_dtype)
        # A freshly built module is in train mode; inherit the base layer's mode so
        # injecting into an eval() model does not silently switch dropout on.
        wrapped.train(base.training)
        _set_submodule(model, name, wrapped)
    return model


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model: nn.Module) -> int:
    """All parameters, counting tied tensors (embed/lm_head) once."""
    return sum(p.numel() for p in model.parameters())


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Adapter weights only, keyed like `model.layers.0.self_attn.q_proj.lora_A.weight`."""
    out: dict[str, torch.Tensor] = {}
    for name, module in lora_modules(model):
        out[f"{name}.lora_A.weight"] = module.lora_A.weight.detach()
        out[f"{name}.lora_B.weight"] = module.lora_B.weight.detach()
    return out


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Copy adapter weights into an injected model; keys must match exactly."""
    own = dict(lora_modules(model))
    expected = {f"{n}.lora_{ab}.weight" for n in own for ab in ("A", "B")}
    missing, unexpected = expected - state.keys(), state.keys() - expected
    if missing or unexpected:
        raise KeyError(
            f"adapter keys mismatch: {len(missing)} missing (e.g. {sorted(missing)[:2]}), "
            f"{len(unexpected)} unexpected (e.g. {sorted(unexpected)[:2]})"
        )
    with torch.no_grad():
        for key, tensor in state.items():
            module_name, ab = key.rsplit(".", 2)[0], key.rsplit(".", 2)[1]
            target = getattr(own[module_name], ab).weight
            if target.shape != tensor.shape:
                raise ValueError(f"{key}: shape {tuple(tensor.shape)} != {tuple(target.shape)}")
            target.copy_(tensor.to(target.dtype))


def merge_lora(model: nn.Module) -> nn.Module:
    """Fold every adapter into its base weight and remove the wrappers.

    Returns the same model object with each `LoRALinear` replaced by its (now
    merged) `nn.Linear`: a plain architecture any loader or engine understands,
    with the base model's exact forward cost.
    """
    for name, module in list(lora_modules(model)):
        if not module.merged:
            module.merge()
        _set_submodule(model, name, module.base_layer)
    return model
