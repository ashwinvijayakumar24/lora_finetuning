"""Save and load adapters in PEFT's on-disk format.

A PEFT LoRA adapter directory holds two files:

- `adapter_config.json`: r, lora_alpha, target_modules, and feature flags.
- `adapter_model.safetensors`: only the A/B matrices, keyed
  `base_model.model.<module path>.lora_A.weight` (and `lora_B`). The prefix is
  PEFT's wrapper nesting (`PeftModel.base_model` is a `LoraModel` whose `.model`
  is the HF model); PEFT strips the adapter name (`.default`) when saving.

Writing exactly this format means vLLM and `PeftModel.from_pretrained` load our
adapters with no conversion, and we can load adapters PEFT trained. Our own
state-dict keys are the module path plus `.lora_A.weight`, so translation is
just adding or removing the prefix.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from playparse.lora.config import LoRAConfig
from playparse.lora.inject import inject_lora, load_lora_state_dict, lora_modules, lora_state_dict

PEFT_PREFIX = "base_model.model."
CONFIG_NAME = "adapter_config.json"
WEIGHTS_NAME = "adapter_model.safetensors"


def to_peft_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {PEFT_PREFIX + k: v for k, v in state.items()}


def from_peft_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in state.items():
        if not k.startswith(PEFT_PREFIX):
            raise KeyError(f"not a PEFT adapter key: {k!r}")
        out[k[len(PEFT_PREFIX):]] = v
    return out


def save_adapter(
    model: nn.Module,
    cfg: LoRAConfig,
    out_dir: str | Path,
    base_model_name_or_path: str | None = None,
    save_dtype: torch.dtype | None = None,
) -> Path:
    """Write `adapter_config.json` + `adapter_model.safetensors` to `out_dir`."""
    if any(m.merged for _, m in lora_modules(model)):
        raise RuntimeError("cannot save a merged adapter: unmerge first")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = {
        k: (v.to(save_dtype) if save_dtype else v).contiguous().cpu()
        for k, v in to_peft_keys(lora_state_dict(model)).items()
    }
    if not state:
        raise ValueError("model has no LoRA layers to save")
    save_file(state, str(out / WEIGHTS_NAME), metadata={"format": "pt"})
    (out / CONFIG_NAME).write_text(
        json.dumps(cfg.to_peft_dict(base_model_name_or_path), indent=2, sort_keys=True) + "\n"
    )
    return out


def read_adapter(adapter_dir: str | Path) -> tuple[LoRAConfig, dict[str, torch.Tensor]]:
    """Parse a PEFT adapter dir into our config and our (prefix-free) keys."""
    d = Path(adapter_dir)
    cfg = LoRAConfig.from_peft_dict(json.loads((d / CONFIG_NAME).read_text()))
    state = from_peft_keys(load_file(str(d / WEIGHTS_NAME)))
    for key, t in state.items():
        if not (key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight")):
            raise NotImplementedError(f"unsupported adapter tensor {key!r}")
        rank_dim = t.shape[0] if ".lora_A." in key else t.shape[1]
        if rank_dim != cfg.r:
            raise ValueError(f"{key}: rank {rank_dim} != config r={cfg.r}")
    return cfg, state


def load_adapter(
    model: nn.Module,
    adapter_dir: str | Path,
    adapter_dtype: torch.dtype | None = torch.float32,
) -> tuple[nn.Module, LoRAConfig]:
    """Inject adapters into a plain base model and fill them from `adapter_dir`.

    The modules to wrap come from the tensor keys, not from `target_modules`,
    because the file is the ground truth: PEFT may record `target_modules` as
    short names, full paths, or `"all-linear"`. We check that every module the
    file names is also selected by the config, to catch a mismatched pair.
    """
    cfg, state = read_adapter(adapter_dir)
    paths = sorted({k.rsplit(".", 2)[0] for k in state})
    for p in paths:
        if not cfg.matches(p):
            raise ValueError(f"adapter tensor for {p!r} is not selected by target_modules")
    exact = LoRAConfig(r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout, target_modules=paths)
    inject_lora(model, exact, adapter_dtype=adapter_dtype)
    load_lora_state_dict(model, state)
    return model, cfg
