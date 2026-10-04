"""Build the pieces a training run needs: base model, LoRA adapter, encoded data.

Kept separate from the loop so the loop stays generic (any nn.Module), and so
scripts/train.py and scripts/p2_smoke_train.py share one code path.

LoRA implementation choice (RunSpec.lora.impl):
  "peft"       Hugging Face PEFT. The reference, and the stand-in until P0 lands.
  "playparse"  the hand-written playparse.lora (inject_lora(model, LoRAConfig)).
  "auto"       playparse if it is importable, else peft.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from playparse.paths import WEIGHTS
from playparse.train.loop import TrainConfig, load_trainable_safetensors, resolve_device, save_trainable_safetensors

ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class ModelSpec:
    weights: str | None = None  # None -> playparse.paths.WEIGHTS ($PLAYPARSE_WEIGHTS)
    dtype: str = "auto"  # auto: bf16 on cuda/mps, fp32 on cpu | bf16 | fp32
    gradient_checkpointing: bool = False
    attn_implementation: str | None = "sdpa"


@dataclass
class LoRASpec:
    impl: str = "auto"  # auto | peft | playparse
    r: int = 16
    alpha: float = 32
    dropout: float = 0.05
    targets: list[str] = field(default_factory=lambda: list(ALL_LINEAR))


@dataclass
class DataSpec:
    train: str = ""
    val: str | None = None
    max_len: int = 512
    mask_prompt: bool = True
    on_overlength: str = "raise"  # raise | drop
    limit_train: int | None = None
    limit_val: int | None = None


@dataclass
class RunSpec:
    """One file describes a whole run: {"model": ..., "lora": ..., "data": ..., "train": ...}."""

    model: ModelSpec = field(default_factory=ModelSpec)
    lora: LoRASpec = field(default_factory=LoRASpec)
    data: DataSpec = field(default_factory=DataSpec)
    train: TrainConfig = field(default_factory=TrainConfig)
    gen_eval_examples: int = 0  # >0: exact-match eval on this many val records at gen_every
    gen_max_new_tokens: int = 160

    @classmethod
    def from_dict(cls, d: dict) -> "RunSpec":
        d = dict(d)
        return cls(
            model=ModelSpec(**d.pop("model", {})),
            lora=LoRASpec(**d.pop("lora", {})),
            data=DataSpec(**d.pop("data", {})),
            train=TrainConfig.from_dict(d.pop("train", {})),
            **d,
        )

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> "RunSpec":
        path = Path(path)
        if path.suffix in (".yaml", ".yml"):
            import yaml

            d = yaml.safe_load(path.read_text())
        else:
            d = json.loads(path.read_text())
        return cls.from_dict(d or {})

    def to_dict(self) -> dict:
        d = asdict(self)
        d["train"] = self.train.to_dict()
        return d


def model_dtype(spec: ModelSpec, device: torch.device) -> torch.dtype:
    if spec.dtype == "bf16":
        return torch.bfloat16
    if spec.dtype == "fp32":
        return torch.float32
    return torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32


def load_base_model(spec: ModelSpec, device: torch.device) -> nn.Module:
    """Load the frozen base model in the chosen dtype, directly onto device."""
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"dtype": model_dtype(spec, device)}
    if spec.attn_implementation:
        kwargs["attn_implementation"] = spec.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(spec.weights or str(WEIGHTS), **kwargs)
    for p in model.parameters():
        p.requires_grad_(False)
    model.config.use_cache = False  # no KV cache during training forward passes
    if spec.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model.to(device)


def _playparse_lora_available() -> bool:
    try:
        from playparse.lora import LoRAConfig, inject_lora  # noqa: F401
    except ImportError:
        return False
    return True


def apply_lora(
    model: nn.Module, spec: LoRASpec
) -> tuple[nn.Module, Callable[[nn.Module, Path], None], Callable[[nn.Module, Path], None], str]:
    """Wrap model with LoRA. Returns (model, save_fn, load_fn, impl_used)."""
    impl = spec.impl
    if impl == "auto":
        impl = "playparse" if _playparse_lora_available() else "peft"
    if impl == "peft":
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from safetensors.torch import load_file

        cfg = LoraConfig(r=spec.r, lora_alpha=spec.alpha, lora_dropout=spec.dropout,
                         target_modules=list(spec.targets), task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)  # adapter weights are kept in fp32

        def save_fn(m: nn.Module, path: Path) -> None:
            m.save_pretrained(str(path))

        def load_fn(m: nn.Module, path: Path) -> None:
            res = set_peft_model_state_dict(m, load_file(str(path / "adapter_model.safetensors")))
            if getattr(res, "unexpected_keys", None):
                raise KeyError(f"unexpected adapter keys: {res.unexpected_keys[:5]}")

        return model, save_fn, load_fn, impl
    if impl == "playparse":
        from playparse.lora import LoRAConfig, inject_lora

        cfg = LoRAConfig(r=spec.r, alpha=spec.alpha, dropout=spec.dropout, target_modules=list(spec.targets))
        model = inject_lora(model, cfg)
        # Generic save/load of requires_grad params until playparse.lora's
        # PEFT-format writer is wired in as save_fn.
        return model, save_trainable_safetensors, load_trainable_safetensors, impl
    raise ValueError(f"unknown LoRA impl {spec.impl!r}")


def read_jsonl(path: str | os.PathLike, limit: int | None = None) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
                if limit is not None and len(out) >= limit:
                    break
    return out


def exact_match_callback(tokenizer: Any, records: list[dict], max_new_tokens: int = 160, batch_size: int = 16,
                         autocast: str = "auto") -> Callable[[nn.Module, int], dict]:
    """A val_callback that greedy-decodes records and scores exact match and parse rate.

    A placeholder until the P1 eval harness owns this; it uses the same schema
    contract (PlayLabel.from_json + matches).
    """
    from playparse.ffscore.schema import PlayLabel, SchemaError
    from playparse.train.generate import generate_for_records

    golds = [PlayLabel.from_json(r["label"]) for r in records]

    def cb(model: nn.Module, step: int) -> dict:
        texts = generate_for_records(model, tokenizer, records, max_new_tokens=max_new_tokens,
                                     batch_size=batch_size, autocast=autocast)
        parsed = em = 0
        for t, g in zip(texts, golds):
            try:
                p = PlayLabel.from_json(t)
            except SchemaError:
                continue
            parsed += 1
            em += int(p.matches(g))
        n = max(1, len(records))
        return {"val_exact_match": em / n, "val_parse_rate": parsed / n, "val_gen_n": len(records),
                "val_gen_sample": texts[0] if texts else ""}

    return cb


def describe_device(device: torch.device) -> dict:
    import platform

    info = {"device": str(device), "torch": torch.__version__, "python": platform.python_version(),
            "platform": platform.platform()}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    if device.type == "mps":
        info["chip"] = platform.processor() or "apple"
    return info


__all__ = [
    "ModelSpec", "LoRASpec", "DataSpec", "RunSpec", "load_base_model", "apply_lora", "read_jsonl",
    "exact_match_callback", "resolve_device", "describe_device", "model_dtype",
]
