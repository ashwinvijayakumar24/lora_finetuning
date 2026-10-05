"""Build the pieces a training run needs: base model, LoRA adapter, encoded data.

Kept separate from the loop so the loop stays generic (any nn.Module), and so
scripts/train.py and scripts/p2_smoke_train.py share one code path.

LoRA implementation choice (RunSpec.lora.impl):
  "playparse"  the default: the hand-written playparse.lora (inject_lora with a
               LoRAConfig). Checkpoints and the best/final adapter are written by
               playparse.lora.io.save_adapter in PEFT's on-disk format, so P5a/P5b
               serving, vLLM, and PeftModel.from_pretrained load them unchanged.
  "peft"       Hugging Face PEFT, kept as the reference implementation.
  "auto"       playparse if it is importable, else peft.
"""
from __future__ import annotations

import shutil

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from playparse.paths import WEIGHTS
from playparse.train.loop import TrainConfig, resolve_device

ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class ModelSpec:
    weights: str | None = None  # None -> playparse.paths.WEIGHTS ($PLAYPARSE_WEIGHTS)
    dtype: str = "auto"  # auto: bf16 on cuda/mps, fp32 on cpu | bf16 | fp32
    gradient_checkpointing: bool = False
    attn_implementation: str | None = "sdpa"


@dataclass
class LoRASpec:
    impl: str = "playparse"  # playparse | peft | auto | full (R7: no LoRA, every weight trains)
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
    limit_train: int | None = None  # first N train records (file order; for smoke runs)
    # >0: train on the first N of a seeded permutation of the train file. Sizes are
    # nested (the 1k sample is inside the 5k sample), which is what a data-size
    # curve needs. 0 = all records.
    train_sample: int = 0
    train_sample_seed: int = 0
    limit_val: int | None = None  # first N val records (file order)
    # >0: val loss on a seeded, bucket-proportional sample of this many val records
    # (seed = gen_eval_seed) instead of the head of the file, which is a few games.
    val_loss_examples: int = 0


@dataclass
class RunSpec:
    """One file describes a whole run: {"model": ..., "lora": ..., "data": ..., "train": ...}."""

    model: ModelSpec = field(default_factory=ModelSpec)
    lora: LoRASpec = field(default_factory=LoRASpec)
    data: DataSpec = field(default_factory=DataSpec)
    train: TrainConfig = field(default_factory=TrainConfig)
    # Generation eval at train.gen_every (playparse.train.val_eval.harness_val_callback):
    # greedy-decode a fixed, seeded, bucket-stratified val subset and log exact match
    # overall and per bucket. 0 disables it.
    gen_eval_examples: int = 0
    gen_eval_seed: int = 0
    gen_eval_strategy: str = "balanced"  # balanced | proportional (see stratified_subset)
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
        model = inject_lora(model, cfg)  # adapters in fp32 on a bf16 base
        save_fn, load_fn = playparse_adapter_io(cfg, getattr(getattr(model, "config", None), "_name_or_path", None))
        return model, save_fn, load_fn, impl
    if impl == "full":
        save_fn, load_fn = full_finetune_io(model)
        return model, save_fn, load_fn, impl
    raise ValueError(f"unknown LoRA impl {spec.impl!r}")


def full_finetune_io(model: nn.Module) -> tuple[Callable[[nn.Module, Path], None], Callable[[nn.Module, Path], None]]:
    """Make every weight trainable (rung R7) and return save_fn/load_fn for full checkpoints.

    AdamW updates on bf16 weights lose small steps to rounding, so full fine-tuning
    requires fp32 weights (bf16 autocast still runs the matmuls in bf16 on CUDA).
    That is ~16 bytes/param of training state (18.4 GiB for the 1B model, see P0),
    which is why this mode is for the H100, not the laptop.
    """
    bad = {p.dtype for p in model.parameters() if p.dtype != torch.float32}
    if bad:
        raise ValueError(f"full fine-tuning needs fp32 weights (set model.dtype: fp32); found {sorted(map(str, bad))}")
    for p in model.parameters():
        p.requires_grad_(True)

    def save_fn(m: nn.Module, path: Path) -> None:
        # A loadable HF checkpoint dir; tokenizer files are copied from the base so
        # the eval CLI can load it with --rung r1 --weights <dir>.
        path = Path(path)
        m.save_pretrained(str(path), safe_serialization=True)
        base = Path(getattr(getattr(m, "config", None), "_name_or_path", "") or "")
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"):
            if (base / name).is_file() and not (path / name).exists():
                shutil.copy2(base / name, path / name)

    def load_fn(m: nn.Module, path: Path) -> None:
        from safetensors.torch import load_file

        state = load_file(str(Path(path) / "model.safetensors"))
        res = m.load_state_dict(state, strict=False)
        # Tied embeddings: lm_head.weight is not saved separately.
        missing = [k for k in res.missing_keys if k != "lm_head.weight"]
        if missing or res.unexpected_keys:
            raise KeyError(f"checkpoint mismatch: missing={missing[:5]} unexpected={res.unexpected_keys[:5]}")

    return save_fn, load_fn


def playparse_adapter_io(
    cfg: Any, base_model_name_or_path: str | None = None
) -> tuple[Callable[[nn.Module, Path], None], Callable[[nn.Module, Path], None]]:
    """save_fn/load_fn for a model already injected with playparse.lora.

    save_fn writes adapter_config.json + adapter_model.safetensors (PEFT format,
    fp32 tensors). load_fn fills the *existing* LoRA layers from such a directory
    (used on resume, where the model is already injected), after checking that
    the saved rank, alpha, and wrapped modules match this run, so a checkpoint from
    a different config fails loudly instead of loading at the wrong scale.
    """
    from playparse.lora import load_lora_state_dict, read_adapter, save_adapter

    name = str(base_model_name_or_path) if base_model_name_or_path else None

    def save_fn(m: nn.Module, path: Path) -> None:
        save_adapter(m, cfg, path, base_model_name_or_path=name)

    def load_fn(m: nn.Module, path: Path) -> None:
        saved, state = read_adapter(path)
        if (saved.r, float(saved.alpha)) != (cfg.r, float(cfg.alpha)):
            raise ValueError(f"{path}: adapter r={saved.r} alpha={saved.alpha} != run r={cfg.r} alpha={cfg.alpha}")
        load_lora_state_dict(m, state)  # raises on any missing or unexpected module

    return save_fn, load_fn


def nested_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    """The first n records of a seeded permutation (nested across n for one seed)."""
    import random

    if n <= 0 or n >= len(records):
        return list(records)
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    return [records[i] for i in order[:n]]


def read_jsonl(path: str | os.PathLike, limit: int | None = None) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
                if limit is not None and len(out) >= limit:
                    break
    return out


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
    "ModelSpec", "LoRASpec", "DataSpec", "RunSpec", "load_base_model", "apply_lora", "playparse_adapter_io",
    "read_jsonl", "nested_sample", "resolve_device", "describe_device", "model_dtype",
]
