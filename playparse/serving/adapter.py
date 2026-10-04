"""Load a PEFT LoRA adapter into a validated, engine-addressed structure.

A LoRA adapter replaces a frozen linear weight ``W`` (shape ``(out, in)``) with
``W + scale * B @ A`` where ``A`` is ``(r, in)``, ``B`` is ``(out, r)`` and
``scale = lora_alpha / r`` (or ``lora_alpha / sqrt(r)`` under rsLoRA). PEFT
stores ``A`` and ``B`` on disk as two files::

    adapter_config.json          r, lora_alpha, target_modules, option flags
    adapter_model.safetensors    base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight ...

This module turns that into a :class:`LoRAAdapter`: a dict from *engine weight
name* (``model.layers.0.self_attn.q_proj.weight``) to a :class:`LoRATensors`
holding ``A``, ``B`` and ``scale`` in fp32 NumPy. Keeping it framework-neutral
lets the same object feed the NumPy reference engine, the torch engine, and the
merge path.

Name mapping is short because ``llm_inference_engine`` keeps the Hugging Face
weight names verbatim (``engine/loader.py``) and stores every weight as
``(out, in)`` applied as ``x @ W.T`` — the same layout PEFT's ``A``/``B`` assume.
So a PEFT key maps to an engine key by stripping the ``base_model.model.`` prefix
and the ``.lora_A`` / ``.lora_B`` segment; no transpose is needed anywhere. The
one layout trap, PEFT's ``fan_in_fan_out`` (GPT-2 style ``(in, out)`` weights), is
rejected explicitly rather than silently transposed.

Anything the engine cannot reproduce exactly is rejected with a message naming
the option: DoRA, LoRA biases, ``modules_to_save``, per-module rank or alpha
patterns, initialisations that rewrite the base weights (PiSSA, OLoRA, LoftQ,
CorDA, LoRA-GA), and any option this file does not know about. Silently ignoring
one of those would produce a model that loads, generates fluent text, and is
wrong.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

# Engine weight names are "model.layers.{i}.{block}.{module}.weight".
# These are the seven projections the engine routes through linear(); they are
# also exactly the set the engine's int8/int4 path quantizes.
SUPPORTED_MODULES: dict[str, str] = {
    "q_proj": "self_attn",
    "k_proj": "self_attn",
    "v_proj": "self_attn",
    "o_proj": "self_attn",
    "gate_proj": "mlp",
    "up_proj": "mlp",
    "down_proj": "mlp",
}

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"

# PEFT key: [base_model.model.]model.layers.{i}.{block}.{module}.lora_{A|B}[.{adapter_name}].weight
_PEFT_KEY = re.compile(
    r"^(?:base_model\.model\.)?"
    r"model\.layers\.(?P<layer>\d+)\.(?P<block>[a-z_]+)\.(?P<module>[a-z_]+)"
    r"\.lora_(?P<which>[AB])(?:\.[A-Za-z0-9_\-]+)?\.weight$"
)


class AdapterError(ValueError):
    """The adapter cannot be served faithfully by the engine."""


# --------------------------------------------------------------------------
# Name mapping
# --------------------------------------------------------------------------

def engine_weight_name(layer: int, module: str) -> str:
    """Engine (and HF) weight name for one projection in one layer."""
    if module not in SUPPORTED_MODULES:
        raise AdapterError(f"unsupported target module {module!r}")
    return f"model.layers.{layer}.{SUPPORTED_MODULES[module]}.{module}.weight"


def parse_peft_key(key: str) -> tuple[int, str, str]:
    """Split a PEFT tensor key into ``(layer, module, "A" | "B")``.

    Raises AdapterError for anything that is not a LoRA A/B matrix on one of the
    seven supported projections — for example ``lm_head`` or ``embed_tokens``
    targets, ``lora_embedding_A``, DoRA's ``lora_magnitude_vector``, or a bias.
    """
    m = _PEFT_KEY.match(key)
    if m is None:
        raise AdapterError(
            f"unsupported adapter tensor {key!r}: only lora_A/lora_B on "
            f"{sorted(SUPPORTED_MODULES)} inside model.layers.* are servable"
        )
    module = m.group("module")
    block = m.group("block")
    if module not in SUPPORTED_MODULES or SUPPORTED_MODULES[module] != block:
        raise AdapterError(f"unsupported target module {block}.{module} in {key!r}")
    return int(m.group("layer")), module, m.group("which")


def peft_to_engine_key(key: str) -> tuple[str, str]:
    """``base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`` →
    ``("model.layers.0.self_attn.q_proj.weight", "A")``."""
    layer, module, which = parse_peft_key(key)
    return engine_weight_name(layer, module), which


def engine_to_peft_key(engine_name: str, which: str) -> str:
    """Inverse of :func:`peft_to_engine_key` (the on-disk spelling PEFT writes)."""
    if which not in ("A", "B"):
        raise ValueError(f"which must be 'A' or 'B', got {which!r}")
    if not engine_name.endswith(".weight"):
        raise ValueError(f"not an engine weight name: {engine_name!r}")
    return f"base_model.model.{engine_name[: -len('.weight')]}.lora_{which}.weight"


def expected_linear_shapes(config: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    """``(out, in)`` of each supported projection, from an HF/engine config."""
    H = config["hidden_size"]
    NH = config["num_attention_heads"]
    NKV = config["num_key_value_heads"]
    HD = config.get("head_dim") or H // NH
    FF = config["intermediate_size"]
    return {
        "q_proj": (NH * HD, H),
        "k_proj": (NKV * HD, H),
        "v_proj": (NKV * HD, H),
        "o_proj": (H, NH * HD),
        "gate_proj": (FF, H),
        "up_proj": (FF, H),
        "down_proj": (H, FF),
    }


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------

# Keys whose value does not change what the adapter computes at inference.
_BENIGN_KEYS = frozenset({
    "alpha_pattern", "rank_pattern",          # checked separately (must be empty)
    "auto_mapping", "base_model_name_or_path", "revision", "peft_version",
    "inference_mode", "task_type", "lora_dropout",   # dropout is identity at inference
    "target_modules", "exclude_modules", "layers_to_transform", "layers_pattern",
    "megatron_core", "qalora_group_size", "ensure_weight_tying",
    "r", "lora_alpha", "use_rslora", "peft_type", "bias", "init_lora_weights",
    "eva_config",   # EVA only chooses A's initial value; base weights are untouched
})

# Options that change the computation (or the base weights) in ways this engine
# does not implement. Any truthy value is rejected with the reason.
_UNSUPPORTED_OPTIONS: dict[str, str] = {
    "use_dora": "DoRA rescales W by a learned magnitude vector; not implemented",
    "lora_bias": "LoRA bias terms are not implemented",
    "fan_in_fan_out": "(in, out) weight layout is not the engine's (out, in) layout",
    "modules_to_save": "fully fine-tuned extra modules would need a second weight set",
    "layer_replication": "layer replication changes model depth",
    "trainable_token_indices": "trainable token embeddings are not implemented",
    "target_parameters": "parameter-targeted LoRA (e.g. MoE experts) is not implemented",
    "use_qalora": "QA-LoRA pooling is not implemented",
    "alora_invocation_tokens": "activated LoRA (token-conditional) is not implemented",
    "use_bdlora": "BD-LoRA is not implemented",
    "megatron_config": "Megatron-parallel LoRA layers are not implemented",
    "arrow_config": "Arrow routing across adapters is not implemented",
    "loftq_config": "LoftQ rewrites the base weights; the adapter is only valid with them",
    "corda_config": "CorDA rewrites the base weights; the adapter is only valid with them",
    "lora_ga_config": "LoRA-GA rewrites the base weights; the adapter is only valid with them",
    "kasa_config": "KaSA is not implemented",
    "monteclora_config": "MonteCLoRA is not implemented",
    "velora_config": "VeLoRA is not implemented",
}

# init_lora_weights values that leave the base model untouched.
_SAFE_INITS = {True, False, "gaussian", "true", "false"}


def validate_peft_config(cfg: Mapping[str, Any]) -> None:
    """Reject PEFT options the engine cannot reproduce. Raises AdapterError."""
    problems: list[str] = []
    if cfg.get("peft_type", "LORA") != "LORA":
        problems.append(f"peft_type={cfg.get('peft_type')!r} (only LORA is supported)")
    if cfg.get("bias", "none") != "none":
        problems.append(f"bias={cfg['bias']!r}: trained bias terms are not implemented")
    init = cfg.get("init_lora_weights", True)
    if init not in _SAFE_INITS:
        problems.append(
            f"init_lora_weights={init!r}: this initialisation rewrites the base weights, "
            "so the adapter is only correct on that modified base"
        )
    for key in ("rank_pattern", "alpha_pattern"):
        if cfg.get(key):
            problems.append(f"{key}={cfg[key]!r}: per-module rank/alpha is not implemented")
    for key, why in _UNSUPPORTED_OPTIONS.items():
        if cfg.get(key):
            problems.append(f"{key}={cfg[key]!r}: {why}")
    for key, value in cfg.items():
        if key in _BENIGN_KEYS or key in _UNSUPPORTED_OPTIONS:
            continue
        if value:   # unknown and set: be conservative
            problems.append(f"{key}={value!r}: unknown PEFT option; refusing to guess its meaning")
    if not isinstance(cfg.get("r"), int) or cfg["r"] <= 0:
        problems.append(f"r={cfg.get('r')!r} must be a positive int")
    if not isinstance(cfg.get("lora_alpha"), (int, float)):
        problems.append(f"lora_alpha={cfg.get('lora_alpha')!r} must be a number")
    if problems:
        raise AdapterError("unsupported adapter configuration:\n  - " + "\n  - ".join(problems))


def lora_scale(r: int, lora_alpha: float, use_rslora: bool = False) -> float:
    """PEFT's scaling factor: alpha / r, or alpha / sqrt(r) under rsLoRA."""
    return float(lora_alpha) / (math.sqrt(r) if use_rslora else r)


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LoRATensors:
    """One projection's adapter. ``A``: (r, in), ``B``: (out, r), both fp32."""

    A: np.ndarray
    B: np.ndarray
    scale: float

    @property
    def r(self) -> int:
        return int(self.A.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        """``(out, in)`` of the weight this adapter modifies."""
        return int(self.B.shape[0]), int(self.A.shape[1])

    def delta(self) -> np.ndarray:
        """``scale * B @ A`` in fp32, shape (out, in) — what merging adds to W."""
        return (np.float32(self.scale) * (self.B.astype(np.float32) @ self.A.astype(np.float32))).astype(np.float32)


@dataclass(frozen=True)
class LoRAAdapter:
    """A validated LoRA adapter addressed by engine weight name."""

    name: str
    r: int
    lora_alpha: float
    use_rslora: bool
    layers: Mapping[str, LoRATensors]
    peft_config: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None

    @property
    def scale(self) -> float:
        return lora_scale(self.r, self.lora_alpha, self.use_rslora)

    @property
    def target_modules(self) -> tuple[str, ...]:
        mods = {k.split(".")[-2] for k in self.layers}
        return tuple(m for m in SUPPORTED_MODULES if m in mods)

    def keys(self) -> Iterable[str]:
        return self.layers.keys()

    def num_params(self) -> int:
        return sum(t.A.size + t.B.size for t in self.layers.values())

    def validate_against(self, config: Mapping[str, Any]) -> None:
        """Check every adapter matrix fits the base model described by ``config``."""
        shapes = expected_linear_shapes(config)
        n_layers = config["num_hidden_layers"]
        for key, t in self.layers.items():
            parts = key.split(".")
            layer, module = int(parts[2]), parts[4]
            if layer >= n_layers:
                raise AdapterError(f"{key}: layer {layer} >= num_hidden_layers {n_layers}")
            out_dim, in_dim = shapes[module]
            if t.A.shape != (t.r, in_dim) or t.B.shape != (out_dim, t.r):
                raise AdapterError(
                    f"{key}: A{tuple(t.A.shape)} / B{tuple(t.B.shape)} do not fit base weight "
                    f"({out_dim}, {in_dim}); expected A({t.r}, {in_dim}) and B({out_dim}, {t.r})"
                )

    def with_zero_B(self, name: str | None = None) -> "LoRAAdapter":
        """Same adapter with every B set to zero — must reproduce the base model exactly."""
        layers = {k: replace(t, B=np.zeros_like(t.B)) for k, t in self.layers.items()}
        return replace(self, name=name or f"{self.name}-zero", layers=layers)

    def to_peft_tensors(self) -> dict[str, np.ndarray]:
        """Tensors keyed the way PEFT writes them (for round-trip tests)."""
        out: dict[str, np.ndarray] = {}
        for key, t in self.layers.items():
            out[engine_to_peft_key(key, "A")] = t.A
            out[engine_to_peft_key(key, "B")] = t.B
        return out


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def adapter_from_tensors(
    peft_config: Mapping[str, Any],
    tensors: Mapping[str, Any],
    *,
    name: str = "adapter",
    base_config: Mapping[str, Any] | None = None,
    source: str | None = None,
) -> LoRAAdapter:
    """Build a LoRAAdapter from a PEFT config dict and a ``{peft_key: tensor}`` map.

    Tensors may be NumPy arrays or torch tensors of any float dtype; they are
    stored as fp32 NumPy.
    """
    validate_peft_config(peft_config)
    r = int(peft_config["r"])
    alpha = float(peft_config["lora_alpha"])
    rslora = bool(peft_config.get("use_rslora", False))
    scale = lora_scale(r, alpha, rslora)

    halves: dict[str, dict[str, np.ndarray]] = {}
    for key, value in tensors.items():
        engine_key, which = peft_to_engine_key(key)
        arr = _to_fp32_numpy(value)
        if arr.ndim != 2:
            raise AdapterError(f"{key}: expected a 2-D matrix, got shape {arr.shape}")
        slot = halves.setdefault(engine_key, {})
        if which in slot:
            raise AdapterError(f"{key}: duplicate lora_{which} for {engine_key}")
        slot[which] = arr

    if not halves:
        raise AdapterError("adapter contains no LoRA tensors")

    layers: dict[str, LoRATensors] = {}
    for engine_key in sorted(halves, key=_engine_sort_key):
        pair = halves[engine_key]
        if set(pair) != {"A", "B"}:
            missing = ({"A", "B"} - set(pair)).pop()
            raise AdapterError(f"{engine_key}: lora_{missing} is missing")
        A, B = pair["A"], pair["B"]
        if A.shape[0] != r or B.shape[1] != r:
            raise AdapterError(
                f"{engine_key}: rank mismatch, A{A.shape} B{B.shape} vs config r={r} "
                "(per-module ranks need rank_pattern, which is not supported)"
            )
        layers[engine_key] = LoRATensors(A=A, B=B, scale=scale)

    adapter = LoRAAdapter(
        name=name, r=r, lora_alpha=alpha, use_rslora=rslora,
        layers=layers, peft_config=dict(peft_config), source=source,
    )
    if base_config is not None:
        adapter.validate_against(base_config)
    return adapter


def load_peft_adapter(
    path: str | Path,
    *,
    name: str | None = None,
    base_config: Mapping[str, Any] | None = None,
) -> LoRAAdapter:
    """Load ``adapter_config.json`` + ``adapter_model.safetensors`` from ``path``.

    Pass ``base_config`` (the engine's ``load_config(weights)``) to also check
    every matrix against the base model's shapes.
    """
    from safetensors import safe_open

    path = Path(path)
    cfg_path = path / ADAPTER_CONFIG
    w_path = path / ADAPTER_WEIGHTS
    if not cfg_path.is_file():
        raise FileNotFoundError(f"{cfg_path} not found")
    if not w_path.is_file():
        raise FileNotFoundError(
            f"{w_path} not found (only the safetensors format is supported, not adapter_model.bin)"
        )
    peft_config = json.loads(cfg_path.read_text())
    tensors: dict[str, Any] = {}
    with safe_open(str(w_path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return adapter_from_tensors(
        peft_config, tensors, name=name or path.name,
        base_config=base_config, source=str(path),
    )


def save_peft_adapter(adapter: LoRAAdapter, path: str | Path) -> Path:
    """Write ``adapter`` back out in PEFT's on-disk format (used by tests/benchmarks)."""
    import torch
    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    cfg = dict(adapter.peft_config) or {
        "peft_type": "LORA", "r": adapter.r, "lora_alpha": adapter.lora_alpha,
        "use_rslora": adapter.use_rslora, "bias": "none",
        "target_modules": list(adapter.target_modules),
    }
    (path / ADAPTER_CONFIG).write_text(json.dumps(cfg, indent=2))
    tensors = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in adapter.to_peft_tensors().items()}
    save_file(tensors, str(path / ADAPTER_WEIGHTS))
    return path


def _to_fp32_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    # torch tensor (bf16 has no NumPy dtype, so go through float())
    return value.detach().to("cpu").float().numpy()


def _engine_sort_key(key: str) -> tuple[int, int]:
    parts = key.split(".")
    return int(parts[2]), list(SUPPORTED_MODULES).index(parts[4])
