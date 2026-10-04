"""Analytic parameter counts and training-memory estimates for Llama models.

Why LoRA saves memory, in numbers. Training with AdamW holds, per *trainable*
parameter, a gradient plus two optimizer moments (m and v, kept in fp32), and
in mixed precision an fp32 master copy of the weight as well. A frozen
parameter costs only its own storage. So the bill is roughly:

    full fine-tuning: N_total x (2 weight + 2 grad + 4 master + 4 m + 4 v) = 16 B/param
    LoRA:             N_total x 2 (frozen bf16 base)
                      + N_lora x (4 weight + 4 grad + 4 m + 4 v)    = 16 B/adapter param

The adapter is already fp32, so it needs no separate master copy. Activations
are excluded: they depend on batch size and sequence length, not on the method,
and are measured separately (scripts/p0_param_table.py --measure).

A LoRA adapter on a linear layer of shape (in -> out) adds r x (in + out)
parameters: A is (r x in) and B is (out x r).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from playparse.lora.config import ALL_LINEAR

TARGET_SETS: dict[str, tuple[str, ...]] = {
    "qv": ("q_proj", "v_proj"),
    "qkvo": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "all-linear": ALL_LINEAR,
}

BF16, FP32 = 2, 4


def load_config(path: str | Path) -> dict[str, Any]:
    """Read an HF `config.json` (a file or the directory holding it)."""
    p = Path(path)
    if p.is_dir():
        p = p / "config.json"
    return json.loads(p.read_text())


def linear_shapes(cfg: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """(in_features, out_features) of each projection in one decoder layer."""
    h = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    head_dim = cfg.get("head_dim") or h // cfg["num_attention_heads"]
    q_out = cfg["num_attention_heads"] * head_dim
    kv_out = cfg.get("num_key_value_heads", cfg["num_attention_heads"]) * head_dim
    return {
        "q_proj": (h, q_out),
        "k_proj": (h, kv_out),
        "v_proj": (h, kv_out),
        "o_proj": (q_out, h),
        "gate_proj": (h, inter),
        "up_proj": (h, inter),
        "down_proj": (inter, h),
    }


def full_param_count(cfg: dict[str, Any]) -> int:
    """All parameters of a LlamaForCausalLM, tied embeddings counted once."""
    if cfg.get("attention_bias") or cfg.get("mlp_bias"):
        raise NotImplementedError("bias terms are not counted")
    h, vocab, layers = cfg["hidden_size"], cfg["vocab_size"], cfg["num_hidden_layers"]
    per_layer = sum(i * o for i, o in linear_shapes(cfg).values()) + 2 * h  # + two RMSNorms
    embed = vocab * h
    lm_head = 0 if cfg.get("tie_word_embeddings", False) else vocab * h
    return embed + layers * per_layer + h + lm_head  # + final norm


def lora_param_count(cfg: dict[str, Any], r: int, targets: tuple[str, ...]) -> int:
    shapes = linear_shapes(cfg)
    unknown = set(targets) - shapes.keys()
    if unknown:
        raise ValueError(f"unknown targets {sorted(unknown)}")
    return cfg["num_hidden_layers"] * sum(r * (shapes[t][0] + shapes[t][1]) for t in targets)


@dataclass(frozen=True)
class MemoryEstimate:
    """Bytes held during training, excluding activations and allocator overhead."""

    weights: int
    grads: int
    master: int
    adam_m: int
    adam_v: int

    @property
    def total(self) -> int:
        return self.weights + self.grads + self.master + self.adam_m + self.adam_v

    def as_dict(self) -> dict[str, int]:
        return {**asdict(self), "total": self.total}


def full_ft_memory(n_total: int) -> MemoryEstimate:
    """Mixed-precision AdamW: bf16 working weights and grads, fp32 master, m, v."""
    return MemoryEstimate(
        weights=BF16 * n_total,
        grads=BF16 * n_total,
        master=FP32 * n_total,
        adam_m=FP32 * n_total,
        adam_v=FP32 * n_total,
    )


def lora_memory(n_total: int, n_lora: int) -> MemoryEstimate:
    """Frozen bf16 base (no grads, no optimizer state) plus fp32 adapters under AdamW."""
    return MemoryEstimate(
        weights=BF16 * n_total + FP32 * n_lora,
        grads=FP32 * n_lora,
        master=0,
        adam_m=FP32 * n_lora,
        adam_v=FP32 * n_lora,
    )


def param_table(
    cfg: dict[str, Any], ranks: tuple[int, ...] = (4, 8, 16, 64)
) -> list[dict[str, Any]]:
    """One row for full fine-tuning, then one per (target set, rank)."""
    n_total = full_param_count(cfg)
    rows = [
        {
            "method": "full",
            "targets": "-",
            "r": None,
            "trainable": n_total,
            "trainable_pct": 100.0,
            "memory_bytes": full_ft_memory(n_total).as_dict(),
        }
    ]
    for name, targets in TARGET_SETS.items():
        for r in ranks:
            n = lora_param_count(cfg, r, targets)
            rows.append(
                {
                    "method": "lora",
                    "targets": name,
                    "r": r,
                    "trainable": n,
                    "trainable_pct": 100.0 * n / n_total,
                    "memory_bytes": lora_memory(n_total, n).as_dict(),
                }
            )
    return rows
