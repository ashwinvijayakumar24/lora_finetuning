"""Shared helpers for the LoRA tests: a tiny random Llama and PEFT weight copying."""
from __future__ import annotations

import copy

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from playparse.lora import LoRAConfig, lora_modules

TINY = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=128,
    max_position_embeddings=64,
    tie_word_embeddings=True,
)


def tiny_llama(seed: int = 0, **overrides) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    model = LlamaForCausalLM(LlamaConfig(**{**TINY, **overrides}))
    return model.eval()


def tiny_pair(seed: int = 0) -> tuple[LlamaForCausalLM, LlamaForCausalLM]:
    """Two models with identical base weights: one for us, one for PEFT."""
    a = tiny_llama(seed)
    return a, copy.deepcopy(a)


def input_ids(batch: int = 2, seq: int = 12, vocab: int = 128, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (batch, seq), generator=g)


def randomize_lora(model: torch.nn.Module, seed: int = 2, std: float = 0.05) -> None:
    """Give every B a nonzero value; with B = 0 the A gradient is zero and tests are vacuous."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, m in lora_modules(model):
            m.lora_B.weight.copy_(torch.randn(m.lora_B.weight.shape, generator=g) * std)


def peft_config(cfg: LoRAConfig):
    from peft import LoraConfig

    targets = cfg.target_modules if cfg.target_modules == "all-linear" else list(cfg.target_modules)
    return LoraConfig(
        r=cfg.r, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout, target_modules=targets
    )


def peft_lora_params(peft_model) -> dict[str, torch.nn.Parameter]:
    """PEFT adapter params keyed the way `lora_state_dict` keys ours."""
    out = {}
    for name, p in peft_model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            key = name.removeprefix("base_model.model.").replace(".default.", ".")
            out[key] = p
    return out


def copy_ours_to_peft(ours: torch.nn.Module, peft_model) -> None:
    theirs = peft_lora_params(peft_model)
    mine = {f"{n}.lora_{ab}.weight": getattr(m, f"lora_{ab}").weight
            for n, m in lora_modules(ours) for ab in ("A", "B")}
    assert mine.keys() == theirs.keys()
    with torch.no_grad():
        for k, p in mine.items():
            theirs[k].copy_(p)
