"""The analytic parameter counts must equal what an injected model actually trains."""
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from playparse.lora import LoRAConfig, count_total, count_trainable, inject_lora
from playparse.lora.memory import (
    TARGET_SETS,
    full_ft_memory,
    full_param_count,
    load_config,
    lora_memory,
    lora_param_count,
    param_table,
)
from playparse.paths import WEIGHTS
from tests._lora_util import TINY


def _meta_model(cfg: dict) -> LlamaForCausalLM:
    with torch.device("meta"):
        return LlamaForCausalLM(LlamaConfig(**cfg))


def _check_counts(cfg: dict, ranks=(4, 16)) -> None:
    assert full_param_count(cfg) == count_total(_meta_model(cfg))
    for targets in TARGET_SETS.values():
        for r in ranks:
            model = inject_lora(_meta_model(cfg), LoRAConfig(r=r, alpha=2 * r, target_modules=list(targets)))
            assert count_trainable(model) == lora_param_count(cfg, r, targets), (targets, r)


def test_counts_match_injected_tiny_model():
    _check_counts(dict(TINY))
    _check_counts({**TINY, "tie_word_embeddings": False})


@pytest.mark.skipif(not (WEIGHTS / "config.json").exists(), reason="no real config.json")
def test_counts_match_injected_real_1b_config_on_meta():
    cfg = load_config(WEIGHTS)
    assert full_param_count(cfg) == 1_235_814_400  # Llama 3.2 1B, tied embeddings
    _check_counts(cfg, ranks=(4, 8, 16, 64))


def test_memory_arithmetic():
    assert full_ft_memory(10).total == 160  # 16 bytes per parameter
    est = lora_memory(n_total=100, n_lora=10)
    assert est.weights == 200 + 40 and est.grads == 40 and est.master == 0
    assert est.total == 200 + 16 * 10


def test_param_table_shape():
    rows = param_table(dict(TINY), ranks=(1, 2, 4, 8))  # r=64 would exceed a hidden-64 layer
    assert rows[0]["method"] == "full" and len(rows) == 1 + 3 * 4
    lora = [r for r in rows if r["method"] == "lora"]
    assert all(r["trainable"] < rows[0]["trainable"] for r in lora)
