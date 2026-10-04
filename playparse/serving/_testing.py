"""Builders shared by the P5a tests and benchmark: tiny Llama checkpoints and PEFT adapters.

Test adapters are made with PEFT itself, so the on-disk format under test is
exactly what training will produce. PEFT initialises ``B = 0`` (an adapter that
does nothing), which would make every "adapter changes the output" check
vacuous, so :func:`make_peft_adapter` overwrites ``B`` with seeded Gaussian noise.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ALL_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# A Llama-3-shaped model small enough for millisecond tests. head_dim, GQA ratio
# and the llama3 RoPE scaling are kept so the same code paths run as on the 1B.
TINY_CONFIG: dict[str, Any] = dict(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=256,
    rms_norm_eps=1e-5,
    rope_theta=500000.0,
    rope_scaling={
        "factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
        "original_max_position_embeddings": 64, "rope_type": "llama3",
    },
    tie_word_embeddings=True,
    initializer_range=0.1,
)


def build_tiny_checkpoint(out_dir: str | Path, seed: int = 0):
    """Save a random tiny Llama in HF format that the engine can also load.

    Returns the in-memory HF model (fp32, eval mode).

    transformers 5 writes RoPE settings as ``rope_parameters`` and drops the
    top-level ``rope_theta``/``rope_scaling`` keys the engine reads
    (``engine/model.py`` reads ``config["rope_theta"]``), so they are written back
    into config.json here. See docs/issues/p5a-transformers5-rope-config.md.
    """
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(**TINY_CONFIG)
    model = LlamaForCausalLM(cfg).float().eval()
    out_dir = Path(out_dir)
    model.save_pretrained(out_dir, safe_serialization=True)
    cfg_path = out_dir / "config.json"
    saved = json.loads(cfg_path.read_text())
    saved["rope_theta"] = TINY_CONFIG["rope_theta"]
    saved["rope_scaling"] = dict(TINY_CONFIG["rope_scaling"])
    saved.setdefault("head_dim", TINY_CONFIG["head_dim"])
    cfg_path.write_text(json.dumps(saved, indent=2))
    return model


def make_peft_adapter(
    base_model,
    out_dir: str | Path,
    *,
    r: int = 8,
    lora_alpha: float = 16,
    target_modules=ALL_TARGETS,
    b_std: float = 0.02,
    seed: int = 0,
    use_rslora: bool = False,
    zero_b: bool = False,
):
    """Wrap ``base_model`` with PEFT LoRA, randomise B, save to ``out_dir``.

    Returns the PeftModel (still wrapping ``base_model``, in eval mode) so the
    caller can use it as the HF + PEFT oracle. Call ``.unload()`` on it to get
    the base model back unmodified.
    """
    import torch
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r, lora_alpha=lora_alpha, target_modules=list(target_modules),
        lora_dropout=0.0, bias="none", use_rslora=use_rslora, task_type="CAUSAL_LM",
    )
    gen = torch.Generator().manual_seed(seed)
    peft_model = get_peft_model(base_model, cfg)
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if "lora_A" in name:
                # PEFT's kaiming init uses the global RNG; reseed for reproducibility.
                p.copy_(torch.empty_like(p).uniform_(-1, 1, generator=gen) / p.shape[1] ** 0.5)
            elif "lora_B" in name:
                if zero_b:
                    p.zero_()
                else:
                    p.copy_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * b_std)
    peft_model.eval()
    peft_model.save_pretrained(str(out_dir))
    return peft_model
