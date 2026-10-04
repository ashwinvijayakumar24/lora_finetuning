"""The LoRA rung: the base model plus a trained adapter, scored by the same harness.

R5 and the sweep runs (and the P3 local pilot) produce PEFT-format adapter
directories. This predictor loads the base model exactly as R1 does, injects the
adapter with `playparse.lora.load_adapter`, and then decodes through
`HFPredictor.predict_batch` unchanged. So the prompt rendering (zero-shot chat
template with the pinned date, the same ids training used; see
tests/test_prompt_parity.py), tokenization, left padding, greedy decoding, stop
tokens, and output decoding are identical to R1. The only difference between R1
and this rung is the adapter.

`merge=True` folds the adapter into the base weights first (`merge_lora`), which
removes the extra matmuls. Greedy outputs should be the same; on a half-precision
base, merging rounds the merged weight, so rare near-tie tokens can differ (see
docs/issues/p0-bf16-unmerge-drift.md). The flag is part of the config hash.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from playparse.eval.baselines.hf_predictor import HFPredictor


def adapter_fingerprint(adapter_dir: str | Path) -> dict[str, Any]:
    """What identifies an adapter: sha256 of its weights and its LoRA config."""
    d = Path(adapter_dir)
    h = hashlib.sha256()
    with open(d / "adapter_model.safetensors", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    cfg = json.loads((d / "adapter_config.json").read_text())
    return {
        "adapter_sha256": h.hexdigest(),
        "adapter_r": cfg.get("r"),
        "adapter_alpha": cfg.get("lora_alpha"),
        "adapter_targets": cfg.get("target_modules"),
    }


class LoRAPredictor(HFPredictor):
    """HFPredictor (zero-shot prompt) with a playparse/PEFT LoRA adapter applied."""

    def __init__(self, weights: str | Path, adapter_dir: str | Path, *, merge: bool = False,
                 name: str | None = None, **kw):
        import torch

        from playparse.lora import load_adapter, merge_lora

        self.adapter_dir = Path(adapter_dir)
        if not (self.adapter_dir / "adapter_config.json").is_file():
            raise FileNotFoundError(f"{self.adapter_dir} is not a PEFT adapter directory")
        self.merge = merge
        self.fingerprint = adapter_fingerprint(self.adapter_dir)
        super().__init__(weights, name=name or "lora", examples_fn=None, examples_id=None, **kw)
        # Adapters stay fp32 (as trained); LoRALinear casts its input and output.
        load_adapter(self.model, self.adapter_dir, adapter_dtype=torch.float32)
        if merge:
            merge_lora(self.model)
        self.model.eval()

    def config(self) -> dict[str, Any]:
        return {**super().config(), "adapter": self.adapter_dir.name, "merged": self.merge, **self.fingerprint}


def make_lora(weights: str | Path, adapter_dir: str | Path, merge: bool = False, **kw) -> LoRAPredictor:
    return LoRAPredictor(weights, adapter_dir, merge=merge, **kw)
