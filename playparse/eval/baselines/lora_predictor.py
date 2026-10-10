"""The LoRA rung: the base model plus a trained adapter, scored by the same harness.

R5 and the sweep runs (and the P3 local pilot) produce PEFT-format adapter
directories. A QLoRA adapter (R6) records `model.quant: nf4` in its run_spec.json
and is scored on the 4-bit base by default (`--base-quant auto`); `--base-quant none`
scores the same adapter on the bf16 base.

This predictor loads the base model exactly as R1 does, injects the
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


def _run_spec(adapter_dir: str | Path) -> dict | None:
    d = Path(adapter_dir).resolve()
    cands = [d.parent / "run_spec.json"]
    if len(d.parents) > 2:
        cands.append(d.parents[2] / "run_spec.json")
    for cand in cands:
        if cand.is_file():
            return json.loads(cand.read_text())
    return None


def trained_schema(adapter_dir: str | Path) -> str | None:
    """The output schema an adapter was trained on ("v1"/"v2"), from its run_spec.json.

    Runs from before T3b have no data.schema: their style decides (all v1).
    """
    from playparse.prompt import schema_of_style

    spec = _run_spec(adapter_dir)
    if spec is None:
        return None
    data = spec.get("data") or {}
    return data.get("schema") or schema_of_style(data.get("prompt_style", "full"))


def trained_base_quant(adapter_dir: str | Path) -> str | None:
    """The base quantization an adapter was trained on (R6: "nf4"), from its run_spec.json.

    None for a bf16 base, for runs from before model.quant existed, and for an
    adapter with no run_spec.json nearby.
    """
    from playparse.train.build import normalize_quant

    spec = _run_spec(adapter_dir)
    return normalize_quant((spec or {}).get("model", {}).get("quant"))


def resolve_base_quant(adapter_dir: str | Path, requested: str | None) -> str | None:
    """--base-quant: "auto" (default) = as trained; "none" or "nf4" = that base, whatever the training."""
    from playparse.train.build import normalize_quant

    if requested in (None, "auto"):
        return trained_base_quant(adapter_dir)
    return normalize_quant(requested)


def trained_prompt_style(adapter_dir: str | Path) -> str | None:
    """The prompt style an adapter was trained with, if its run directory says so.

    scripts/train.py writes <output_dir>/run_spec.json and saves adapters under
    <output_dir>/checkpoints/step_N/adapter. A run_spec without data.prompt_style
    predates the option and was trained with "full". Returns None when no
    run_spec.json is found (an adapter copied elsewhere).
    """
    d = Path(adapter_dir).resolve()
    cands = [d.parent / "run_spec.json"]
    if len(d.parents) > 2:
        cands.append(d.parents[2] / "run_spec.json")
    for cand in cands:
        if cand.is_file():
            spec = json.loads(cand.read_text())
            return (spec.get("data") or {}).get("prompt_style", "full")
    return None


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
        # An adapter scored on a prompt it was not trained on still produces output,
        # just worse; refuse that when the run directory records the style.
        trained = trained_prompt_style(self.adapter_dir)
        style = kw.get("prompt_style", "full")
        if trained is not None and trained != style:
            raise ValueError(f"{self.adapter_dir} was trained with prompt_style={trained!r}, "
                             f"not {style!r}; pass --prompt-style {trained}")
        from playparse.prompt import schema_of_style

        schema = trained_schema(self.adapter_dir)
        if schema is not None and schema != schema_of_style(style):
            raise ValueError(f"{self.adapter_dir} was trained on schema {schema!r}; prompt style {style!r} "
                             f"decodes schema {schema_of_style(style)!r}")
        # A QLoRA adapter (R6) is scored on the 4-bit base it was trained on unless
        # the caller asks for another base (e.g. --base-quant none: the same adapter
        # on the bf16 base, which isolates what the 4-bit base itself costs).
        kw["base_quant"] = resolve_base_quant(self.adapter_dir, kw.get("base_quant"))
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
