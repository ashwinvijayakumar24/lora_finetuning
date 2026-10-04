"""Slow: the real Llama 3.2 1B + PEFT LoRA through the loop for a few steps.

Run with:  pytest -m slow tests/test_p2_smoke_slow.py
The full benchmark is scripts/p2_smoke_train.py (writes results/p2/smoke.json).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from playparse.paths import WEIGHTS

REPO = Path(__file__).resolve().parent.parent


@pytest.mark.slow
def test_real_model_loss_decreases(tmp_path):
    if not (WEIGHTS / "model.safetensors").exists():
        pytest.skip(f"no weights under {WEIGHTS}")
    spec = importlib.util.spec_from_file_location("p2_smoke", REPO / "scripts" / "p2_smoke_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    device = "mps" if torch.backends.mps.is_available() else "auto"
    res = mod.run_smoke(steps=6, n_train=24, n_val=8, micro=1, accum=4, lr=2e-4, device=device, gen=False,
                        output_dir=str(tmp_path))
    train = res["loss"]["train"]
    val = list(res["loss"]["val"].values())
    assert train[-1] < train[0]
    assert val[-1] < val[0]
    assert res["params"]["trainable_pct"] < 2.0
    assert res["adapter_checkpoint_bytes"] < 100 * 2**20  # adapter-only: tens of MB
