"""PEFT oracle on the real Llama 3.2 1B weights (CPU, fp32). Run with `-m slow`.

Only one fp32 copy of the model (~5 GB) is held: we inject our adapters, record
logits, unwrap them (no merge), and let PEFT wrap the same base in place from
the adapter we just saved. That also exercises the PEFT-format writer at full
scale (16 layers x 7 projections).
"""
import gc

import pytest
import torch

from playparse.lora import LoRAConfig, inject_lora, lora_modules, save_adapter, unload_lora
from playparse.paths import WEIGHTS

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def real_model():
    if not (WEIGHTS / "config.json").exists():
        pytest.skip(f"no weights at {WEIGHTS}")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(WEIGHTS, dtype=torch.float32).eval()
    yield model
    del model
    gc.collect()


def test_real_1b_logits_match_peft(real_model, tmp_path):
    from peft import PeftModel
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(WEIGHTS)
    ids = tok("Pass short right to A.Brown for 14 yards.", return_tensors="pt").input_ids

    with torch.no_grad():
        base_logits = real_model(ids).logits

    cfg = LoRAConfig(r=16, alpha=32, dropout=0.0)
    inject_lora(real_model, cfg)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for _, m in lora_modules(real_model):
            m.lora_B.weight.copy_(torch.randn(m.lora_B.weight.shape, generator=g) * 0.01)
        ours = real_model(ids).logits
    save_adapter(real_model, cfg, tmp_path)
    unload_lora(real_model)

    pm = PeftModel.from_pretrained(real_model, tmp_path).eval()
    with torch.no_grad():
        theirs = pm(input_ids=ids).logits
    pm.unload()

    assert (ours - base_logits).abs().max().item() > 1e-2  # adapter is not a no-op
    assert (ours - theirs).abs().max().item() < 1e-5
