"""P5a oracle on the real Llama 3.2 1B Instruct: engine + adapter vs HF + PEFT.

Slow (loads the 1B model twice: HF fp32, then the engine fp16). Run with ``pytest -m slow tests/test_p5a_real_oracle.py -s``.

Memory budget on a 16 GB laptop decides the structure. HF fp32 is ~5 GB, and
a second fp32 copy for the engine would double that (the first attempt did, and
thrashed swap; see docs/issues/p5a-oracle-memory.md). So the fp32 engine runs
on zero-copy views of the HF parameters, each fixture computes everything it
needs up front, stores small results (logits for one prompt, token lists),
frees the model, and the tests only assert on those results.

Oracles and what each one proves:

* NumPy fp32 engine vs HF fp32 + PEFT — the tight oracle (same precision on
  both sides): logits within 1e-3 (the engine's own CPU tolerance,
  ``tests/test_forward.py``), identical greedy tokens, for merged and unmerged.
* torch fp16 engine (MPS locally, CUDA under ``-m gpu``) — claim L2: merged and
  unmerged produce identical greedy tokens and logits within fp16 tolerance of
  each other, and both stay close to the fp32 oracle.
* Zero adapter (B = 0) reproduces the base engine bit for bit, in both modes.

Set ``PLAYPARSE_RECORD=1`` to write the achieved tolerances to
``results/p5a/oracle_<device>.json``.
"""
from __future__ import annotations

import gc
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from playparse.paths import RESULTS, WEIGHTS
from playparse.serving._testing import make_peft_adapter
from playparse.serving.adapter import load_peft_adapter
from playparse.serving.lora_engine import LoRALlamaModel, LoRAModelGPU, generate_with_adapter
from playparse.serving.merge import merge_adapter

from engine.loader import load_config, load_weights_gpu
from engine.model import LlamaModel
from engine.model_gpu import LlamaModelGPU
from engine.sampler import greedy
from engine.scheduler import generate

pytestmark = pytest.mark.slow

N_NEW = 16
ADAPTER_KW = dict(r=8, lora_alpha=16, b_std=0.01, seed=0)   # b_std chosen so greedy text changes
FP32_ATOL = 1e-3
# Depth of the model under test. 16 is the real model and the default. On a
# memory-starved laptop, PLAYPARSE_ORACLE_LAYERS=4 runs the same oracle on the
# real weights of the first 4 layers (HF and the engine both truncated
# identically; embedding, final norm and tied LM head unchanged), ~1.5 GB in
# fp32 instead of ~5 GB. Recorded results carry the depth used.
N_LAYERS = int(os.environ.get("PLAYPARSE_ORACLE_LAYERS", "16"))
PLAY = "Parse this play: J.Hurts pass short right to A.Brown for 14 yards (C.Gardner-Johnson)."

_RECORD: dict = {}


def _weights_dir() -> Path:
    if not (WEIGHTS / "model.safetensors").is_file():
        pytest.skip(f"real weights not found at {WEIGHTS} (set PLAYPARSE_WEIGHTS)")
    return WEIGHTS


def _prompt_ids(weights_dir: Path) -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(weights_dir))
    ids = tok.apply_chat_template([{"role": "user", "content": PLAY}], add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return [int(i) for i in ids]


def _free():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _np_greedy(model, ids, adapter=None):
    kw = dict(max_tokens=N_NEW, max_seq=len(ids) + N_NEW + 1)
    if adapter is None:
        return list(generate(model, ids, greedy, **kw))
    return list(generate_with_adapter(model, ids, greedy, adapter, **kw))


@pytest.fixture(scope="module")
def hf_refs(tmp_path_factory):
    """HF fp32 + PEFT references AND the fp32 NumPy engine results, sharing one copy of the weights.

    The engine's NumPy weight dict is built from the HF model's own fp32
    parameters as zero-copy NumPy views instead of a second ``load_weights``
    (which would hold another ~5 GB). The values are identical either way: both
    are the bf16 checkpoint cast to fp32, which is exact; one tensor is checked
    against the safetensors file to make sure. The order below is load-bearing:
    HF references first, then unmerged engine runs, then the in-place merge.
    """
    from safetensors import safe_open
    from transformers import LlamaForCausalLM

    wdir = _weights_dir()
    ids = _prompt_ids(wdir)
    cfg = {**load_config(str(wdir)), "num_hidden_layers": N_LAYERS}
    ad_dir = tmp_path_factory.mktemp("p5a_real") / "adapter_r8"
    hf = LlamaForCausalLM.from_pretrained(str(wdir), dtype=torch.float32, attn_implementation="eager",
                                          num_hidden_layers=N_LAYERS).eval()
    x = torch.tensor([ids])
    gen_kw = dict(max_new_tokens=N_NEW, do_sample=False, attention_mask=torch.ones_like(x), pad_token_id=128001)
    refs: dict = dict(ids=ids, cfg=cfg, wdir=wdir)
    np_out: dict = {}

    with torch.no_grad():
        refs["base_logits"] = hf(x).logits[0].numpy()
        refs["base_greedy"] = hf.generate(x, **gen_kw)[0, len(ids):].tolist()

        weights = {k: v.detach().numpy() for k, v in hf.state_dict().items()}
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
        probe = f"model.layers.{N_LAYERS - 1}.mlp.down_proj.weight"
        with safe_open(str(wdir / "model.safetensors"), framework="pt") as f:
            assert np.array_equal(f.get_tensor(probe).float().numpy(), weights[probe])
        np_out["base_logits"] = LlamaModel(weights, cfg).forward(ids)

        pm = make_peft_adapter(hf, ad_dir, **ADAPTER_KW)
        refs["ad_logits"] = pm(x).logits[0].numpy()
        refs["ad_greedy"] = pm.generate(x, **gen_kw)[0, len(ids):].tolist()
        hf = pm.unload()     # LoRA layers removed; base parameters (and our views) untouched
        # Drop the HF model now. `weights` keeps each parameter's storage alive through its
        # NumPy view, so the in-place merge below frees each old matrix as it replaces it
        # (peak stays ~5 GB instead of ~9 GB).
        del pm, hf
        _free()

    adapter = load_peft_adapter(ad_dir, name="r8", base_config=cfg)
    refs["adapter"] = adapter

    um = LoRALlamaModel(weights, cfg)
    um.load_adapter(adapter, "r8")
    um.load_adapter(adapter.with_zero_B("zero"), "zero")
    np_out["unmerged_logits"] = um.forward(ids, adapter="r8")
    np_out["unmerged_none_logits"] = um.forward(ids, adapter=None)
    np_out["zero_unmerged_logits"] = um.forward(ids, adapter="zero")
    np_out["unmerged_greedy"] = _np_greedy(um, ids, adapter="r8")
    del um

    merge_adapter(weights, adapter, inplace=True)
    merged = LlamaModel(weights, cfg)
    np_out["merged_logits"] = merged.forward(ids)
    np_out["merged_greedy"] = _np_greedy(merged, ids)
    del merged, weights
    _free()
    refs["np"] = np_out
    return refs


@pytest.fixture(scope="module")
def np_results(hf_refs):
    return hf_refs["np"]


def _torch_device_params():
    local = "mps" if torch.backends.mps.is_available() else "cpu"
    return [pytest.param(local, id=local), pytest.param("cuda", id="cuda", marks=pytest.mark.gpu)]


@pytest.fixture(scope="module", params=_torch_device_params())
def torch_results(request):
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    hf_refs = request.getfixturevalue("hf_refs")
    ids, cfg, ad = hf_refs["ids"], hf_refs["cfg"], hf_refs["adapter"]
    weights = load_weights_gpu(str(hf_refs["wdir"]), cfg, device=device)
    weights = {k: v for k, v in weights.items()
               if not k.startswith("model.layers.") or int(k.split(".")[2]) < N_LAYERS}
    out = {"device": device}
    kw = dict(max_tokens=N_NEW, max_seq=len(ids) + N_NEW + 1)

    base = LlamaModelGPU(weights, cfg, device=device)
    out["base_logits"] = base.forward_all(ids)
    out["base_greedy"] = list(generate(base, ids, greedy, **kw))

    um = LoRAModelGPU(weights, cfg, device=device)
    um.load_adapter(ad, "r8")
    um.load_adapter(ad.with_zero_B("zero"), "zero")
    out["unmerged_logits"] = um.forward_all(ids, adapter="r8")
    out["zero_unmerged_logits"] = um.forward_all(ids, adapter="zero")
    out["unmerged_greedy"] = list(generate_with_adapter(um, ids, greedy, "r8", **kw))
    out["zero_unmerged_greedy"] = list(generate_with_adapter(um, ids, greedy, "zero", **kw))
    del um
    _free()

    zero_merged = LlamaModelGPU(merge_adapter(weights, ad.with_zero_B()), cfg, device=device)
    out["zero_merged_logits"] = zero_merged.forward_all(ids)
    del zero_merged
    _free()

    merged = LlamaModelGPU(merge_adapter(weights, ad), cfg, device=device)
    out["merged_logits"] = merged.forward_all(ids)
    out["merged_greedy"] = list(generate(merged, ids, greedy, **kw))
    del merged, base, weights
    _free()
    return out


def _maxabs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def _record(device: str, **values):
    _RECORD.setdefault(device, {}).update(values)


# ---------------------------------------------------------------- sanity

def test_adapter_is_not_vacuous(hf_refs):
    assert _maxabs(hf_refs["ad_logits"], hf_refs["base_logits"]) > 0.5
    # A single greedy string is fragile: on PACE's x86 CPUs the random adapter left
    # one 16-token continuation unchanged even though logits moved by > 0.5. The top
    # token at *some* of the 63 teacher-forced prompt positions must change.
    changed = (hf_refs["ad_logits"].argmax(-1) != hf_refs["base_logits"].argmax(-1)).sum()
    assert changed >= 1, "adapter should change the top token somewhere in the prompt"


# ---------------------------------------------------------------- fp32 NumPy engine (tight oracle)

def test_np_base_matches_hf(hf_refs, np_results):
    err = _maxabs(np_results["base_logits"], hf_refs["base_logits"])
    _record("cpu_fp32", base_vs_hf_max_abs=err)
    assert err < FP32_ATOL


def test_np_unmerged_matches_peft(hf_refs, np_results):
    err = _maxabs(np_results["unmerged_logits"], hf_refs["ad_logits"])
    _record("cpu_fp32", unmerged_vs_peft_max_abs=err)
    assert err < FP32_ATOL


def test_np_merged_matches_peft(hf_refs, np_results):
    err = _maxabs(np_results["merged_logits"], hf_refs["ad_logits"])
    _record("cpu_fp32", merged_vs_peft_max_abs=err,
            merged_vs_unmerged_max_abs=_maxabs(np_results["merged_logits"], np_results["unmerged_logits"]))
    assert err < FP32_ATOL


def test_np_greedy_tokens_identical(hf_refs, np_results):
    _record("cpu_fp32", greedy_tokens=N_NEW, greedy_peft=hf_refs["ad_greedy"])
    assert np_results["unmerged_greedy"] == hf_refs["ad_greedy"]
    assert np_results["merged_greedy"] == hf_refs["ad_greedy"]


def test_np_zero_adapter_and_unselected_are_base(np_results):
    # (Zero adapter *merged* on the fp32 path is W + 0 = W; it is checked on the
    # fp16 path below and on the tiny model, to avoid a second 4 GB copy here.)
    base = np_results["base_logits"]
    assert np.array_equal(np_results["zero_unmerged_logits"], base)
    assert np.array_equal(np_results["unmerged_none_logits"], base)


# ---------------------------------------------------------------- fp16 torch engine (claim L2)

def test_torch_merged_equals_unmerged_greedy(torch_results):
    """Claim L2: token-identical greedy output, merged vs unmerged."""
    d = torch_results["device"]
    err = _maxabs(torch_results["merged_logits"], torch_results["unmerged_logits"])
    _record(f"{d}_fp16", merged_vs_unmerged_max_abs=err,
            merged_greedy=torch_results["merged_greedy"], unmerged_greedy=torch_results["unmerged_greedy"])
    assert torch_results["merged_greedy"] == torch_results["unmerged_greedy"]
    assert err < 0.25, f"merged vs unmerged fp16 logits differ by {err}"


def test_torch_close_to_fp32_oracle(hf_refs, torch_results):
    d = torch_results["device"]
    ref = hf_refs["ad_logits"]
    errs = {m: _maxabs(torch_results[f"{m}_logits"], ref) for m in ("merged", "unmerged")}
    agree = sum(a == b for a, b in zip(torch_results["merged_greedy"], hf_refs["ad_greedy"]))
    _record(f"{d}_fp16", merged_vs_peft_max_abs=errs["merged"], unmerged_vs_peft_max_abs=errs["unmerged"],
            base_vs_hf_max_abs=_maxabs(torch_results["base_logits"], hf_refs["base_logits"]),
            greedy_agreement_with_fp32_peft=f"{agree}/{N_NEW}")
    for m in ("merged", "unmerged"):
        got = torch_results[f"{m}_logits"]
        # Same bar as the engine's GPU oracle (tests/test_gpu_oracle.py): argmax and top-10 overlap
        # at every prompt position, plus a bounded max error.
        assert (got.argmax(-1) == ref.argmax(-1)).mean() >= 0.95, m
        top_g = np.argsort(got[-1])[-10:]
        top_r = np.argsort(ref[-1])[-10:]
        assert len(set(top_g) & set(top_r)) >= 8, m
        assert errs[m] < 0.5, f"{m}: fp16 vs fp32 max |dlogit| {errs[m]}"


def test_torch_zero_adapter_is_base(torch_results):
    base = torch_results["base_logits"]
    assert np.array_equal(torch_results["zero_merged_logits"], base)
    # Unmerged adds (x@A^T)@0 = exact zeros, so the sum is bit-identical too.
    assert np.array_equal(torch_results["zero_unmerged_logits"], base)
    assert torch_results["zero_unmerged_greedy"] == torch_results["base_greedy"]


# ---------------------------------------------------------------- optional artifact

def teardown_module(module):
    if os.environ.get("PLAYPARSE_RECORD") != "1" or not _RECORD:
        return
    out_dir = RESULTS / "p5a"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                      cwd=Path(__file__).resolve().parent).strip()
    except Exception:  # pragma: no cover
        sha = "unknown"
    meta = {"host": platform.node(), "platform": platform.platform(), "torch": torch.__version__,
            "git_sha": sha, "adapter": {**ADAPTER_KW, "targets": "q,k,v,o,gate,up,down"},
            "prompt": PLAY, "n_layers": N_LAYERS, "label": "local, Apple M4; fp32 rows are the authoritative oracle"}
    (out_dir / f"oracle_local_L{N_LAYERS}.json").write_text(json.dumps({"meta": meta, "results": _RECORD}, indent=2))
