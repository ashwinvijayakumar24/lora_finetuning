"""P5b correctness gate on the real Llama 3.2 1B (slow; one CUDA + FlashInfer arm).

Same comparison as tests/test_p5b_scheduler.py, on the real weights: a batch
mixing three synthetic adapters (r = 8, 16, 64, all seven projections) and the
base model, run through the AdapterScheduler, must give each request the same
greedy tokens it gets when run alone through P5a's single-adapter engine path.

    pytest -m slow tests/test_p5b_real.py -v                        # MPS if available, else CPU
    PLAYPARSE_P5B_LAYERS=4 pytest -m slow tests/test_p5b_real.py    # low-memory: first 4 layers
    pytest -m "slow or gpu" tests/test_p5b_real.py                  # CUDA node: + FlashInfer arm
    PLAYPARSE_RECORD=1 ...                                          # write results/p5b/gate_*.json

Memory: one fp16 copy of the weights (~2.5 GB, shared by the reference and the
pooled model), plus ~0.4 GB of adapter slots.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from playparse.paths import RESULTS, WEIGHTS
from playparse.serving._serving_path import ensure_serving_importable
from playparse.serving._testing import paged_backend
from playparse.serving.adapter_pool import AdapterPool, synthetic_adapter
from playparse.serving.lora_engine import LoRAModelGPU, generate_with_adapter

ensure_serving_importable()

from engine.loader import load_config, load_weights_gpu  # noqa: E402
from engine.sampler import greedy  # noqa: E402
from engine.scheduler import generate  # noqa: E402
from serving.memory.allocator import BlockAllocator  # noqa: E402
from serving.scheduler.scheduler import SchedulerConfig  # noqa: E402

from playparse.serving.adapter_scheduler import AdapterRequest, AdapterScheduler  # noqa: E402
from playparse.serving.multi_lora import MultiLoRAModelGPU  # noqa: E402

BLOCK = 16
N_NEW = 12
# Real Llama 3 token ids (the serving layer's batch-invariance prompts), unequal lengths.
PROMPTS = {
    "a": [128000, 9906, 11, 358, 1097],
    "b": [128000, 791, 4062, 14198, 39935, 27096, 927, 279, 16053, 5679, 13],
    "c": [128000, 3923, 374, 279, 6864, 315, 9822, 30],
    "d": [128000] + [9906] * 35,
}
MIX = {"a": "r8", "b": None, "c": "r64", "d": "r16", "a2": "r64", "c2": None}
RANKS = {"r8": 8, "r16": 16, "r64": 64}


def _prompt(rid):
    return PROMPTS[rid[0]]


def _have_weights() -> bool:
    return (Path(WEIGHTS) / "config.json").is_file()


def _device() -> str:
    env = os.environ.get("PLAYPARSE_TEST_DEVICE")
    if env:
        return env
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load(device):
    cfg = load_config(WEIGHTS)
    weights = load_weights_gpu(WEIGHTS, cfg, device=device)
    layers = int(os.environ.get("PLAYPARSE_P5B_LAYERS", cfg["num_hidden_layers"]))
    if layers < cfg["num_hidden_layers"]:
        cfg = dict(cfg, num_hidden_layers=layers)
        weights = {k: v for k, v in weights.items()
                   if not k.startswith("model.layers.") or int(k.split(".")[2]) < layers}
    adapters = {n: synthetic_adapter(cfg, r, seed=r, name=n, b_std=0.01) for n, r in RANKS.items()}
    return cfg, weights, adapters


def _solo(cfg, weights, adapters, device):
    ref = LoRAModelGPU(weights, cfg, device=device)
    for n, a in adapters.items():
        ref.load_adapter(a, n)
    out = {}
    for key in {(rid[0], aid) for rid, aid in MIX.items()}:
        p, aid = key
        if aid is None:
            with ref.use_adapter(None):
                out[key] = list(generate(ref, PROMPTS[p], greedy, max_tokens=N_NEW))
        else:
            out[key] = list(generate_with_adapter(ref, PROMPTS[p], greedy, aid, max_tokens=N_NEW))
    return out


def _batched(cfg, weights, adapters, device, kernel, backend_kind="paged"):
    pool = AdapterPool(cfg, n_slots=len(adapters), max_rank=64, device=device, kernel=kernel)
    for n, a in adapters.items():
        pool.register(a, n)
    model = MultiLoRAModelGPU(weights, cfg, pool, device=device)
    num_blocks = 64
    alloc = BlockAllocator(num_blocks, BLOCK)
    if backend_kind == "flashinfer":
        from serving.backends.flashinfer_backend import FlashInferBackend
        backend = FlashInferBackend(
            num_layers=cfg["num_hidden_layers"], num_blocks=num_blocks, block_size=BLOCK,
            n_kv_heads=cfg["num_key_value_heads"], n_heads=cfg["num_attention_heads"],
            head_dim=cfg["head_dim"], device=device, dtype=torch.float16,
        )
    else:
        backend = paged_backend(cfg, num_blocks, BLOCK, device=device)
    sched = AdapterScheduler(model, backend, alloc, SchedulerConfig(max_batch_size=8))
    for rid, aid in MIX.items():
        sched.add_request(AdapterRequest(request_id=rid, prompt_ids=_prompt(rid), max_tokens=N_NEW,
                                         adapter_id=aid, ignore_eos=True))
    t0 = time.perf_counter()
    sched.run_until_idle()
    secs = time.perf_counter() - t0
    del model, pool
    return {r.request_id: list(r.output_ids) for r in sched.finished}, secs


def _compare(solo, got):
    rows, bad = [], []
    for rid, aid in MIX.items():
        want = solo[(rid[0], aid)]
        div = next((i for i, (x, y) in enumerate(zip(got[rid], want)) if x != y), None)
        rows.append({"request": rid, "adapter": aid, "equal": got[rid] == want, "first_divergence": div})
        if got[rid] != want:
            bad.append(f"{rid} ({aid}): batched {got[rid]} != solo {want} (first diff at {div})")
    return rows, bad


@pytest.fixture(scope="module")
def real_results():
    if not _have_weights():
        pytest.skip(f"no weights at {WEIGHTS} (set PLAYPARSE_WEIGHTS)")
    device = _device()
    cfg, weights, adapters = _load(device)
    solo = _solo(cfg, weights, adapters, device)
    res = {"device": device, "layers": cfg["num_hidden_layers"], "solo": solo, "batched": {}, "secs": {}}
    for kernel in ("v1", "v2"):
        res["batched"][kernel], res["secs"][kernel] = _batched(cfg, weights, adapters, device, kernel)
    # Prompt "c" runs under both the base model and r64 in MIX.
    res["base_differs"] = {"c": solo[("c", None)] != solo[("c", "r64")]}
    yield res
    if os.environ.get("PLAYPARSE_RECORD") == "1":
        out = Path(RESULTS) / "p5b"
        out.mkdir(parents=True, exist_ok=True)
        from playparse.serving._provenance import run_metadata

        summary = {
            "metadata": run_metadata(device, {"layers": res["layers"], "n_new": N_NEW, "block": BLOCK}),
            "device": device, "layers": res["layers"], "n_new": N_NEW, "mix": MIX,
            "secs": res["secs"], "base_differs_from_adapter": res["base_differs"],
            "kernels": {k: _compare(solo, v)[0] for k, v in res["batched"].items()},
        }
        (out / f"gate_real_{device}_L{res['layers']}.json").write_text(json.dumps(summary, indent=2))


@pytest.mark.slow
def test_adapters_change_real_output(real_results):
    assert any(real_results["base_differs"].values()), "synthetic adapters do not change the greedy text"


@pytest.mark.slow
@pytest.mark.parametrize("kernel", ["v1", "v2"])
def test_real_mixed_batch_equals_solo(real_results, kernel):
    _, bad = _compare(real_results["solo"], real_results["batched"][kernel])
    assert not bad, f"[{real_results['device']}] batched != solo:\n  " + "\n  ".join(bad)


@pytest.mark.slow
def test_real_v1_v2_logits_close():
    """One mixed prefill forward at real shapes: v2 logits within fp16 noise of v1."""
    if not _have_weights():
        pytest.skip("no weights")
    from serving.engine_iface.batch import ScheduledSeq, build_batch_meta, build_token_tensor
    from serving.memory.block_table import SequenceBlocks

    device = _device()
    cfg, weights, adapters = _load(device)
    cfg4 = dict(cfg, num_hidden_layers=min(2, cfg["num_hidden_layers"]))
    weights = {k: v for k, v in weights.items()
               if not k.startswith("model.layers.") or int(k.split(".")[2]) < cfg4["num_hidden_layers"]}
    adapters = {n: synthetic_adapter(cfg4, r, seed=r, name=n, b_std=0.01) for n, r in RANKS.items()}
    pool = AdapterPool(cfg4, n_slots=3, max_rank=64, device=device, kernel="v1")
    for n, a in adapters.items():
        pool.register(a, n)
    slots = [pool.acquire(a) for a in ("r8", None, "r64", "r16")]
    model = MultiLoRAModelGPU(weights, cfg4, pool, device=device)
    alloc = BlockAllocator(64, BLOCK)
    seqs = []
    for i, p in enumerate(PROMPTS.values()):
        b = SequenceBlocks(alloc, seq_id=i)
        b.append(len(p))
        seqs.append(ScheduledSeq(blocks=b, new_token_ids=list(p)))
    meta = build_batch_meta(seqs, device, BLOCK)
    tokens = build_token_tensor(seqs, device)
    out = {}
    for kernel in ("v1", "v2"):
        pool.kernel = kernel
        be = paged_backend(cfg4, 64, BLOCK, device=device)
        out[kernel] = model.forward_seq_slots(tokens, meta, be, slots).float().cpu().numpy()
    np.testing.assert_allclose(out["v2"], out["v1"], atol=0.25, rtol=0)
    assert (out["v2"].argmax(-1) == out["v1"].argmax(-1)).all()


def _cuda_reason():
    if not torch.cuda.is_available():
        return "no CUDA"
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        return "flashinfer not installed"
    return None


@pytest.mark.gpu
@pytest.mark.parametrize("backend_kind", ["paged", "flashinfer"])
def test_cuda_mixed_batch_equals_solo(backend_kind):
    reason = _cuda_reason()
    if reason and not (backend_kind == "paged" and torch.cuda.is_available()):
        if os.environ.get("REQUIRE_GPU") == "1":
            pytest.fail(f"REQUIRE_GPU=1 but {reason}")
        pytest.skip(reason)
    if not _have_weights():
        pytest.skip("no weights")
    cfg, weights, adapters = _load("cuda:0")
    solo = _solo(cfg, weights, adapters, "cuda:0")
    for kernel in ("v1", "v2"):
        got, _ = _batched(cfg, weights, adapters, "cuda:0", kernel, backend_kind)
        _, bad = _compare(solo, got)
        assert not bad, f"[cuda/{backend_kind}/{kernel}] batched != solo:\n  " + "\n  ".join(bad)
