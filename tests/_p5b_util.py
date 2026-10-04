"""Shared helpers for the P5b serving tests: a tiny multi-adapter stack and its references.

The reference for every batched output is the request run ALONE with its
adapter through P5a's single-adapter engine path (``LoRAModelGPU`` +
``generate_with_adapter``): contiguous KV cache, no batching, no pool, no paging,
per-adapter ``LoRALinear`` weights. It shares no serving-layer or P5b code with
the system under test, which is what makes agreement meaningful.
"""
from __future__ import annotations

from functools import lru_cache

from playparse.serving._serving_path import ensure_serving_importable
from playparse.serving._testing import TINY_CONFIG, paged_backend, random_engine_weights
from playparse.serving.adapter_pool import AdapterPool, synthetic_adapter
from playparse.serving.lora_engine import LoRAModelGPU, generate_with_adapter

ensure_serving_importable()

from engine.sampler import greedy  # noqa: E402
from engine.scheduler import generate  # noqa: E402
from serving.memory.allocator import BlockAllocator  # noqa: E402
from serving.scheduler.scheduler import SchedulerConfig  # noqa: E402

from playparse.serving.adapter_radix import AdapterRadixCache  # noqa: E402
from playparse.serving.adapter_scheduler import AdapterRequest, AdapterScheduler  # noqa: E402
from playparse.serving.multi_lora import MultiLoRAModelGPU  # noqa: E402

CFG = dict(TINY_CONFIG)
BLOCK = 4
MAX_RANK = 16

# Strong adapters (large B) so each one visibly changes the greedy text; a test
# whose adapters do nothing would pass with every adapter mix-up imaginable.
ADAPTER_SPECS = {
    "r4": dict(r=4, seed=101, b_std=0.25),
    "r8": dict(r=8, seed=102, b_std=0.25),
    "r16": dict(r=16, seed=103, b_std=0.2),
    "qv": dict(r=8, seed=104, b_std=0.4, target_modules=("q_proj", "v_proj")),
    "mlp": dict(r=4, seed=105, b_std=0.3, target_modules=("gate_proj", "up_proj", "down_proj")),
}

# Unequal lengths, so sequences cross block boundaries at different steps.
PROMPTS = {
    "p5": [1, 17, 42, 99, 3],
    "p11": [5, 9, 13, 200, 31, 7, 7, 64, 128, 2, 77],
    "p8": [250, 249, 248, 1, 2, 3, 4, 5],
    "p3": [11, 12, 13],
    "p17": [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2],
}


@lru_cache(maxsize=1)
def weights():
    return random_engine_weights(CFG, seed=0, std=0.1)


@lru_cache(maxsize=None)
def adapter(name: str):
    spec = dict(ADAPTER_SPECS[name])
    r, seed = spec.pop("r"), spec.pop("seed")
    return synthetic_adapter(CFG, r, seed, name=name, **spec)


@lru_cache(maxsize=1)
def reference_model() -> LoRAModelGPU:
    m = LoRAModelGPU(weights(), CFG, device="cpu")
    for n in ADAPTER_SPECS:
        m.load_adapter(adapter(n), n)
    return m


@lru_cache(maxsize=None)
def reference(prompt: tuple[int, ...], adapter_id: str | None, max_tokens: int) -> tuple[int, ...]:
    """Greedy tokens for ``prompt`` run alone under ``adapter_id`` (P5a contiguous path)."""
    m = reference_model()
    if adapter_id is None:
        with m.use_adapter(None):
            return tuple(generate(m, list(prompt), greedy, max_tokens=max_tokens))
    return tuple(generate_with_adapter(m, list(prompt), greedy, adapter_id, max_tokens=max_tokens))


def make_pool(n_slots: int, kernel: str = "v2", names=tuple(ADAPTER_SPECS)) -> AdapterPool:
    pool = AdapterPool(CFG, n_slots=n_slots, max_rank=MAX_RANK, kernel=kernel)
    for n in names:
        pool.register(adapter(n), n)
    return pool


def make_scheduler(
    *,
    n_slots: int = 8,
    kernel: str = "v2",
    num_blocks: int = 256,
    cache: bool | str = False,
    backend=None,
    device: str = "cpu",
    **cfg,
) -> AdapterScheduler:
    """A complete serving stack: pool, pooled model, paged backend, allocator, scheduler.

    ``cache``: False (no prefix cache), True (adapter-keyed), or ``"unkeyed"``
    (fault injection: token-only keys).
    """
    pool = make_pool(n_slots, kernel)
    model = MultiLoRAModelGPU(weights(), CFG, pool, device=device)
    alloc = BlockAllocator(num_blocks, BLOCK)
    be = backend if backend is not None else paged_backend(CFG, num_blocks, BLOCK, device=device)
    prefix = None
    if cache:
        prefix = AdapterRadixCache(alloc, adapter_keyed=(cache != "unkeyed"), block_copy=be.copy_block)
    return AdapterScheduler(model, be, alloc, SchedulerConfig(**cfg), prefix_cache=prefix)


def submit(sched: AdapterScheduler, rid: str, prompt, adapter_id, max_tokens: int) -> AdapterRequest:
    req = AdapterRequest(request_id=rid, prompt_ids=list(prompt), max_tokens=max_tokens,
                         adapter_id=adapter_id, ignore_eos=True)
    assert sched.add_request(req), req.error
    return req


def outputs(sched: AdapterScheduler) -> dict[str, list[int]]:
    return {r.request_id: list(r.output_ids) for r in sched.finished}


def mismatches(sched: AdapterScheduler, requests, max_tokens: int) -> list[str]:
    """Human-readable list of requests whose output differs from the solo reference."""
    got = outputs(sched)
    bad = []
    for rid, (prompt, aid) in requests.items():
        want = list(reference(tuple(prompt), aid, max_tokens))
        if got.get(rid) != want:
            bad.append(f"{rid} (adapter={aid}): batched {got.get(rid)} != solo {want}")
    return bad
