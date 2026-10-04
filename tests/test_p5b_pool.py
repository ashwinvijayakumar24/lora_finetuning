"""P5b adapter pool and kernels (fast, CPU).

Three layers, each checked against the one below it:

    lora_delta_bgmv (v2)  ==  lora_delta_loop (v1)  ==  P5a LoRALinear per adapter
    MultiLoRAModelGPU mixed batch  ==  P5a LoRAModelGPU, one request at a time

plus the pool's residency rules: LRU eviction, refcounts and pins that block
eviction, waiting instead of failing, and zero-filling a reused slot.
"""
import numpy as np
import pytest
import torch

from playparse.serving._serving_path import ensure_serving_importable
from playparse.serving._testing import TINY_CONFIG, random_engine_weights
from playparse.serving.adapter import AdapterError
from playparse.serving.adapter_pool import (
    AdapterPool,
    PoolFull,
    lora_delta_bgmv,
    lora_delta_loop,
    synthetic_adapter,
)
from playparse.serving.lora_engine import AdapterSelection, LoRALinear, LoRAModelGPU, install_lora_linear
from playparse.serving.multi_lora import (
    MultiLoRAModelGPU,
    PooledLoRALinear,
    install_multi_lora_linear,
    multi_lora_linear,
    row_slots_for,
)

ensure_serving_importable()

CFG = dict(TINY_CONFIG)
Q = "model.layers.0.self_attn.q_proj.weight"
DOWN = "model.layers.1.mlp.down_proj.weight"


def _stacks(n_slots=3, R=8, d_in=24, d_out=40, ranks=(8, 3, 5), dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.zeros(n_slots + 1, R, d_in, dtype=dtype)
    B = torch.zeros(n_slots + 1, d_out, R, dtype=dtype)
    for s, r in enumerate(ranks):
        A[s, :r] = torch.randn(r, d_in, generator=g).to(dtype)
        B[s, :, :r] = torch.randn(d_out, r, generator=g).to(dtype)
    return A, B, list(ranks)


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------

@pytest.mark.parametrize("chunk_rows", [1, 3, 64])
def test_bgmv_equals_loop_fp32_mixed_ranks_and_base_rows(chunk_rows):
    A, B, ranks = _stacks()
    x = torch.randn(11, 24, generator=torch.Generator().manual_seed(1))
    slots = torch.tensor([0, -1, 2, 2, 1, 0, -1, 1, 0, 2, -1])
    v1 = lora_delta_loop(x, A, B, ranks, slots)
    v2 = lora_delta_bgmv(x, A, B, slots, chunk_rows)
    torch.testing.assert_close(v2, v1, atol=1e-5, rtol=1e-5)
    # Base rows are exactly zero in both kernels, not merely small.
    base = slots < 0
    assert torch.count_nonzero(v1[base]) == 0 and torch.count_nonzero(v2[base]) == 0


def test_bgmv_equals_loop_fp16():
    A, B, ranks = _stacks(dtype=torch.float16)
    x = (torch.randn(9, 24, generator=torch.Generator().manual_seed(2)) * 0.5).half()
    slots = torch.tensor([2, 2, 0, -1, 1, 1, 0, 2, -1])
    v1 = lora_delta_loop(x, A, B, ranks, slots)
    v2 = lora_delta_bgmv(x, A, B, slots, 4)
    torch.testing.assert_close(v2.float(), v1.float(), atol=2e-2, rtol=1e-2)


def test_loop_matches_formula_per_row():
    A, B, ranks = _stacks()
    x = torch.randn(5, 24, generator=torch.Generator().manual_seed(3))
    slots = torch.tensor([1, 0, -1, 2, 1])
    got = lora_delta_loop(x, A, B, ranks, slots)
    for t, s in enumerate(slots.tolist()):
        want = torch.zeros(40) if s < 0 else (x[t] @ A[s].T) @ B[s].T
        torch.testing.assert_close(got[t], want, atol=1e-5, rtol=1e-5)


def test_pure_base_batch():
    A, B, ranks = _stacks()
    x = torch.randn(4, 24)
    slots = torch.full((4,), -1)
    assert lora_delta_loop(x, A, B, ranks, slots) is None
    assert torch.count_nonzero(lora_delta_bgmv(x, A, B, slots, 8)) == 0


def test_pool_v1_and_v2_match_p5a_lora_linear():
    """The pool's stacked layout computes what P5a's per-adapter LoRALinear computes."""
    pool = AdapterPool(CFG, n_slots=3, max_rank=16, kernel="v1")
    ads = {f"a{r}": synthetic_adapter(CFG, r, seed=r, b_std=0.1) for r in (4, 8, 16)}
    base = random_engine_weights(CFG)[Q]
    lin = LoRALinear(Q, base)
    for aid, ad in ads.items():
        pool.register(ad, aid)
        pool.acquire(aid)
        t = ad.layers[Q]
        lin.add(aid, t.A, t.B, t.scale)
    x = (torch.randn(7, 64, generator=torch.Generator().manual_seed(5)) * 0.5).half()
    slots = torch.tensor([pool.slot_of("a8"), -1, pool.slot_of("a4"), pool.slot_of("a16"),
                          pool.slot_of("a8"), -1, pool.slot_of("a4")])
    ref = lin.delta(x, AdapterSelection.per_row(pool.slot_names(), slots))
    v1 = pool.delta(Q, x, slots)
    pool.kernel = "v2"
    v2 = pool.delta(Q, x, slots)
    torch.testing.assert_close(v1, ref, atol=0, rtol=0)           # same algorithm, same numbers
    torch.testing.assert_close(v2.float(), ref.float(), atol=2e-3, rtol=1e-2)


def test_pool_chunking_does_not_change_v2():
    pool = AdapterPool(CFG, n_slots=2, max_rank=8, kernel="v2")
    pool.register(synthetic_adapter(CFG, 8, seed=1, b_std=0.1), "a")
    pool.register(synthetic_adapter(CFG, 4, seed=2, b_std=0.1), "b")
    sa, sb = pool.acquire("a"), pool.acquire("b")
    x = (torch.randn(13, 128) * 0.5).half()
    slots = torch.tensor([sa, sb, -1] * 4 + [sa])
    whole = pool.delta(DOWN, x, slots)
    pool.chunk_bytes = 1                     # forces one row per chunk
    torch.testing.assert_close(pool.delta(DOWN, x, slots), whole, atol=0, rtol=0)


# --------------------------------------------------------------------------
# Pool residency
# --------------------------------------------------------------------------

def _pool(n_slots=2, names=("a", "b", "c"), **kw):
    pool = AdapterPool(CFG, n_slots=n_slots, max_rank=8, **kw)
    for i, n in enumerate(names):
        pool.register(synthetic_adapter(CFG, 4, seed=i + 1), n)
    return pool


def test_register_validation():
    pool = AdapterPool(CFG, n_slots=2, max_rank=8, target_modules=("q_proj", "v_proj"))
    with pytest.raises(AdapterError, match="max_rank"):
        pool.register(synthetic_adapter(CFG, 16, seed=1, target_modules=("q_proj",)), "big")
    with pytest.raises(AdapterError, match="does not adapt"):
        pool.register(synthetic_adapter(CFG, 4, seed=1, target_modules=("k_proj",)), "k")
    pool.register(synthetic_adapter(CFG, 4, seed=1, target_modules=("q_proj",)), "q")
    with pytest.raises(AdapterError, match="already registered"):
        pool.register(synthetic_adapter(CFG, 4, seed=1, target_modules=("q_proj",)), "q")
    other = dict(CFG, hidden_size=32, intermediate_size=64, head_dim=8)
    with pytest.raises(AdapterError):
        pool.register(synthetic_adapter(other, 4, seed=1, target_modules=("q_proj",)), "wrong-shape")


def test_hit_miss_and_lru_eviction():
    pool = _pool()
    sa = pool.acquire("a"); pool.release(sa)
    sb = pool.acquire("b"); pool.release(sb)
    assert pool.stats.misses == 2 and pool.stats.evictions == 0
    pool.release(pool.acquire("a"))          # hit; "a" is now more recent than "b"
    assert pool.stats.hits == 1
    sc = pool.acquire("c")                   # evicts the LRU adapter, "b"
    assert sc == sb and not pool.is_resident("b") and pool.is_resident("a")
    assert pool.stats.evictions == 1 and pool.stats.loads == 3
    pool.check_invariants()


def test_release_does_not_refresh_lru():
    pool = _pool()
    sa = pool.acquire("a")
    pool.release(pool.acquire("b"))
    pool.release(sa)                         # released after b, but acquired before it
    pool.acquire("c")
    assert not pool.is_resident("a") and pool.is_resident("b")


def test_in_use_slot_is_never_evicted_and_requests_wait():
    pool = _pool()
    sa, sb = pool.acquire("a"), pool.acquire("b")
    assert not pool.can_acquire("c")         # both slots held: c must wait, not fail
    with pytest.raises(PoolFull):
        pool.acquire("c")
    assert pool.can_acquire("a")             # already resident: always fine
    pool.release(sb)
    assert pool.can_acquire("c")
    assert pool.acquire("c") == sb and pool.slot_adapter[sa] == "a"


def test_can_acquire_is_side_effect_free():
    pool = _pool()
    before = (pool.stats.as_dict(), list(pool.slot_adapter), list(pool.refcount))
    for aid in ("a", "b", "c", None):
        pool.can_acquire(aid)
    assert (pool.stats.as_dict(), list(pool.slot_adapter), list(pool.refcount)) == before


def test_pinned_adapter_is_never_evicted():
    pool = _pool()
    pool.pin("a")
    for n in ("b", "c", "b", "c"):
        pool.release(pool.acquire(n))
    assert pool.is_resident("a") and pool.stats.evictions >= 3
    pool.release(pool.acquire("a"))
    assert pool.stats.hits >= 1
    pool.unpin("a")
    pool.release(pool.acquire("b")); pool.release(pool.acquire("c"))
    assert not pool.is_resident("a")


def test_unknown_adapter_and_base():
    pool = _pool()
    assert pool.acquire(None) == -1 and pool.can_acquire(None)
    assert not pool.can_acquire("nope") and not pool.is_registered("nope")
    with pytest.raises(KeyError):
        pool.acquire("nope")


def test_reused_slot_is_zero_filled():
    """A slot's previous tenant must not leak through projections the new one does not target."""
    pool = AdapterPool(CFG, n_slots=1, max_rank=8)
    pool.register(synthetic_adapter(CFG, 8, seed=1, b_std=0.1), "full")
    pool.register(synthetic_adapter(CFG, 4, seed=2, target_modules=("q_proj",)), "q_only")
    pool.release(pool.acquire("full"))
    s = pool.acquire("q_only")
    assert torch.count_nonzero(pool.A[DOWN][s]) == 0 and torch.count_nonzero(pool.B[DOWN][s]) == 0
    assert torch.count_nonzero(pool.A[Q][s, 4:]) == 0      # rank padding beyond r=4 is zero too
    pool.check_invariants()


def test_unregister_refuses_in_use_and_frees_slot():
    pool = _pool()
    s = pool.acquire("a")
    with pytest.raises(RuntimeError, match="in use"):
        pool.unregister("a")
    pool.release(s)
    pool.unregister("a")
    assert not pool.is_registered("a") and pool.slot_adapter[s] is None
    pool.check_invariants()


def test_release_underflow_is_loud():
    pool = _pool()
    s = pool.acquire("a")
    pool.release(s)
    with pytest.raises(RuntimeError):
        pool.release(s)


def test_memory_accounting():
    pool = AdapterPool(CFG, n_slots=4, max_rank=16)
    per_slot_params = sum(16 * (i + o) for (o, i) in pool.shapes.values())
    assert pool.bytes_per_slot() == per_slot_params * 2
    assert pool.device_bytes() == 5 * per_slot_params * 2          # 4 slots + the zero slot
    pool.register(synthetic_adapter(CFG, 8, seed=1), "a")
    assert pool.host_bytes("a") == per_slot_params // 2 * 2         # r=8 is half of max_rank=16


# --------------------------------------------------------------------------
# Model level
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stack():
    weights = random_engine_weights(CFG, seed=0, std=0.1)
    ads = {"r4": synthetic_adapter(CFG, 4, seed=11, b_std=0.2),
           "r8": synthetic_adapter(CFG, 8, seed=12, b_std=0.2),
           "q": synthetic_adapter(CFG, 8, seed=13, b_std=0.3, target_modules=("q_proj", "v_proj"))}
    ref = LoRAModelGPU(weights, CFG, device="cpu")
    for n, a in ads.items():
        ref.load_adapter(a, n)
    return weights, ads, ref


def _meta_for(prompts):
    """BatchMeta for one prefill step over several fresh sequences."""
    from serving.engine_iface.batch import ScheduledSeq, build_batch_meta, build_token_tensor
    from serving.memory.allocator import BlockAllocator
    from serving.memory.block_table import SequenceBlocks

    alloc = BlockAllocator(64, 4)
    seqs = []
    for i, p in enumerate(prompts):
        b = SequenceBlocks(alloc, seq_id=i)
        b.append(len(p))
        seqs.append(ScheduledSeq(blocks=b, new_token_ids=list(p)))
    return build_token_tensor(seqs, "cpu"), build_batch_meta(seqs, "cpu", 4)


@pytest.mark.parametrize("kernel", ["v1", "v2"])
def test_mixed_batch_forward_matches_p5a_per_request(stack, kernel):
    from playparse.serving._testing import paged_backend

    weights, ads, ref = stack
    pool = AdapterPool(CFG, n_slots=3, max_rank=8, kernel=kernel)
    for n, a in ads.items():
        pool.register(a, n)
    model = MultiLoRAModelGPU(weights, CFG, pool, device="cpu")
    prompts = [[1, 5, 9, 2, 7], [3, 3, 8], [10, 20, 30, 40, 50, 60, 70], [4, 2], [9, 9, 9, 1]]
    adapters = ["r8", None, "q", "r4", "r8"]
    slots = [pool.acquire(a) for a in adapters]
    tokens, meta = _meta_for(prompts)
    logits = model.forward_varlen(tokens, meta, paged_backend(CFG, 64, 4), adapter=model.selection_for(slots, meta))
    for i, (p, a) in enumerate(zip(prompts, adapters)):
        want = ref.forward_all(p, adapter=a)[-1]
        got = logits[i].float().numpy()
        np.testing.assert_allclose(got, want, atol=2e-2, rtol=0)
        assert int(np.argmax(got)) == int(np.argmax(want))


def test_forward_varlen_requires_explicit_selection(stack):
    weights, ads, _ = stack
    pool = AdapterPool(CFG, n_slots=1, max_rank=8)
    model = MultiLoRAModelGPU(weights, CFG, pool, device="cpu")
    tokens, meta = _meta_for([[1, 2, 3]])
    with pytest.raises(TypeError):
        model.forward_varlen(tokens, meta, None)                # no adapter= keyword
    with pytest.raises(TypeError, match="per_row"):
        model.forward_varlen(tokens, meta, None, adapter=AdapterSelection.single("x"))


def test_misaligned_row_slots_raise(stack):
    weights, _, _ = stack
    pool = AdapterPool(CFG, n_slots=1, max_rank=8)
    w = PooledLoRALinear(Q, weights[Q], pool)
    sel = AdapterSelection.per_row(pool.slot_names(), torch.tensor([-1, -1]))
    with pytest.raises(ValueError, match="row_slots"):
        w.delta(torch.zeros(3, 64, dtype=torch.float16), sel)


def test_row_slots_for_uses_batch_indices():
    _, meta = _meta_for([[1, 2, 3], [4], [5, 6]])
    assert row_slots_for([7, -1, 2], meta, "cpu").tolist() == [7, 7, 7, -1, 2, 2]
    with pytest.raises(ValueError):
        row_slots_for([1, 2], meta, "cpu")


def test_install_composes_with_p5a_and_preserves_plain_weights(stack):
    from engine import components_gpu

    weights, _, _ = stack
    x = torch.randn(3, 64).half()
    w = weights[Q]
    # Alternating re-installs used to build a cycle (p5b-linear-wrapper-cycle):
    # each wrapper ended up delegating to the other, RecursionError on first use.
    for install in (install_lora_linear, install_multi_lora_linear, install_lora_linear,
                    install_multi_lora_linear, install_multi_lora_linear):
        install()
        assert torch.equal(components_gpu.linear(x, w), x @ w.T)
    assert torch.equal(multi_lora_linear(x, w), x @ w.T)
    with pytest.raises(RuntimeError, match="install_multi_lora_linear"):
        _ = PooledLoRALinear(Q, w, AdapterPool(CFG, n_slots=1, max_rank=4)).T
