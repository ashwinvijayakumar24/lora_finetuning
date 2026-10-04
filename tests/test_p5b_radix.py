"""Claim L6: the radix prefix cache is adapter-safe (fast, CPU).

A KV block computed under adapter A must never be reused for adapter B, or for
the base model, because the hidden states (and so K and V) differ. The tests:

* cache-level: same tokens under two adapters never cross-hit; same adapter hits;
  base vs adapter never cross-hit; calls outside a scope raise.
* end to end through the scheduler: outputs with the cache on equal the solo
  references, the audit counter of cross-adapter hits stays at zero, and real
  hits still happen within one adapter (so the cache is not simply disabled).
* FAULT INJECTION: the same workload with token-only keys (the bug) is caught by
  the very same output comparison, and the audit counter sees it. A test that
  could not fail on the bug would not be evidence for the claim.
"""
import pytest

from _p5b_util import BLOCK, PROMPTS, make_scheduler, mismatches, outputs, reference, submit

from playparse.serving.adapter_radix import NAMESPACE_STRIDE, AdapterRadixCache

from serving.memory.allocator import BlockAllocator
from serving.memory.block_table import SequenceBlocks

PROMPT = list(range(1, 18))       # 17 tokens = 4 full blocks + 1 at block_size 4


def _seq_with_blocks(alloc, n_tokens):
    s = SequenceBlocks(alloc, seq_id=0)
    s.append(n_tokens)
    return s


@pytest.fixture
def cache():
    alloc = BlockAllocator(64, BLOCK)
    return AdapterRadixCache(alloc)


def _publish(cache, adapter_id, tokens):
    seq = _seq_with_blocks(cache.allocator, len(tokens))
    with cache.scope(adapter_id):
        cache.insert(tokens, seq.block_ids)
    seq.free()                         # the cache keeps its own references


# --------------------------------------------------------------------------
# Cache level
# --------------------------------------------------------------------------

def test_same_prompt_two_adapters_no_cross_hit(cache):
    _publish(cache, "A", PROMPT)
    with cache.scope("B"):
        assert cache.match(PROMPT).n_tokens == 0
    with cache.scope("A"):
        assert cache.match(PROMPT).n_tokens == 16


def test_base_and_adapter_never_share(cache):
    _publish(cache, None, PROMPT)
    with cache.scope("A"):
        assert cache.match(PROMPT).n_tokens == 0
    _publish(cache, "A", PROMPT)
    with cache.scope(None):
        res = cache.acquire(PROMPT)
    assert res.n_tokens == 16 and cache.cross_adapter_hits == 0
    cache.allocator.free(res.block_ids)


def test_base_namespace_uses_raw_token_keys(cache):
    """Base-only traffic keys exactly like the upstream cache (namespace code 0)."""
    _publish(cache, None, PROMPT)
    keys = {n.key for n in cache._nodes}
    assert tuple(PROMPT[:BLOCK]) in keys
    _publish(cache, "A", PROMPT)
    assert any(min(n.key) >= NAMESPACE_STRIDE for n in cache._nodes)


def test_unscoped_calls_raise(cache):
    with pytest.raises(RuntimeError, match="outside scope"):
        cache.match(PROMPT)
    with pytest.raises(RuntimeError, match="outside scope"):
        cache.acquire(PROMPT)
    with pytest.raises(RuntimeError, match="outside scope"):
        cache.insert(PROMPT, [0, 1, 2, 3])


def test_fault_injection_token_only_keys_cross_hit_and_are_counted():
    alloc = BlockAllocator(64, BLOCK)
    broken = AdapterRadixCache(alloc, adapter_keyed=False)
    _publish(broken, "A", PROMPT)
    with broken.scope("B"):
        res = broken.acquire(PROMPT)
    assert res.n_tokens == 16                    # the bug: B is handed A's KV
    assert broken.cross_adapter_hits == 1 and broken.cross_adapter_blocks == 4
    alloc.free(res.block_ids)


def test_namespaces_share_one_lru_and_stay_consistent(cache):
    _publish(cache, "A", PROMPT)
    _publish(cache, "B", PROMPT)
    _publish(cache, None, PROMPT)
    cache.check_invariants()
    assert cache.cached_blocks == 12
    cache.evict(4)                               # oldest first: A's chain, leaf to root
    with cache.scope("A"):
        assert cache.match(PROMPT).n_tokens == 0
    with cache.scope("B"):
        assert cache.match(PROMPT).n_tokens == 16
    cache.check_invariants()
    assert cache.clear() == 8 and cache.allocator.num_free == 64


# --------------------------------------------------------------------------
# End to end through the scheduler
# --------------------------------------------------------------------------

N = 6


def _sequential(sched, plan):
    """Run requests one after another so each can hit what the previous ones published."""
    hits = {}
    for rid, (prompt, aid) in plan.items():
        submit(sched, rid, prompt, aid, N)
        before = sched.prefix_cache.stats.blocks_reused
        sched.run_until_idle()
        hits[rid] = sched.prefix_cache.stats.blocks_reused - before
    return hits


PLAN = {
    "A1": (PROMPT, "r8"),
    "B1": (PROMPT, "qv"),
    "base1": (PROMPT, None),
    "A2": (PROMPT, "r8"),
    "B2": (PROMPT, "qv"),
    "base2": (PROMPT, None),
}


def test_adapter_keyed_cache_hits_within_adapter_only_and_stays_correct():
    sched = make_scheduler(n_slots=4, cache=True)
    hits = _sequential(sched, PLAN)
    assert hits["A1"] == hits["B1"] == hits["base1"] == 0     # first under each namespace
    assert hits["A2"] == hits["B2"] == hits["base2"] == 4     # same adapter: full reuse
    assert sched.prefix_cache.cross_adapter_hits == 0
    bad = mismatches(sched, PLAN, N)
    assert not bad, "\n  ".join(bad)


def test_fault_injection_is_caught_by_output_comparison():
    """With token-only keys the second adapter reuses the first's KV and its text changes."""
    sched = make_scheduler(n_slots=4, cache="unkeyed")
    _sequential(sched, PLAN)
    assert sched.prefix_cache.cross_adapter_hits > 0
    bad = mismatches(sched, PLAN, N)
    assert any(rid.startswith(("B", "base")) for rid in (b.split()[0] for b in bad)), (
        "a cross-adapter KV hit went undetected by the output comparison"
    )


def test_concurrent_mixed_traffic_cache_on_equals_cache_off():
    """Shared prompts under several adapters, interleaved: cache on == cache off == solo."""
    plan = {}
    for i in range(12):
        prompt = PROMPT if i % 2 == 0 else PROMPTS["p11"] + PROMPT[:6]
        plan[f"r{i}"] = (prompt, ["r8", "qv", None, "r16", "mlp", "r8"][i % 6])
    results = {}
    for cache in (False, True):
        sched = make_scheduler(n_slots=3, cache=cache, max_batch_size=4)
        for j, (rid, (prompt, aid)) in enumerate(plan.items()):
            submit(sched, rid, prompt, aid, N)
            if j % 3 == 2:
                sched.step()
        sched.run_until_idle()
        results[cache] = outputs(sched)
        assert not mismatches(sched, plan, N)
        if cache:
            assert sched.prefix_cache.cross_adapter_hits == 0
            assert sched.prefix_cache.stats.blocks_reused > 0, "the cache never hit; test is vacuous"
    assert results[False] == results[True]
