"""P5b correctness gate: mixed-adapter continuous batching == each request alone (fast, CPU).

Same discipline as the serving layer's batch-invariance gate
(llm_serving_layer/tests/test_batch_invariance.py): exact greedy-token equality,
not a tolerance. Each request's batched output must equal the tokens it gets
when run ALONE with its adapter through P5a's single-adapter engine path.

The batches mix adapters of different ranks, adapters that target different
projections, and base-model requests; they run with chunked prefill, staggered
arrivals, an adapter pool smaller than the number of adapters (so requests wait
for slots and adapters are evicted and reloaded), and forced KV-memory pressure
under both preemption policies. Every run checks a POSITIVE property that proves
the hard path was actually exercised (waits > 0, evictions > 0, preemptions > 0),
so a rig that silently stopped forcing the condition cannot pass.
"""
import sys

import pytest

from tests._p5b_util import (
    ADAPTER_SPECS,
    CFG,
    PROMPTS,
    make_scheduler,
    mismatches,
    outputs,
    reference,
    submit,
    weights,
)

from playparse.serving._serving_path import serving_dir
from playparse.serving.adapter_scheduler import AdapterRequest, AdapterRoutingError, AdapterScheduler

from serving.cache.radix import RadixCache
from serving.scheduler.preemption import PreemptionPolicy
from serving.scheduler.scheduler import Request, RequestState, Scheduler

N = 8
MIX = {   # request id -> (prompt, adapter)
    "a": (PROMPTS["p5"], "r8"),
    "b": (PROMPTS["p11"], None),
    "c": (PROMPTS["p8"], "qv"),
    "d": (PROMPTS["p17"], "r16"),
    "e": (PROMPTS["p3"], "r4"),
    "f": (PROMPTS["p11"], "mlp"),
    "g": (PROMPTS["p5"], "r4"),
}


def _run(sched, requests=MIX, n=N):
    for rid, (prompt, aid) in requests.items():
        submit(sched, rid, prompt, aid, n)
    sched.run_until_idle()
    return mismatches(sched, requests, n)


def _assert_clean(sched):
    sched.pool.check_invariants()
    sched.allocator.check_invariants()
    assert all(c == 0 for c in sched.pool.refcount), f"leaked adapter refs {sched.pool.refcount}"
    if sched.prefix_cache is None:
        assert sched.allocator.num_free == sched.allocator.num_blocks, "leaked KV blocks"


# --------------------------------------------------------------------------
# Guards on the test itself
# --------------------------------------------------------------------------

def test_adapters_change_the_output():
    """Without this, every routing bug would pass: all adapters must give distinct text."""
    prompt = tuple(PROMPTS["p11"])
    outs = {aid: reference(prompt, aid, N) for aid in [None, *ADAPTER_SPECS]}
    assert len(set(outs.values())) == len(outs), outs


def test_serving_import_does_not_expose_other_packages():
    root = str(serving_dir())
    assert root not in sys.path
    import bench
    import serving
    assert serving.__file__.startswith(root) and bench.__file__.startswith(root)


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kernel", ["v1", "v2"])
def test_mixed_adapter_batch_equals_solo(kernel):
    sched = make_scheduler(kernel=kernel, n_slots=8)
    bad = _run(sched)
    assert not bad, "batched != solo:\n  " + "\n  ".join(bad)
    _assert_clean(sched)


def test_batches_really_mix_adapters():
    """The gate is only meaningful if one forward pass carried several adapters at once."""
    sched = make_scheduler(n_slots=8)
    for rid, (prompt, aid) in MIX.items():
        submit(sched, rid, prompt, aid, N)
    widest = 0
    while sched.has_work:
        sched.step()
        widest = max(widest, len(set(sched.last_seq_slots)))
    assert widest >= 5, f"at most {widest} distinct adapter slots shared a forward pass"


@pytest.mark.parametrize("kernel", ["v1", "v2"])
def test_small_pool_waits_evicts_and_stays_correct(kernel):
    """5 adapters through 2 slots: requests wait for a slot (never rejected), adapters cycle."""
    sched = make_scheduler(kernel=kernel, n_slots=2)
    bad = _run(sched)
    assert not bad, "batched != solo:\n  " + "\n  ".join(bad)
    st = sched.pool.stats
    assert st.admission_waits > 0 and st.evictions > 0 and st.misses > 2
    assert all(r.state == RequestState.FINISHED for r in sched.finished)
    _assert_clean(sched)


def test_chunked_prefill_and_staggered_arrivals():
    long_prompt = list(range(1, 31))                 # 30 tokens, forced across chunks
    reqs = {"long": (long_prompt, "r16"), **{k: MIX[k] for k in ("a", "b", "c", "e")}}
    sched = make_scheduler(n_slots=3, max_prefill_tokens=8)
    submit(sched, "long", long_prompt, "r16", N)
    for _ in range(2):
        sched.step()
    for i, k in enumerate(("a", "b", "c", "e")):
        submit(sched, k, *MIX[k], N)
        sched.step()
    sched.run_until_idle()
    bad = mismatches(sched, reqs, N)
    assert not bad, "\n  ".join(bad)
    _assert_clean(sched)


@pytest.mark.parametrize("policy", [PreemptionPolicy.RECOMPUTE, PreemptionPolicy.SWAP])
def test_preemption_under_memory_pressure_keeps_adapters(policy):
    """KV pool too small for the batch: sequences are preempted and resumed with the right adapter."""
    n = 16
    reqs = {k: MIX[k] for k in ("a", "c", "d", "f")}
    sched = make_scheduler(n_slots=4, num_blocks=14, preemption_policy=policy)
    bad = _run(sched, reqs, n)
    assert sched.preemption.total > 0, "the rig never forced a preemption"
    assert not bad, "\n  ".join(bad)
    _assert_clean(sched)


def test_preemption_with_small_pool_releases_and_reacquires():
    """Recompute-preempted requests give their slot back and re-acquire it on readmission."""
    n = 16
    reqs = {k: MIX[k] for k in ("a", "c", "d", "f", "g")}
    sched = make_scheduler(n_slots=2, num_blocks=14, preemption_policy=PreemptionPolicy.RECOMPUTE)
    bad = _run(sched, reqs, n)
    assert sched.preemption.total > 0 and sched.pool.stats.admission_waits > 0
    assert not bad, "\n  ".join(bad)
    _assert_clean(sched)


def test_base_only_and_plain_requests():
    """Plain upstream Requests are base-model requests; the forward pass gets adapter=None."""
    sched = make_scheduler(n_slots=1)
    sched.add_request(Request(request_id="plain", prompt_ids=PROMPTS["p8"], max_tokens=N, ignore_eos=True))
    submit(sched, "none", PROMPTS["p5"], None, N)
    sched.run_until_idle()
    got = outputs(sched)
    assert got["plain"] == list(reference(tuple(PROMPTS["p8"]), None, N))
    assert got["none"] == list(reference(tuple(PROMPTS["p5"]), None, N))
    assert sched.pool.stats.loads == 0


# --------------------------------------------------------------------------
# Pool interaction
# --------------------------------------------------------------------------

def test_unknown_adapter_is_rejected_not_queued():
    sched = make_scheduler()
    req = AdapterRequest(request_id="x", prompt_ids=[1, 2], adapter_id="nope")
    assert sched.add_request(req) is False
    assert req.state == RequestState.FAILED and "unknown adapter" in req.error
    assert not sched.waiting


def test_waiting_request_is_admitted_when_a_slot_frees():
    sched = make_scheduler(n_slots=1)
    first = submit(sched, "first", PROMPTS["p5"], "r8", 6)
    second = submit(sched, "second", PROMPTS["p5"], "qv", 6)
    sched.step()
    assert first.state != RequestState.WAITING and second.state == RequestState.WAITING
    sched.run_until_idle()
    assert second.state == RequestState.FINISHED
    assert outputs(sched)["second"] == list(reference(tuple(PROMPTS["p5"]), "qv", 6))


def test_pinned_adapter_survives_a_churning_workload():
    sched = make_scheduler(n_slots=2)
    sched.pool.pin("r4")
    reqs = {f"q{i}": (PROMPTS["p8"], aid) for i, aid in enumerate(["r8", "qv", "mlp", "r16", "r8", "r4"])}
    bad = _run(sched, reqs, 5)
    assert not bad
    assert sched.pool.is_resident("r4") and sched.pool.stats.evictions >= 3


def test_cancellation_releases_the_slot():
    sched = make_scheduler(n_slots=1)
    submit(sched, "long", PROMPTS["p5"], "r8", 40)
    waiting = submit(sched, "next", PROMPTS["p3"], "qv", 4)
    for _ in range(3):
        sched.step()
    assert waiting.state == RequestState.WAITING
    assert sched.cancel("long")
    sched.run_until_idle()
    assert waiting.state == RequestState.FINISHED
    _assert_clean(sched)


# --------------------------------------------------------------------------
# Loud failures instead of silent misrouting
# --------------------------------------------------------------------------

def test_refuses_plain_radix_cache_and_plain_model():
    sched = make_scheduler()
    with pytest.raises(TypeError, match="AdapterRadixCache"):
        AdapterScheduler(sched.lora_model, sched.backend, sched.allocator,
                         prefix_cache=RadixCache(sched.allocator))
    from engine.model_gpu import LlamaModelGPU
    with pytest.raises(TypeError, match="MultiLoRAModelGPU"):
        AdapterScheduler(LlamaModelGPU(weights(), CFG, device="cpu"), sched.backend, sched.allocator)


def test_missing_record_is_detected():
    """If upstream stopped calling _tokens_for once per sequence, the forward pass must refuse."""

    class Broken(AdapterScheduler):
        def _tokens_for(self, req, q):
            return Scheduler._tokens_for(self, req, q)      # skips the adapter record

    good = make_scheduler()
    broken = Broken(good.lora_model, good.backend, good.allocator, good.config)
    submit(broken, "a", PROMPTS["p5"], "r8", 3)
    with pytest.raises(AdapterRoutingError, match="recorded 0"):
        broken.step()


def test_slot_holding_another_adapter_is_detected():
    sched = make_scheduler(n_slots=2)
    req = submit(sched, "a", PROMPTS["p5"], "r8", 4)
    submit(sched, "b", PROMPTS["p5"], "qv", 4)
    sched.step()
    req.adapter_slot = sched.pool.slot_of("qv")           # simulate a refcount bug
    with pytest.raises(AdapterRoutingError, match="wants adapter 'r8'"):
        sched.step()
