"""The shared P5b workload and scoring (fast): what both our benchmark and the vLLM one use."""
import pytest

from playparse.serving.p5b_workload import (
    SLO,
    Outcome,
    WorkloadSpec,
    assign_adapters,
    build_requests,
    summarize,
    tenant_ids,
)


def test_adapter_assignment_is_seeded_and_skewed():
    a = assign_adapters(2000, 256, "zipf", seed=3)
    assert a == assign_adapters(2000, 256, "zipf", seed=3)
    assert a != assign_adapters(2000, 256, "zipf", seed=4)
    top = max(a.count(t) for t in set(a)) / len(a)
    u = assign_adapters(2000, 256, "uniform", seed=3)
    top_u = max(u.count(t) for t in set(u)) / len(u)
    assert top > 0.1 > top_u              # Zipf s=1.1 over 256: ~16% to the top tenant
    assert set(a) <= set(tenant_ids(256))
    with pytest.raises(ValueError):
        assign_adapters(10, 4, "pareto")


def test_build_requests_controls_lengths_and_arrivals():
    spec = WorkloadSpec(n_requests=50, n_adapters=8, popularity="uniform", rate_rps=10.0,
                        prompt_mean=40, prompt_max=80, output_mean=12, output_max=30, vocab_size=256)
    reqs, realized = build_requests(spec)
    assert len(reqs) == 50
    assert all(4 <= len(r.prompt_ids) <= 80 and 2 <= r.max_tokens <= 30 for r in reqs)
    assert all(0 <= t < 256 for r in reqs for t in r.prompt_ids)
    arr = [r.arrival_s for r in reqs]
    assert arr == sorted(arr) and arr[0] == 0.0
    assert 2 < arr[-1] < 10                # ~49 gaps of mean 0.1 s
    assert realized["distinct_adapters_used"] == 8
    again, _ = build_requests(spec)
    assert [r.prompt_ids for r in again] == [r.prompt_ids for r in reqs]


def test_base_only_workload():
    reqs, realized = build_requests(WorkloadSpec(n_requests=5, n_adapters=0, vocab_size=256))
    assert all(r.adapter_id is None for r in reqs)


def test_slo_refuses_impossible_anchor():
    with pytest.raises(ValueError, match="refusing"):
        SLO.from_unloaded(-4056.7, 27.1)       # the negative-TTFT calibration bug
    slo = SLO.from_unloaded(20.0, 5.0)
    assert (slo.ttft_ms, slo.tpot_ms) == (200.0, 15.0)


def test_ttft_is_measured_from_intended_arrival():
    o = Outcome("r", None, arrival_s=1.0, first_token_s=1.25, finish_s=2.25, n_tokens=11)
    assert o.ttft_ms == pytest.approx(250.0) and o.tpot_ms == pytest.approx(100.0)


def test_goodput_counts_only_requests_meeting_both_bounds():
    slo = SLO(ttft_ms=100.0, tpot_ms=10.0)
    outs = [
        Outcome("ok", "a", 0.0, 0.05, 0.15, 11),         # ttft 50, tpot 10 -> good
        Outcome("slow_ttft", "a", 0.0, 0.2, 0.3, 11),    # ttft 200 -> bad
        Outcome("slow_tpot", "a", 0.0, 0.05, 0.5, 11),   # tpot 45 -> bad
        Outcome("unfinished", "a", 0.0, 0.05, None, 3),
    ]
    s = summarize(outs, slo, window_s=2.0, wall_s=4.0)
    assert s["n_within_slo"] == 1 and s["goodput_rps"] == 0.5
    assert s["n_completed"] == 3 and s["slo_attainment"] == 0.25
    assert s["throughput_tok_s"] == pytest.approx(33 / 4.0)


def test_exported_peft_adapters_are_the_bench_adapters(tmp_path):
    """The vLLM driver serves the same tensors as ours: same seeds, PEFT on disk."""
    import numpy as np

    from playparse.serving._testing import TINY_CONFIG
    from playparse.serving.adapter import load_peft_adapter
    from playparse.serving.adapter_pool import synthetic_adapter
    from playparse.serving.p5b_workload import ADAPTER_SEED0, export_peft_adapters

    paths = export_peft_adapters(TINY_CONFIG, 2, 8, tmp_path)
    assert [p.name for p in paths] == tenant_ids(2)
    back = load_peft_adapter(paths[1], base_config=TINY_CONFIG)
    want = synthetic_adapter(TINY_CONFIG, 8, seed=ADAPTER_SEED0 + 1, b_std=0.01)
    assert back.r == 8 and back.scale == want.scale
    for k, t in want.layers.items():
        np.testing.assert_array_equal(back.layers[k].A, t.A)
        np.testing.assert_array_equal(back.layers[k].B, t.B)
