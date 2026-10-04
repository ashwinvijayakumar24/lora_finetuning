"""The P5b multi-tenant workload and its scoring, shared by our benchmark and the vLLM one.

Claim L7 compares this serving stack with vLLM's multi-LoRA mode "on the same
workload". The only way to make that literally true is for both drivers to
build the workload and score the results with the same code, so it lives here:

* prompts, output lengths and arrival times come from the serving layer's own
  workload generator (``bench.workloads.generator``): seeded, token-id based,
  with a realized-distribution audit;
* each request is assigned a tenant adapter, with **uniform** or **Zipf**
  popularity (PRD §10, L4). Zipf with s = 1.1 over 256 tenants sends ~17% of
  traffic to the top tenant and leaves a long tail of rarely used adapters,
  which is what makes an LRU adapter pool hit or miss;
* the SLO is anchored to the measured unloaded latency with the serving layer's
  own multipliers (``bench/run_p2.py``: TTFT x10, per-token x3), fixed in source
  before any loaded run;
* goodput = requests that finished within BOTH SLO bounds, per second of the
  arrival window.
"""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from playparse.serving._serving_path import ensure_serving_importable

ensure_serving_importable()

from bench.workloads.generator import LengthSpec, WorkloadConfig, arrival_times, generate  # noqa: E402

# Same multipliers as the serving layer's Phase 2 driver (bench/run_p2.py:75-76),
# restated here (not imported) so this module does not pull in the HTTP client.
SLO_TTFT_MULTIPLIER = 10.0
SLO_TPOT_MULTIPLIER = 3.0
POPULARITIES = ("uniform", "zipf")


def tenant_ids(n: int) -> list[str]:
    return [f"tenant-{i:03d}" for i in range(n)]


def popularity_weights(n: int, kind: str, s: float = 1.1) -> list[float]:
    if kind == "uniform":
        return [1.0] * n
    if kind == "zipf":
        return [1.0 / (i + 1) ** s for i in range(n)]
    raise ValueError(f"unknown popularity {kind!r}; expected {POPULARITIES}")


def assign_adapters(n_requests: int, n_adapters: int, popularity: str, seed: int = 0,
                    zipf_s: float = 1.1) -> list[str]:
    """Tenant adapter per request. Seeded, independent of the prompt/arrival streams."""
    ids = tenant_ids(n_adapters)
    rng = random.Random(f"p5b-adapters:{seed}")
    return rng.choices(ids, weights=popularity_weights(n_adapters, popularity, zipf_s), k=n_requests)


@dataclass
class WorkloadSpec:
    n_requests: int = 128
    n_adapters: int = 16
    popularity: str = "uniform"
    rate_rps: float = 4.0
    prompt_mean: int = 64
    prompt_max: int = 256
    output_mean: int = 32
    output_max: int = 128
    vocab_size: int = 32_000
    block_size: int = 16
    seed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TimedRequest:
    request_id: str
    prompt_ids: list[int]
    max_tokens: int
    adapter_id: str | None
    arrival_s: float


def build_requests(spec: WorkloadSpec) -> tuple[list[TimedRequest], dict[str, Any]]:
    """Requests with prompts, controlled output lengths, adapters and Poisson arrival times."""
    wl = generate(WorkloadConfig(
        n_requests=spec.n_requests, structure="zero", block_size=spec.block_size, seed=spec.seed,
        prompt=LengthSpec(dist="lognormal", mean=spec.prompt_mean, sigma=0.5, min_len=4, max_len=spec.prompt_max),
        output=LengthSpec(dist="lognormal", mean=spec.output_mean, sigma=0.5, min_len=2, max_len=spec.output_max),
        vocab_size=spec.vocab_size, marker_space=min(1024, spec.vocab_size // 4),
        name="p5b-multi-tenant",
    ))
    arrivals = arrival_times(spec.n_requests, spec.rate_rps, "poisson", seed=spec.seed)
    adapters = assign_adapters(spec.n_requests, spec.n_adapters, spec.popularity, spec.seed) \
        if spec.n_adapters > 0 else [None] * spec.n_requests
    reqs = [
        TimedRequest(f"req-{i:05d}", list(r.token_ids), int(r.max_tokens), adapters[i], arrivals[i])
        for i, r in enumerate(wl.requests)
    ]
    realized = {
        "prompt_len_mean": statistics.fmean(len(r.prompt_ids) for r in reqs),
        "output_len_mean": statistics.fmean(r.max_tokens for r in reqs),
        "distinct_adapters_used": len({r.adapter_id for r in reqs}),
        "top_adapter_share": max(adapters.count(a) for a in set(adapters)) / len(adapters) if reqs else 0.0,
        "workload_fingerprint": wl.fingerprint(),
        "degeneracy_warnings": wl.degeneracy_warnings,
    }
    return reqs, realized


ADAPTER_SEED0 = 1000      # tenant i is synthetic_adapter(seed=ADAPTER_SEED0 + i) in BOTH drivers


def export_peft_adapters(config, n: int, rank: int, out_dir, *, b_std: float = 0.01) -> list:
    """Write tenant adapters 0..n-1 as PEFT directories (for vLLM's LoRARequest).

    Same seeds as ``scripts/p5b_bench.py::make_pool``, so the vLLM run serves
    literally the same adapter tensors as ours. Returns the directory paths.
    """
    from pathlib import Path

    from playparse.serving.adapter import save_peft_adapter
    from playparse.serving.adapter_pool import synthetic_adapter

    out = Path(out_dir)
    paths = []
    for i, aid in enumerate(tenant_ids(n)):
        p = out / aid
        if not (p / "adapter_model.safetensors").is_file():
            save_peft_adapter(synthetic_adapter(config, rank, seed=ADAPTER_SEED0 + i, b_std=b_std, name=aid), p)
        paths.append(p)
    return paths


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

@dataclass
class SLO:
    ttft_ms: float
    tpot_ms: float
    anchored_on: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_unloaded(cls, ttft_ms: float, tpot_ms: float) -> "SLO":
        if not (ttft_ms > 0 and tpot_ms > 0):
            raise ValueError(f"refusing to anchor an SLO on ttft={ttft_ms} tpot={tpot_ms}")
        return cls(ttft_ms * SLO_TTFT_MULTIPLIER, tpot_ms * SLO_TPOT_MULTIPLIER,
                   {"unloaded_ttft_ms": ttft_ms, "unloaded_tpot_ms": tpot_ms,
                    "ttft_multiplier": SLO_TTFT_MULTIPLIER, "tpot_multiplier": SLO_TPOT_MULTIPLIER})


@dataclass
class Outcome:
    """One request's timeline, in seconds from the start of the run."""

    request_id: str
    adapter_id: str | None
    arrival_s: float
    first_token_s: float | None = None
    finish_s: float | None = None
    n_tokens: int = 0

    @property
    def ttft_ms(self) -> float | None:
        # Measured from the INTENDED arrival time, never from when the request was
        # actually handed to the scheduler (the coordinated-omission rule).
        return None if self.first_token_s is None else 1e3 * (self.first_token_s - self.arrival_s)

    @property
    def tpot_ms(self) -> float | None:
        if self.finish_s is None or self.first_token_s is None or self.n_tokens < 2:
            return None
        return 1e3 * (self.finish_s - self.first_token_s) / (self.n_tokens - 1)

    def meets(self, slo: SLO) -> bool:
        if self.finish_s is None or self.ttft_ms is None:
            return False
        return self.ttft_ms <= slo.ttft_ms and (self.tpot_ms is None or self.tpot_ms <= slo.tpot_ms)


def _pct(xs: Sequence[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(outcomes: Sequence[Outcome], slo: SLO, window_s: float, wall_s: float) -> dict[str, Any]:
    done = [o for o in outcomes if o.finish_s is not None]
    good = [o for o in done if o.meets(slo)]
    ttft = [o.ttft_ms for o in done if o.ttft_ms is not None]
    tpot = [o.tpot_ms for o in done if o.tpot_ms is not None]
    tokens = sum(o.n_tokens for o in done)
    return {
        "n_requests": len(outcomes),
        "n_completed": len(done),
        "n_within_slo": len(good),
        "slo_attainment": len(good) / len(outcomes) if outcomes else 0.0,
        "goodput_rps": len(good) / window_s if window_s > 0 else 0.0,
        "throughput_tok_s": tokens / wall_s if wall_s > 0 else 0.0,
        "ttft_ms_p50": _pct(ttft, 0.5), "ttft_ms_p99": _pct(ttft, 0.99),
        "tpot_ms_p50": _pct(tpot, 0.5), "tpot_ms_p99": _pct(tpot, 0.99),
        "window_s": window_s, "wall_s": wall_s,
        "slo": asdict(slo),
    }
