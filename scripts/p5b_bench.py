#!/usr/bin/env python3
"""P5b benchmark: many LoRA adapters on one base model (claims L3 batch-32, L4, L5).

Arms (``--arms``, comma-separated):

  kernel    The adapted projections of one forward pass (7 per layer), base vs
            base + low-rank term, for the v1 (loop) and v2 (BGMV) kernels, at
            batch 1 and 32, ranks 8/16/64, and 1..32 distinct adapters in the
            batch. Isolates the kernel from everything else.            -> L4 kernel, L3
  decode32  End-to-end decode steps through the AdapterScheduler with 32
            sequences decoding together: base, one shared adapter at r=8/16/64
            (the L3 batch-32 arm: "unmerged overhead at batch 32"), and 32
            distinct adapters (v1 vs v2).                               -> L3, L4
  goodput   Open-loop multi-tenant traffic (Poisson arrivals) through the
            scheduler + adapter pool. N = 1/4/16/64/256 tenant adapters, uniform
            vs Zipf popularity, v1 vs v2. Goodput under an SLO anchored to the
            unloaded base latency (serving layer multipliers).          -> L4
  capacity  Max tenants on one GPU: one merged model copy per tenant vs one base
            + an adapter pool. Measured bytes per base copy, per adapter slot and
            per KV token, then the analytic count for a target GPU.      -> L5

Models: ``--model real`` loads the 1B weights; ``--model synthetic`` builds a
random model with the real 1B shapes and ``--layers`` layers (timing and memory do
not depend on weight values, and fewer layers keep a laptop run small). Adapters
are always synthetic (seeded random A, B), as in the S-LoRA paper's scale tests.

Timing: host clock around ``Scheduler.step()``; every step ends in
``argmax(...).tolist()``, which synchronises the device. The kernel arm
synchronises explicitly around its timed loops.

    python scripts/p5b_bench.py --device mps --model synthetic --layers 4 --quick
    python scripts/p5b_bench.py --device cuda --model real                 # H100 (Slurm)
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from playparse.paths import WEIGHTS  # noqa: E402
from playparse.serving._serving_path import ensure_serving_importable  # noqa: E402

ensure_serving_importable()

import torch  # noqa: E402

from engine.loader import load_config, load_weights_gpu  # noqa: E402
from serving.memory.allocator import BlockAllocator  # noqa: E402
from serving.scheduler.scheduler import SchedulerConfig  # noqa: E402

from playparse.serving._provenance import run_metadata  # noqa: E402
from playparse.serving._testing import paged_backend, random_engine_weights  # noqa: E402
from playparse.serving.adapter import SUPPORTED_MODULES, engine_weight_name  # noqa: E402
from playparse.serving.adapter_pool import AdapterPool, build_plan, synthetic_adapter  # noqa: E402
from playparse.serving.adapter_scheduler import AdapterRequest, AdapterScheduler  # noqa: E402
from playparse.serving.lora_engine import use_adapter  # noqa: E402
from playparse.serving.multi_lora import (  # noqa: E402
    MultiLoRAModelGPU,
    PlannedSelection,
    PooledLoRALinear,
    install_multi_lora_linear,
    multi_lora_linear,
)
from playparse.serving.p5b_workload import (  # noqa: E402
    ADAPTER_SEED0,
    SLO,
    Outcome,
    WorkloadSpec,
    build_requests,
    summarize,
)

BLOCK = 16
LLAMA_1B = {
    "hidden_size": 2048, "intermediate_size": 8192, "num_hidden_layers": 16,
    "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 64,
    "vocab_size": 128256, "max_position_embeddings": 4096, "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0, "initializer_range": 0.02,
    "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                     "original_max_position_embeddings": 8192, "rope_type": "llama3"},
}

# Stated before any GPU run (PRD §10, "threshold stated in advance"). The owner may
# revise them before the authoritative H100 run, never after seeing it.
L3_THRESHOLD_PCT_R16_BATCH32 = 15.0     # unmerged decode-step overhead, one shared r=16 adapter
L4_V2_MIN_RETENTION_N256 = 0.5          # v2 goodput at N=256 >= 50% of N=1, uniform popularity
L7_FACTOR = 2.0                         # our goodput within 2x of vLLM's on the same workload


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def device_allocated(device: str) -> int | None:
    if device.startswith("cuda"):
        return torch.cuda.memory_allocated()
    if device == "mps":
        return torch.mps.current_allocated_memory()
    return None


def build_model(args):
    if args.model == "real":
        cfg = load_config(WEIGHTS)
        weights = load_weights_gpu(WEIGHTS, cfg, device=args.device)
        if args.layers and args.layers < cfg["num_hidden_layers"]:
            cfg = dict(cfg, num_hidden_layers=args.layers)
            weights = {k: v for k, v in weights.items()
                       if not k.startswith("model.layers.") or int(k.split(".")[2]) < args.layers}
    else:
        cfg = dict(LLAMA_1B, num_hidden_layers=args.layers or 16)
        weights = random_engine_weights(cfg, seed=0, device=args.device)
    return cfg, weights


def make_pool(cfg, args, n_slots, max_rank, kernel, n_adapters, rank, seed0=ADAPTER_SEED0):
    pool = AdapterPool(cfg, n_slots=n_slots, max_rank=max_rank, device=args.device, kernel=kernel)
    for i in range(n_adapters):
        # Generated and registered one at a time: the pool keeps only its fp16 host
        # copy, so 256 adapters never sit in memory twice.
        pool.register(synthetic_adapter(cfg, rank, seed=seed0 + i, b_std=0.01), f"tenant-{i:03d}")
    return pool


# --------------------------------------------------------------------------
# Arm: kernel microbenchmark
# --------------------------------------------------------------------------

def arm_kernel(cfg, weights, args) -> list[dict]:
    """Per (batch, rank, distinct adapters): base, v1 and v2, measured in interleaved rounds.

    Each round times base, then every kernel, back to back; the reported number is
    the MINIMUM over rounds of the per-round mean. On a shared machine contention
    only ever adds time, so the minimum is the best available estimate of the
    uncontended cost, and timing base inside every round keeps drift off the ratio.
    """
    install_multi_lora_linear()
    keys = [engine_weight_name(i, m) for i in range(cfg["num_hidden_layers"]) for m in SUPPORTED_MODULES]
    rows = []
    for batch in args.batches:
        xs = {}
        for k in keys:
            d_in = weights[k].shape[1]
            if d_in not in xs:
                xs[d_in] = torch.randn(batch, d_in, device=args.device, dtype=torch.float16) * 0.1

        def run(ws, sel):
            with use_adapter(sel):
                for k in keys:
                    w = ws[k]
                    multi_lora_linear(xs[(w.base if isinstance(w, PooledLoRALinear) else w).shape[1]], w)

        def timed(ws, sel, iters):
            sync(args.device)
            t = time.perf_counter()
            for _ in range(iters):
                run(ws, sel)
            sync(args.device)
            return (time.perf_counter() - t) / iters * 1e3

        for r in args.ranks:
            for n in sorted({min(n, batch) for n in args.kernel_distinct}):
                pool = make_pool(cfg, args, n_slots=n, max_rank=r, kernel="v1", n_adapters=n, rank=r)
                slots = [pool.acquire(f"tenant-{i:03d}") for i in range(n)]
                # A decode-shaped batch: one row per sequence, adapters round-robin. The
                # plan is built once, as the serving path builds it once per forward pass.
                seq_slots = [slots[i % n] for i in range(batch)]
                sel = PlannedSelection(names=pool.slot_names(),
                                       row_slots=torch.tensor(seq_slots, device=args.device),
                                       plan=build_plan(seq_slots, [1] * batch, args.device))
                ws = {k: PooledLoRALinear(k, weights[k], pool) for k in keys}
                for kernel in args.kernels:          # warm every path once
                    pool.kernel = kernel
                    for _ in range(args.warmup):
                        run(ws, sel)
                for _ in range(args.warmup):
                    run(weights, None)
                per_iter = max(1, args.iters // args.rounds)
                samples = {"base": [], **{k: [] for k in args.kernels}}
                for _ in range(args.rounds):
                    samples["base"].append(timed(weights, None, per_iter))
                    for kernel in args.kernels:
                        pool.kernel = kernel
                        samples[kernel].append(timed(ws, sel, per_iter))
                base_ms = min(samples["base"])
                rows.append({"batch": batch, "rank": r, "distinct_adapters": n, "arm": "base",
                             "ms_min": base_ms, "ms_median": statistics.median(samples["base"])})
                msg = f"  kernel batch={batch:<3} r={r:<3} distinct={n:<3} base {base_ms:7.2f} ms"
                for kernel in args.kernels:
                    ms = min(samples[kernel])
                    rows.append({"batch": batch, "rank": r, "distinct_adapters": n, "arm": kernel,
                                 "kernel": kernel, "ms_min": ms, "ms_median": statistics.median(samples[kernel]),
                                 "overhead_pct": 100 * (ms - base_ms) / base_ms})
                    msg += f" | {kernel} {ms:7.2f} ms ({100 * (ms - base_ms) / base_ms:+.0f}%)"
                print(msg)
                del pool, ws
                gc.collect()
    return rows


# --------------------------------------------------------------------------
# Scheduler driver
# --------------------------------------------------------------------------

def make_scheduler(cfg, weights, pool, args, max_batch, num_blocks):
    model = MultiLoRAModelGPU(weights, cfg, pool, device=args.device)
    alloc = BlockAllocator(num_blocks, BLOCK)
    backend = paged_backend(cfg, num_blocks, BLOCK, device=args.device)
    return AdapterScheduler(model, backend, alloc, SchedulerConfig(
        max_batch_size=max_batch, max_prefill_tokens=args.max_prefill_tokens))


def arm_decode32(cfg, weights, args) -> list[dict]:
    """Decode steps with 32 sequences all decoding, under several adapter mixes, interleaved.

    Every arm gets its own scheduler (own pool, own KV), all built up front and
    prefilled; then each round steps every arm once, in turn. Slow drift on a
    shared machine therefore lands on all arms alike. Reported: median and
    minimum step time over the rounds, overhead vs base from the medians.
    """
    batch, prompt_len, steps = 32, 32, args.decode_steps
    num_blocks = batch * ((prompt_len + steps + args.warmup + 8) // BLOCK + 2)
    arms = [("base", "-", None, 0)]
    for kernel in args.kernels:
        arms += [(f"shared_r{r}", kernel, "shared", r) for r in args.ranks]
        arms += [("distinct32_r16", kernel, "distinct", 16)]
    scheds = []
    for label, kernel, mode, r in arms:
        n_ad = 0 if mode is None else (1 if mode == "shared" else batch)
        pool = make_pool(cfg, args, n_slots=max(1, n_ad), max_rank=max(r, 1),
                         kernel=kernel if mode else "v2", n_adapters=n_ad, rank=max(r, 1))
        sched = make_scheduler(cfg, weights, pool, args, max_batch=batch, num_blocks=num_blocks)
        for i in range(batch):
            aid = None if mode is None else f"tenant-{(0 if mode == 'shared' else i):03d}"
            sched.add_request(AdapterRequest(
                request_id=f"d{i}", prompt_ids=[(7 * i + j) % 1000 + 10 for j in range(prompt_len)],
                max_tokens=10_000, adapter_id=aid, ignore_eos=True))
        while any(not r_.prefill_done for r_ in sched.running) or sched.waiting:
            sched.step()
        for _ in range(args.warmup):
            sched.step()
        scheds.append(sched)
    times = [[] for _ in arms]
    for _ in range(steps):
        for i, sched in enumerate(scheds):
            t = time.perf_counter()
            st = sched.step()
            times[i].append(time.perf_counter() - t)
            assert st.n_decode == batch, f"expected {batch} decodes, got {st.n_decode}"
    base_ms = 1e3 * statistics.median(times[0])
    rows = []
    for (label, kernel, mode, r), ts in zip(arms, times):
        ms = 1e3 * statistics.median(ts)
        row = {"arm": label, "kernel": kernel, "rank": r, "batch": batch,
               "ms_per_step_p50": ms, "ms_per_step_min": 1e3 * min(ts), "tok_s": batch / (ms / 1e3),
               "overhead_pct": None if mode is None else 100 * (ms - base_ms) / base_ms}
        rows.append(row)
        print(f"  decode32 {label:<16} {kernel:<3} p50 {ms:8.2f} ms/step  min {row['ms_per_step_min']:8.2f}"
              f"  {row['tok_s']:7.1f} tok/s" + ("" if mode is None else f"  ({row['overhead_pct']:+.1f}%)"))
    del scheds
    gc.collect()
    return rows


def drive(sched, reqs, *, realtime=True):
    """Open-loop: hand each request to the scheduler at its intended arrival time."""
    outcomes = {r.request_id: Outcome(r.request_id, r.adapter_id, r.arrival_s) for r in reqs}
    pending = sorted(reqs, key=lambda r: r.arrival_s)
    t0 = time.perf_counter()

    def now():
        return time.perf_counter() - t0

    def mk(r):
        o = outcomes[r.request_id]

        def on_token(_tok, o=o):
            if o.first_token_s is None:
                o.first_token_s = now()
            o.n_tokens += 1

        def on_finish(req, o=o):
            if req.state == "finished":
                o.finish_s = now()
        return AdapterRequest(request_id=r.request_id, prompt_ids=r.prompt_ids, max_tokens=r.max_tokens,
                              adapter_id=r.adapter_id, ignore_eos=True, on_token=on_token, on_finish=on_finish)

    i = 0
    while i < len(pending) or sched.has_work:
        t = now()
        while i < len(pending) and (not realtime or pending[i].arrival_s <= t):
            ok = sched.add_request(mk(pending[i]))
            assert ok, f"request {pending[i].request_id} was shed"
            i += 1
        if sched.has_work:
            sched.step()
        elif i < len(pending):
            time.sleep(max(0.0, min(0.005, pending[i].arrival_s - now())))
    return list(outcomes.values()), now()


def arm_goodput(cfg, weights, args) -> dict:
    vocab = cfg["vocab_size"]
    num_blocks = args.kv_blocks
    common = dict(prompt_mean=args.prompt_mean, output_mean=args.output_mean, vocab_size=vocab,
                  block_size=BLOCK, seed=args.seed)

    # 1. Unloaded latency (base model, one request at a time) -> the SLO, frozen here.
    pool0 = make_pool(cfg, args, 1, 8, "v2", 0, 8)
    sched = make_scheduler(cfg, weights, pool0, args, args.max_batch, num_blocks)
    warm, _ = build_requests(WorkloadSpec(n_requests=4, n_adapters=0, rate_rps=1.0, **common))
    drive(sched, warm, realtime=False)
    cal_reqs, _ = build_requests(WorkloadSpec(n_requests=args.calib_requests, n_adapters=0, rate_rps=1.0,
                                              **dict(common, seed=args.seed + 1)))
    ttfts, tpots = [], []
    for r in cal_reqs:
        # Dispatched immediately, so its intended arrival is the start of its own
        # run. Keeping the Poisson arrival time here gave a NEGATIVE TTFT
        # (docs/issues/p5b-calibration-negative-ttft.md).
        r.arrival_s = 0.0
        out, _ = drive(sched, [r], realtime=False)
        ttfts.append(out[0].ttft_ms)
        tpots.append(out[0].tpot_ms)
    slo = SLO.from_unloaded(statistics.median(ttfts), statistics.median(t for t in tpots if t))

    # 2. Closed-loop base capacity -> the offered rate.
    burst, _ = build_requests(WorkloadSpec(n_requests=2 * args.max_batch, n_adapters=0, rate_rps=1.0, **common))
    for r in burst:
        r.arrival_s = 0.0
    out, wall = drive(sched, burst, realtime=False)
    capacity_rps = len(out) / wall
    rate = args.rate or args.load_factor * capacity_rps
    print(f"  SLO: TTFT <= {slo.ttft_ms:.1f} ms, TPOT <= {slo.tpot_ms:.2f} ms "
          f"(unloaded p50 x{slo.anchored_on['ttft_multiplier']:.0f}/x{slo.anchored_on['tpot_multiplier']:.0f}); "
          f"base capacity {capacity_rps:.2f} req/s -> offered {rate:.2f} req/s")
    del sched, pool0
    gc.collect()

    cells = []
    for n in args.ns:
        # With one tenant, popularity cannot matter: run it once.
        for pop in (args.popularities[:1] if n == 1 else args.popularities):
            reqs, realized = build_requests(WorkloadSpec(
                n_requests=args.n_requests, n_adapters=n, popularity=pop, rate_rps=rate, **common))
            window = max(r.arrival_s for r in reqs)
            for kernel in args.kernels:
                pool = make_pool(cfg, args, n_slots=min(args.n_slots, n), max_rank=args.rank, kernel=kernel,
                                 n_adapters=n, rank=args.rank)
                sched = make_scheduler(cfg, weights, pool, args, args.max_batch, num_blocks)
                outcomes, wall = drive(sched, reqs)
                summ = summarize(outcomes, slo, window, wall)
                summ.update({"n_adapters": n, "popularity": pop, "kernel": kernel,
                             "n_slots": pool.n_slots, "rank": args.rank, "offered_rps": rate,
                             "pool": pool.snapshot(), "realized": realized,
                             "preemptions": sched.preemption.total})
                line = (f"  goodput N={n:<4} {pop:<7} {kernel}  goodput {summ['goodput_rps']:6.2f} req/s  "
                        f"attain {summ['slo_attainment']:5.1%}  TTFT p99 {summ['ttft_ms_p99'] or float('nan'):7.1f} ms  "
                        f"hit {pool.stats.hit_rate:4.0%} loads {pool.stats.loads}")
                del sched
                if args.saturation:
                    # Same requests, all at t=0: the system's capacity for this tenant mix,
                    # far less sensitive to where the offered rate sits relative to the knee.
                    pool.stats = type(pool.stats)()
                    sched_s = make_scheduler(cfg, weights, pool, args, args.max_batch, num_blocks)
                    burst = [type(r)(r.request_id, r.prompt_ids, r.max_tokens, r.adapter_id, 0.0) for r in reqs]
                    outs_s, wall_s = drive(sched_s, burst, realtime=False)
                    toks = sum(o.n_tokens for o in outs_s)
                    summ["saturation"] = {"req_s": len(outs_s) / wall_s, "tok_s": toks / wall_s,
                                          "wall_s": wall_s, "pool": pool.snapshot()}
                    line += f"  | saturation {toks / wall_s:7.1f} tok/s"
                    del sched_s
                cells.append(summ)
                print(line)
                del pool
                gc.collect()
    return {"slo": slo.__dict__, "capacity_rps": capacity_rps, "offered_rps": rate, "cells": cells}


# --------------------------------------------------------------------------
# Arm: capacity (L5)
# --------------------------------------------------------------------------

def arm_capacity(cfg, weights, args) -> dict:
    seen, base_bytes = set(), 0
    for t in weights.values():
        if isinstance(t, torch.Tensor) and t.data_ptr() not in seen:   # tied lm_head counted once
            seen.add(t.data_ptr())
            base_bytes += t.nbytes
    full_layers = 16
    per_layer_scale = full_layers / cfg["num_hidden_layers"]
    measured = {}
    for r in args.ranks:
        before = device_allocated(args.device)
        pool = AdapterPool(cfg, n_slots=8, max_rank=r, device=args.device)
        sync(args.device)
        after = device_allocated(args.device)
        measured[r] = {
            "bytes_per_slot": pool.bytes_per_slot(),
            "device_alloc_delta_8_slots_plus_zero": None if before is None else after - before,
            "formula_8_slots_plus_zero": pool.device_bytes(),
        }
        del pool
        gc.collect()
    kv_bytes_per_token = 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * cfg["head_dim"] * 2

    # Scale to the full 16-layer 1B for the analytic count (all three quantities are per-layer linear,
    # except embeddings, which are layer-independent).
    emb = weights["model.embed_tokens.weight"].nbytes
    base_full = emb + (base_bytes - emb) * per_layer_scale
    gpu_bytes = args.gpu_gb * 2**30 * args.usable_fraction
    table = []
    for kv_tokens in args.kv_tokens_per_tenant:
        kv_full = kv_bytes_per_token * per_layer_scale * kv_tokens
        row = {"kv_tokens_per_tenant": kv_tokens,
               "merged_copies": int(gpu_bytes // (base_full + kv_full))}
        for r in args.ranks:
            ad = measured[r]["bytes_per_slot"] * per_layer_scale
            row[f"pool_all_resident_r{r}"] = int((gpu_bytes - base_full) // (ad + kv_full))
        table.append(row)
    return {
        "measured_model_layers": cfg["num_hidden_layers"],
        "base_bytes_measured": base_bytes, "base_bytes_full_1b": base_full,
        "adapter_slot_bytes_measured": {r: m["bytes_per_slot"] for r, m in measured.items()},
        "adapter_slot_bytes_full_1b": {r: m["bytes_per_slot"] * per_layer_scale for r, m in measured.items()},
        "pool_alloc_check": measured,
        "kv_bytes_per_token_full_1b": kv_bytes_per_token * per_layer_scale,
        "target_gpu_gb": args.gpu_gb, "usable_fraction": args.usable_fraction,
        "table": table,
        "note": ("pool_all_resident counts tenants whose adapters are ALL on the GPU at once. With host "
                 "offload the number of registered tenants is bounded by host RAM, and only the active "
                 "set must fit in the slots."),
    }


# --------------------------------------------------------------------------

def render_markdown(result: dict) -> str:
    """Markdown tables for every arm present in a result file (``--render``)."""
    out = [f"label: {result['metadata']['label']}, device: {result['metadata']['device']}, "
           f"layers: {result['model_config']['num_hidden_layers']}"]
    arms = result["arms"]
    if "kernel" in arms:
        out += ["", "### kernel (adapted projections of one forward pass, min over rounds)", "",
                "| batch | rank | distinct | base ms | v1 ms (overhead) | v2 ms (overhead) |",
                "|---|---|---|---|---|---|"]
        rows = arms["kernel"]
        keys = sorted({(r["batch"], r["rank"], r["distinct_adapters"]) for r in rows})
        for b, rk, n in keys:
            cell = {r["arm"]: r for r in rows if (r["batch"], r["rank"], r["distinct_adapters"]) == (b, rk, n)}
            def fmt(k):
                r = cell.get(k)
                return "-" if r is None else f"{r['ms_min']:.2f} ({r['overhead_pct']:+.0f}%)"
            out.append(f"| {b} | {rk} | {n} | {cell['base']['ms_min']:.2f} | {fmt('v1')} | {fmt('v2')} |")
    if "decode32" in arms:
        out += ["", "### decode32 (32 sequences decoding, interleaved rounds)", "",
                "| arm | kernel | ms/step p50 | ms/step min | tok/s | overhead vs base (p50) |", "|---|---|---|---|---|---|"]
        for r in arms["decode32"]:
            ov = "-" if r["overhead_pct"] is None else f"{r['overhead_pct']:+.1f}%"
            out.append(f"| {r['arm']} | {r['kernel']} | {r['ms_per_step_p50']:.1f} | {r['ms_per_step_min']:.1f} "
                       f"| {r['tok_s']:.0f} | {ov} |")
    if "goodput" in arms:
        g = arms["goodput"]
        out += ["", f"### goodput (SLO TTFT <= {g['slo']['ttft_ms']:.0f} ms, TPOT <= {g['slo']['tpot_ms']:.1f} ms; "
                f"offered {g['offered_rps']:.2f} req/s = load factor x base capacity {g['capacity_rps']:.2f})", "",
                "| N | popularity | kernel | goodput req/s | attainment | TTFT p50 / p99 ms | pool hit rate | loads | saturation tok/s |",
                "|---|---|---|---|---|---|---|---|---|"]
        for c in g["cells"]:
            sat = c.get("saturation", {}).get("tok_s")
            out.append(f"| {c['n_adapters']} | {c['popularity']} | {c['kernel']} | {c['goodput_rps']:.2f} | "
                       f"{c['slo_attainment']:.0%} | {c['ttft_ms_p50'] or 0:.0f} / {c['ttft_ms_p99'] or 0:.0f} | "
                       f"{c['pool']['hit_rate']:.0%} | {c['pool']['loads']} | "
                       f"{'-' if sat is None else f'{sat:.1f}'} |")
    if "capacity" in arms:
        c = arms["capacity"]
        ranks = sorted(int(r) for r in c["adapter_slot_bytes_full_1b"])
        out += ["", f"### capacity (full 16-layer 1B, {c['target_gpu_gb']:.0f} GB x {c['usable_fraction']} usable)", "",
                f"base copy {c['base_bytes_full_1b'] / 2**30:.2f} GiB; adapter slot "
                + ", ".join(f"r={r}: {c['adapter_slot_bytes_full_1b'][r if r in c['adapter_slot_bytes_full_1b'] else str(r)] / 2**20:.1f} MiB"
                            for r in ranks)
                + f"; KV {c['kv_bytes_per_token_full_1b'] / 2**10:.0f} KiB/token", "",
                "| KV tokens per tenant | merged copies | " + " | ".join(f"pool, r={r}" for r in ranks) + " |",
                "|---" * (2 + len(ranks)) + "|"]
        for row in c["table"]:
            out.append(f"| {row['kv_tokens_per_tenant']} | {row['merged_copies']} | "
                       + " | ".join(str(row[f'pool_all_resident_r{r}']) for r in ranks) + " |")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--model", choices=("synthetic", "real"), default="synthetic")
    ap.add_argument("--layers", type=int, default=None, help="truncate/build with this many layers")
    ap.add_argument("--arms", default="kernel,decode32,goodput,capacity")
    ap.add_argument("--kernels", default="v1,v2")
    ap.add_argument("--ranks", default="8,16,64")
    ap.add_argument("--batches", default="1,32")
    ap.add_argument("--kernel-distinct", default="1,4,16,32")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=5, help="kernel arm: interleaved timing rounds")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--decode-steps", type=int, default=20)
    ap.add_argument("--ns", default="1,4,16,64,256")
    ap.add_argument("--popularities", default="uniform,zipf")
    ap.add_argument("--n-slots", type=int, default=32, help="adapter pool slots for the goodput arm")
    ap.add_argument("--rank", type=int, default=16, help="adapter rank for the goodput arm")
    ap.add_argument("--n-requests", type=int, default=256)
    ap.add_argument("--calib-requests", type=int, default=5)
    ap.add_argument("--prompt-mean", type=int, default=128)
    ap.add_argument("--output-mean", type=int, default=64)
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--max-prefill-tokens", type=int, default=512)
    ap.add_argument("--kv-blocks", type=int, default=2048)
    ap.add_argument("--rate", type=float, default=None, help="offered req/s (default: load-factor x capacity)")
    ap.add_argument("--load-factor", type=float, default=0.7)
    ap.add_argument("--saturation", action="store_true",
                    help="goodput arm: also measure closed-loop capacity per cell")
    ap.add_argument("--gpu-gb", type=float, default=80.0)
    ap.add_argument("--usable-fraction", type=float, default=0.9)
    ap.add_argument("--kv-tokens-per-tenant", default="0,2048,8192")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="laptop-sized settings for every arm")
    ap.add_argument("--out", default=str(REPO / "results" / "p5b"))
    ap.add_argument("--render", metavar="RESULT_JSON", help="print markdown tables for a result file and exit")
    args = ap.parse_args(argv)
    if args.render:
        print(render_markdown(json.loads(Path(args.render).read_text())))
        return 0

    if args.quick:
        args.iters, args.warmup, args.decode_steps = 15, 3, 12
        args.n_requests, args.prompt_mean, args.output_mean = 48, 48, 16
        args.n_slots, args.kv_blocks, args.max_batch = 8, 512, 16
        args.load_factor, args.saturation = 0.5, True
        if args.layers is None:
            args.layers = 4
    for name in ("kernels", "popularities", "arms"):
        setattr(args, name, [s for s in getattr(args, name).split(",") if s])
    for name in ("ranks", "batches", "kernel_distinct", "ns", "kv_tokens_per_tenant"):
        setattr(args, name, [int(s) for s in getattr(args, name).split(",") if s])

    cfg, weights = build_model(args)
    settings = {k: v for k, v in vars(args).items() if k != "out"}
    result = {
        "metadata": run_metadata(args.device, settings),
        "model_config": {k: cfg[k] for k in ("hidden_size", "intermediate_size", "num_hidden_layers",
                                             "num_attention_heads", "num_key_value_heads", "vocab_size")},
        "thresholds_stated_in_advance": {
            "L3_pct_r16_batch32": L3_THRESHOLD_PCT_R16_BATCH32,
            "L4_v2_min_retention_n256_uniform": L4_V2_MIN_RETENTION_N256,
            "L7_factor_vs_vllm": L7_FACTOR,
        },
        "arms": {},
    }
    print(f"P5b bench on {args.device}, model={args.model}, layers={cfg['num_hidden_layers']} "
          f"[{result['metadata']['label']}]")
    arms = {"kernel": arm_kernel, "decode32": arm_decode32, "goodput": arm_goodput, "capacity": arm_capacity}
    for name in args.arms:
        print(f"\n## {name}")
        t = time.perf_counter()
        result["arms"][name] = arms[name](cfg, weights, args)
        print(f"  ({time.perf_counter() - t:.0f} s)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out / f"bench_{args.device.split(':')[0]}_{args.model}_L{cfg['num_hidden_layers']}_{stamp}.json"
    path.write_text(json.dumps(result, indent=1, default=str))
    print(f"\nwrote {path}\n")
    print(render_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
