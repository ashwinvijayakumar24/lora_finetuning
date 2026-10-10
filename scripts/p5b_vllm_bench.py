#!/usr/bin/env python3
"""Claim L7: the same multi-tenant workload on vLLM with ``--enable-lora``. PENDING (needs CUDA + vLLM).

vLLM is not installed locally (Apple M4, no CUDA), so this script has NOT been
run. It is written so the comparison is fair by construction:

* **Same requests.** Prompts (token ids), output lengths, arrival times and the
  tenant assigned to each request come from ``playparse.serving.p5b_workload``,
  the module ``scripts/p5b_bench.py`` uses, with the same seed.
* **Same adapters.** Tenant ``i`` is ``synthetic_adapter(seed=1000 + i)`` in both
  drivers; here it is exported to a PEFT directory and served by vLLM as a
  ``LoRARequest``.
* **Same SLO and offered rate.** Pass ``--from-artifact`` with one of our
  ``results/p5b/bench_cuda_*.json`` files: the frozen SLO thresholds and the
  offered rate are copied from it, so both systems are judged against the same
  absolute TTFT/TPOT bounds at the same load. (Without it, the script anchors an
  SLO on vLLM's own unloaded latency, which answers a different question.)
* **Same scoring.** ``p5b_workload.summarize``: goodput = requests within both
  bounds per second of the arrival window, TTFT from the intended arrival time.
* **Same model, dtype and greedy decoding**, EOS ignored so output length is
  controlled.

L7 is earned if our goodput is within the factor stated in
``scripts/p5b_bench.py`` (``L7_FACTOR = 2.0``) of vLLM's, cell by cell. Run both
in ONE allocation, back to back (see scripts/slurm/p5b_vllm.sbatch).

Setup on the cluster (separate env; vLLM pins its own torch):

    conda create -n vllm python=3.11 -y && conda activate vllm
    pip install "vllm>=0.6" safetensors numpy
    pip install -e <this repo> --no-deps        # for playparse.serving.p5b_workload
    export PLAYPARSE_SERVING_DIR=<llm_serving_layer>   # its bench/ package builds the workload
    export PLAYPARSE_ENGINE_DIR=<llm_inference_engine> # imported (not run) by the workload helpers

    python scripts/p5b_vllm_bench.py --model-path $PLAYPARSE_WEIGHTS \
        --from-artifact results/p5b/bench_cuda_real_L16_<stamp>.json --ns 1,4,16,64,256
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from playparse.serving.p5b_workload import (  # noqa: E402
    SLO,
    Outcome,
    WorkloadSpec,
    build_requests,
    export_peft_adapters,
    summarize,
)


async def run_open_loop(engine, reqs, lora_paths, max_lora_rank):
    """Dispatch each request at its intended arrival time; stream to time first token and finish."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    outcomes = {r.request_id: Outcome(r.request_id, r.adapter_id, r.arrival_s) for r in reqs}
    t0 = time.perf_counter()

    async def one(r):
        delay = r.arrival_s - (time.perf_counter() - t0)
        if delay > 0:
            await asyncio.sleep(delay)
        lora = None
        if r.adapter_id is not None:
            idx = int(r.adapter_id.split("-")[1])
            lora = LoRARequest(r.adapter_id, idx + 1, str(lora_paths[idx]))
        params = SamplingParams(max_tokens=r.max_tokens, temperature=0.0, ignore_eos=True)
        o = outcomes[r.request_id]
        n_prev = 0
        async for out in engine.generate(TokensPrompt(prompt_token_ids=r.prompt_ids), params,
                                         request_id=r.request_id, lora_request=lora):
            n = len(out.outputs[0].token_ids)
            if n > n_prev and o.first_token_s is None:
                o.first_token_s = time.perf_counter() - t0
            n_prev = n
            if out.finished:
                o.finish_s = time.perf_counter() - t0
                o.n_tokens = n
    await asyncio.gather(*(one(r) for r in reqs))
    return list(outcomes.values()), time.perf_counter() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True, help="HF-format Llama 3.2 1B directory")
    ap.add_argument("--from-artifact", help="our bench JSON: copy SLO, offered rate and workload settings")
    ap.add_argument("--ns", default="1,4,16,64,256")
    ap.add_argument("--popularities", default="uniform,zipf")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-loras", type=int, default=32, help="vLLM GPU LoRA slots (match our --n-slots)")
    ap.add_argument("--max-cpu-loras", type=int, default=256)
    ap.add_argument("--n-requests", type=int, default=256)
    ap.add_argument("--rate", type=float, default=None)
    ap.add_argument("--prompt-mean", type=int, default=128)
    ap.add_argument("--output-mean", type=int, default=64)
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--saturation", action="store_true",
                    help="also replay each cell's requests all at t=0 and report tokens/s (the L7 metric, docs/phases/L7.md)")
    ap.add_argument("--adapter-dir", default=str(REPO / "results" / "p5b" / "vllm_adapters"))
    ap.add_argument("--out", default=str(REPO / "results" / "p5b"))
    args = ap.parse_args(argv)

    import vllm
    from vllm import AsyncEngineArgs, AsyncLLMEngine

    slo = None
    if args.from_artifact:
        art = json.loads(Path(args.from_artifact).read_text())
        g = art["arms"]["goodput"]
        slo = SLO(g["slo"]["ttft_ms"], g["slo"]["tpot_ms"], dict(g["slo"].get("anchored_on", {}),
                                                                  source=args.from_artifact))
        s = art["metadata"]["settings"]
        args.rate = args.rate or g["offered_rps"]
        for k in ("n_requests", "prompt_mean", "output_mean", "rank", "max_batch", "seed"):
            setattr(args, k, s.get(k, getattr(args, k)))
        args.max_loras = s.get("n_slots", args.max_loras)

    cfg = json.loads((Path(args.model_path) / "config.json").read_text())
    ns = [int(n) for n in args.ns.split(",")]
    lora_paths = export_peft_adapters(cfg, max(ns), args.rank, args.adapter_dir)

    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=args.model_path, dtype="float16", enable_lora=True, max_lora_rank=max(16, args.rank),
        max_loras=args.max_loras, max_cpu_loras=max(args.max_cpu_loras, max(ns)),
        max_num_seqs=args.max_batch, enable_prefix_caching=False, seed=args.seed,
    ))
    common = dict(prompt_mean=args.prompt_mean, output_mean=args.output_mean,
                  vocab_size=cfg["vocab_size"], seed=args.seed)

    if slo is None and args.rate is None:
        raise SystemExit("--rate or --from-artifact is required so both systems see the same offered load")

    async def amain():
        # ONE event loop for the whole run: the async engine's background loop is
        # bound to the loop it was first used on.
        nonlocal slo
        if slo is None:   # vLLM's own unloaded latency (a different question; prefer --from-artifact)
            cal, _ = build_requests(WorkloadSpec(n_requests=5, n_adapters=0, rate_rps=1.0,
                                                 **dict(common, seed=args.seed + 1)))
            ttft, tpot = [], []
            for r in cal:
                r.arrival_s = 0.0
                out, _ = await run_open_loop(engine, [r], lora_paths, args.rank)
                ttft.append(out[0].ttft_ms)
                tpot.append(out[0].tpot_ms)
            slo = SLO.from_unloaded(statistics.median(ttft), statistics.median(t for t in tpot if t))
        cells = []
        for n in ns:
            for pop in (["uniform"] if n == 1 else args.popularities.split(",")):
                reqs, realized = build_requests(WorkloadSpec(n_requests=args.n_requests, n_adapters=n,
                                                             popularity=pop, rate_rps=args.rate, **common))
                outcomes, wall = await run_open_loop(engine, reqs, lora_paths, args.rank)
                summ = summarize(outcomes, slo, max(r.arrival_s for r in reqs), wall)
                summ.update({"n_adapters": n, "popularity": pop, "system": "vllm", "realized": realized})
                line = (f"  vllm N={n:<4} {pop:<7} goodput {summ['goodput_rps']:6.2f} req/s  "
                        f"attain {summ['slo_attainment']:5.1%}")
                if args.saturation:
                    # Same requests, all at t=0: throughput independent of the SLO and offered rate.
                    burst = [type(r)(r.request_id + "-sat", r.prompt_ids, r.max_tokens, r.adapter_id, 0.0)
                             for r in reqs]
                    outs_s, wall_s = await run_open_loop(engine, burst, lora_paths, args.rank)
                    toks = sum(o.n_tokens for o in outs_s)
                    summ["saturation"] = {"req_s": len(outs_s) / wall_s, "tok_s": toks / wall_s, "wall_s": wall_s}
                    line += f"  | saturation {toks / wall_s:7.1f} tok/s"
                cells.append(summ)
                print(line)
        return cells

    cells = asyncio.run(amain())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = out / f"vllm_{stamp}.json"
    path.write_text(json.dumps({
        "metadata": {"vllm": vllm.__version__, "timestamp_utc": stamp, "settings": vars(args),
                     "slo": slo.__dict__},
        "cells": cells,
    }, indent=1, default=str))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
