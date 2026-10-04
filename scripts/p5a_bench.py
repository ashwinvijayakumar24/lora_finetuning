#!/usr/bin/env python3
"""P5a benchmark: what merged and unmerged LoRA cost at inference (claims L1, L3).

Arms, all on the engine's torch path (``LlamaModelGPU``), batch 1, greedy:

  base            the stock engine, before linear() is replaced
  base_repeat     the same model measured again: the run-to-run noise floor for L1
  merged_r16      stock engine on merged weights (W + s*B@A)            -> L1
  base_patched    LoRAModelGPU with no adapter selected: cost of the hook itself
  unmerged_r{R}   LoRAModelGPU with an r=R adapter on all 7 projections  -> L3

Arms are measured interleaved (round-robin per run), so slow drift on a shared
machine spreads across arms instead of landing on whichever ran last.

Timing follows the engine's own harness (bench/harness.py in the engine repo):
host-clock timestamps around each generated token. That is valid because the
engine's prefill/decode_step end in ``.cpu()``, which synchronises the device on
every token (documented in engine/model_gpu.py::forward_varlen).

Batch 32: the engine is batch-1 by construction (README, "Benchmarks": all
numbers are batch 1), so end-to-end batch-32 decode belongs to P5b, where the
serving layer provides batching. ``--micro`` measures the part LoRA actually
changes at batch 1 and 32 — the 112 projections of one decode step, base vs
base + low-rank path — as a labelled microbenchmark, not an end-to-end number.

Adapters are synthetic (seeded random A and B with the real model's shapes):
latency does not depend on the values, and skipping PEFT/HF keeps memory low.

Usage:
    python scripts/p5a_bench.py --device mps --max-tokens 64 --n-runs 3 --micro
    python scripts/p5a_bench.py --device cuda --max-tokens 128 --n-runs 5 --micro   # H100
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from playparse.paths import WEIGHTS  # noqa: E402
from playparse.serving._engine_path import ensure_engine_importable  # noqa: E402

ENGINE_ROOT = ensure_engine_importable()

import torch  # noqa: E402

from playparse.serving.adapter import adapter_from_tensors, engine_to_peft_key, engine_weight_name, \
    expected_linear_shapes  # noqa: E402

# Same three prompts as the engine's harness, so numbers are comparable to its BENCHMARKS.md.
PROMPTS = {
    "short": "Hello, I am",
    "medium": "The quick brown fox jumped over the lazy dog.",
    "long": ("Explain the fundamental theorem of calculus in detail, "
             "including both the first and second parts, with examples."),
}
MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# Stated before any H100 run (PRD §10, L3 "threshold stated in advance"); the owner may revise
# it before the authoritative run, never after.
L3_THRESHOLD_PCT_R16_BATCH1 = 15.0


def synthetic_adapter(config: dict, r: int, seed: int, b_std: float = 0.01, alpha: float | None = None):
    rng = np.random.default_rng(seed)
    shapes = expected_linear_shapes(config)
    tensors = {}
    for layer in range(config["num_hidden_layers"]):
        for m in MODULES:
            out_dim, in_dim = shapes[m]
            key = engine_weight_name(layer, m)
            tensors[engine_to_peft_key(key, "A")] = (rng.uniform(-1, 1, (r, in_dim)) / np.sqrt(in_dim)).astype(np.float32)
            tensors[engine_to_peft_key(key, "B")] = (rng.standard_normal((out_dim, r)) * b_std).astype(np.float32)
    cfg = {"peft_type": "LORA", "r": r, "lora_alpha": alpha or 2 * r, "bias": "none"}
    return adapter_from_tensors(cfg, tensors, name=f"r{r}", base_config=config)


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def time_generate(model, ids, max_tokens, adapter=None) -> dict:
    """One generate() call, timestamps per token (mirrors the engine harness)."""
    from engine.sampler import greedy
    from engine.scheduler import generate
    from playparse.serving.lora_engine import generate_with_adapter

    gen = generate(model, ids, greedy, max_tokens=max_tokens, max_seq=len(ids) + max_tokens + 1) if adapter is None \
        else generate_with_adapter(model, ids, greedy, adapter, max_tokens=max_tokens, max_seq=len(ids) + max_tokens + 1)
    stamps, toks = [], []
    t0 = time.perf_counter()
    for tok in gen:
        stamps.append(time.perf_counter())
        toks.append(tok)
    itl = np.diff(stamps)
    return {
        "n_prompt": len(ids), "n_new": len(toks),
        "ttft_ms": (stamps[0] - t0) * 1e3,
        "decode_tok_s": (len(toks) - 1) / float(itl.sum()) if len(itl) else float("nan"),
        "itl_p50_ms": float(np.percentile(itl, 50) * 1e3) if len(itl) else float("nan"),
        "itl_p99_ms": float(np.percentile(itl, 99) * 1e3) if len(itl) else float("nan"),
        "tokens": toks,
    }


def micro_linear(weights, adapters, device, batches=(1, 32), iters=50, warmup=10) -> list[dict]:
    """Time the 112 projections of one decode step, base vs + low-rank path, at several batch sizes."""
    from engine import components_gpu
    from playparse.serving.lora_engine import LoRALinear, install_lora_linear, use_adapter

    install_lora_linear()
    keys = [k for k in weights if k.endswith(tuple(f"{m}.weight" for m in MODULES))]
    rows = []
    for batch in batches:
        xs = {}
        for k in keys:
            in_dim = weights[k].shape[1]
            xs[in_dim] = xs.get(in_dim, torch.randn(batch, in_dim, device=device, dtype=torch.float16) * 0.1)

        def run(ws, sel):
            with use_adapter(sel):
                for k in keys:
                    w = ws[k]
                    components_gpu.linear(xs[w.shape[1]], w)

        def bench(ws, sel):
            for _ in range(warmup):
                run(ws, sel)
            sync(device)
            t = time.perf_counter()
            for _ in range(iters):
                run(ws, sel)
            sync(device)
            return (time.perf_counter() - t) / iters * 1e3

        base_ms = bench(weights, None)
        rows.append({"batch": batch, "arm": "base", "ms_per_step_linears": base_ms})
        for name, ad in adapters.items():
            wrapped = dict(weights)
            for k, t in ad.layers.items():
                lw = LoRALinear(k, weights[k])
                lw.add(name, t.A, t.B, t.scale)
                wrapped[k] = lw
            ms = bench(wrapped, name)
            rows.append({"batch": batch, "arm": f"unmerged_{name}", "ms_per_step_linears": ms,
                         "overhead_pct": 100 * (ms - base_ms) / base_ms})
            del wrapped
    return rows


def metadata(args, device) -> dict:
    def _cmd(*c, cwd=REPO):
        try:
            return subprocess.check_output(list(c), text=True, cwd=cwd, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    meta = {
        "label": "LOCAL / INDICATIVE" if not device.startswith("cuda") else "CUDA run",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(), "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "device": device,
        "git_sha": _cmd("git", "rev-parse", "HEAD"),
        "git_dirty": bool(_cmd("git", "status", "--porcelain")),
        "engine_dir": str(ENGINE_ROOT),
        "engine_git_sha": _cmd("git", "rev-parse", "HEAD", cwd=ENGINE_ROOT),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "settings": {k: v for k, v in vars(args).items() if k != "out"},
        "timing": "host clock per token; engine syncs every token via .cpu() (same as engine bench/harness.py)",
        "l3_threshold_pct_r16_batch1": L3_THRESHOLD_PCT_R16_BATCH1,
    }
    if sys.platform == "darwin":
        meta["chip"] = _cmd("sysctl", "-n", "machdep.cpu.brand_string")
        meta["mem_gb"] = int(_cmd("sysctl", "-n", "hw.memsize") or 0) / 2**30
    if device.startswith("cuda"):
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["cuda"] = torch.version.cuda
    return meta


def summarize(rows: list[dict]) -> dict:
    by_arm: dict[str, dict[str, list]] = {}
    for r in rows:
        a = by_arm.setdefault(r["arm"], {"decode_tok_s": [], "ttft_ms": [], "itl_p50_ms": []})
        for k in a:
            a[k].append(r[k])
    base_tps = statistics.median(by_arm["base"]["decode_tok_s"])
    base_ttft = statistics.median(by_arm["base"]["ttft_ms"])
    out = {}
    for arm, v in by_arm.items():
        tps = statistics.median(v["decode_tok_s"])
        ttft = statistics.median(v["ttft_ms"])
        out[arm] = {
            "decode_tok_s_median": tps,
            "decode_tok_s_min": min(v["decode_tok_s"]), "decode_tok_s_max": max(v["decode_tok_s"]),
            "ttft_ms_median": ttft,
            "itl_p50_ms_median": statistics.median(v["itl_p50_ms"]),
            "decode_overhead_pct_vs_base": 100 * (base_tps - tps) / base_tps,
            "ttft_delta_pct_vs_base": 100 * (ttft - base_ttft) / base_ttft,
            "n": len(v["decode_tok_s"]),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else
                    ("mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 64])
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--n-warmup", type=int, default=1)
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument("--micro", action="store_true", help="also run the batch 1/32 linear microbenchmark")
    ap.add_argument("--micro-only", action="store_true")
    ap.add_argument("--out", default=str(REPO / "results" / "p5a"))
    args = ap.parse_args()
    device = args.device

    from transformers import AutoTokenizer
    from engine.loader import load_config, load_weights_gpu
    from engine.model_gpu import LlamaModelGPU
    from playparse.serving.merge import merge_adapter

    cfg = load_config(args.weights)
    tok = AutoTokenizer.from_pretrained(args.weights)
    weights = load_weights_gpu(args.weights, cfg, device=device)
    adapters = {f"r{r}": synthetic_adapter(cfg, r, seed=r) for r in args.ranks}
    meta = metadata(args, device)
    result = {"meta": meta}

    if not args.micro_only:
        prompts = {}
        for key, text in PROMPTS.items():
            ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                          tokenize=True)
            prompts[key] = [int(i) for i in (ids["input_ids"] if hasattr(ids, "keys") else ids)]

        # Base and merged run before linear() is replaced, so they are the stock engine.
        base = LlamaModelGPU(weights, cfg, device=device)
        merge_src = adapters.get("r16") or next(iter(adapters.values()))
        merged = LlamaModelGPU(merge_adapter(weights, merge_src), cfg, device=device)
        from playparse.serving.lora_engine import LoRAModelGPU, install_lora_linear, uninstall_lora_linear
        lora = LoRAModelGPU(weights, cfg, device=device)
        for name, ad in adapters.items():
            lora.load_adapter(ad, name)
        # (arm, model, adapter, needs the linear() replacement). Stock arms run with the engine's
        # original linear() restored, so "base" and "merged" are the unmodified engine.
        arms = [("base", base, None, False), ("base_repeat", base, None, False),
                (f"merged_{merge_src.name}", merged, None, False),
                ("base_patched", lora, None, True)]
        arms += [(f"unmerged_{n}", lora, n, True) for n in adapters]

        def measure(model, ids, ad, patched):
            (install_lora_linear if patched else uninstall_lora_linear)()
            return time_generate(model, ids, args.max_tokens, ad)

        rows = []
        for pkey, ids in prompts.items():
            for _ in range(args.n_warmup):
                for arm, model, ad, patched in arms:
                    measure(model, ids, ad, patched)
            for run in range(args.n_runs):
                for arm, model, ad, patched in arms:
                    r = measure(model, ids, ad, patched)
                    r.update(arm=arm, prompt=pkey, run=run)
                    rows.append(r)
                    print(f"[{pkey} run {run}] {arm:14s} ttft={r['ttft_ms']:7.1f} ms  "
                          f"decode={r['decode_tok_s']:6.1f} tok/s", flush=True)
        # Greedy-token identity between merged and unmerged (L2 cross-check under benchmark settings).
        same = all(
            next(x["tokens"] for x in rows if x["arm"] == f"merged_{merge_src.name}" and x["prompt"] == p)
            == next(x["tokens"] for x in rows if x["arm"] == f"unmerged_{merge_src.name}" and x["prompt"] == p)
            for p in prompts
        )
        for r in rows:
            r.pop("tokens")
        result["end_to_end"] = {"rows": rows, "summary": summarize(rows),
                                "merged_unmerged_greedy_identical": same}
        install_lora_linear()
        del base, merged, lora

    if args.micro or args.micro_only:
        result["micro_linears"] = micro_linear(weights, adapters, device)
        for r in result["micro_linears"]:
            print(f"[micro b={r['batch']:2d}] {r['arm']:12s} {r['ms_per_step_linears']:.3f} ms"
                  + (f"  (+{r['overhead_pct']:.1f}%)" if "overhead_pct" in r else ""))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"bench_{device}_{stamp}.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {path}")
    if "end_to_end" in result:
        print(json.dumps(result["end_to_end"]["summary"], indent=2))


if __name__ == "__main__":
    main()
