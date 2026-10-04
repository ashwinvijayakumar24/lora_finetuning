#!/usr/bin/env python
"""P2 smoke run: the real Llama 3.2 1B Instruct + PEFT LoRA through the hand-written loop.

Purpose: prove the loop works on the real model and measure what is feasible
locally (tokens/sec, peak memory) before GPU time. The data are template plays
(playparse.train.synthetic), so the exact-match numbers say nothing about the real
task; they only show the loop teaches the model the output format.

    python scripts/p2_smoke_train.py                       # 40 steps, writes results/p2/smoke.json
    python scripts/p2_smoke_train.py --steps 10 --out /tmp/x.json --no-gen
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from playparse.paths import WEIGHTS  # noqa: E402
from playparse.train.build import LoRASpec, ModelSpec, apply_lora, exact_match_callback, load_base_model  # noqa: E402
from playparse.train.collate import encode_records, pad_token_id  # noqa: E402
from playparse.train.loop import TrainConfig, release_cached_memory, resolve_device, train  # noqa: E402
from playparse.train.synthetic import synth_records  # noqa: E402


def _sh(*cmd: str) -> str | None:
    try:
        return subprocess.check_output(list(cmd), text=True, cwd=REPO, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def hardware() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__}
    if sys.platform == "darwin":
        info["chip"] = _sh("sysctl", "-n", "machdep.cpu.brand_string")
        mem = _sh("sysctl", "-n", "hw.memsize")
        info["ram_gib"] = round(int(mem) / 2**30, 1) if mem else None
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    import transformers

    info["transformers"] = transformers.__version__
    try:
        import peft

        info["peft"] = peft.__version__
    except ImportError:
        pass
    return info


def run_smoke(steps: int = 40, n_train: int = 64, n_val: int = 16, micro: int = 1, accum: int = 8,
              lr: float = 2e-4, device: str = "auto", autocast: str = "auto", gen: bool = True,
              output_dir: str | None = None, gen_n: int = 16, grad_ckpt: bool = False,
              lora_dropout: float = 0.05, mps_mem_fraction: float | None = 0.75,
              pad_to_multiple_of: int | None = 64) -> dict:
    from transformers import AutoTokenizer

    dev = resolve_device(device)
    if dev.type == "mps" and mps_mem_fraction:
        # Fail with an OOM error instead of pushing a 16 GB machine into swap.
        torch.mps.set_per_process_memory_fraction(mps_mem_fraction)
    tok = AutoTokenizer.from_pretrained(str(WEIGHTS))
    train_recs, val_recs = synth_records(n_train, seed=0), synth_records(n_val, seed=1)
    train_ex, rep = encode_records(train_recs, tok, max_len=512)
    val_ex, _ = encode_records(val_recs, tok, max_len=512)

    t_load = time.perf_counter()
    model_spec = ModelSpec(weights=str(WEIGHTS), dtype="auto", gradient_checkpointing=grad_ckpt)
    model = load_base_model(model_spec, dev)
    lora = LoRASpec(impl="peft", r=16, alpha=32, dropout=lora_dropout)
    model, save_fn, load_fn, impl = apply_lora(model, lora)
    load_s = time.perf_counter() - t_load
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    cb = exact_match_callback(tok, val_recs[:gen_n], max_new_tokens=96, batch_size=16, autocast=autocast) if gen else None
    before = {}
    if cb is not None:
        t = time.perf_counter()
        before = cb(model.eval(), 0)
        before["gen_time_s"] = time.perf_counter() - t
        model.train()
        release_cached_memory(dev)

    tmp = output_dir or tempfile.mkdtemp(prefix="p2_smoke_")
    cfg = TrainConfig(output_dir=tmp, seed=0, lr=lr, max_steps=steps, warmup_steps=max(1, steps // 10),
                      min_lr_ratio=0.1, micro_batch_size=micro, grad_accum_steps=accum, log_every=1,
                      eval_every=max(1, steps // 4), eval_batch_size=4, save_every=None, save_final=True,
                      device=str(dev), autocast=autocast, pad_to_multiple_of=pad_to_multiple_of)
    t_train = time.perf_counter()
    res = train(model, train_ex, val_ex, cfg, pad_id=pad_token_id(tok), save_fn=save_fn, load_fn=load_fn,
                log_to_stdout=True)
    train_s = time.perf_counter() - t_train

    after = {}
    if cb is not None:
        t = time.perf_counter()
        after = cb(model.eval(), steps)
        after["gen_time_s"] = time.perf_counter() - t

    tr = [r for r in res.history if r["event"] == "train"]
    ev = [r for r in res.history if r["event"] == "eval"]
    steady = tr[2:] if len(tr) > 4 else tr  # skip warm-up steps (kernel compilation, allocator growth)
    ckpt = Path(res.last_checkpoint) if res.last_checkpoint else None
    ckpt_bytes = sum(f.stat().st_size for f in (ckpt / "adapter").rglob("*") if f.is_file()) if ckpt else None
    return {
        "git_sha": _sh("git", "rev-parse", "HEAD"),
        "date": time.strftime("%Y-%m-%d"),
        "hardware": hardware(),
        "settings": {
            "device": str(dev), "autocast": autocast, "gradient_checkpointing": grad_ckpt, "pad_to_multiple_of": pad_to_multiple_of,
            "mps_mem_fraction": mps_mem_fraction if dev.type == "mps" else None, "base_dtype": str(next(model.parameters()).dtype),
            "lora": {"impl": impl, "r": lora.r, "alpha": lora.alpha, "dropout": lora.dropout, "targets": lora.targets},
            "steps": steps, "micro_batch_size": micro, "grad_accum_steps": accum,
            "effective_batch": micro * accum, "lr": lr, "warmup_steps": cfg.warmup_steps, "min_lr_ratio": 0.1,
            "n_train": n_train, "n_val": n_val, "data": "synthetic template plays (playparse.train.synthetic)",
        },
        "data": {"max_tokens": rep.max_tokens, "mean_tokens": round(rep.mean_tokens, 1),
                 "mean_completion_tokens": round(sum(e.n_completion for e in train_ex) / len(train_ex), 1)},
        "params": {"trainable": n_trainable, "total": n_total, "trainable_pct": round(100 * n_trainable / n_total, 3)},
        "timing": {"model_load_s": round(load_s, 1), "train_wall_s": round(train_s, 1),
                   "median_step_s": round(statistics.median(r["step_time_s"] for r in steady), 3),
                   "median_tokens_per_sec": round(statistics.median(r["tokens_per_sec"] for r in steady), 1),
                   "median_target_tokens_per_sec": round(statistics.median(r["target_tokens_per_sec"] for r in steady), 1)},
        "memory": {"peak_bytes": tr[-1]["peak_mem_bytes"], "peak_gib": round(tr[-1]["peak_mem_bytes"] / 2**30, 2),
                   "kind": tr[-1]["mem_kind"],
                   "peak_alloc_gib": round((tr[-1].get("peak_alloc_bytes") or 0) / 2**30, 2)},
        "loss": {
            "train": [round(r["train_loss"], 4) for r in tr],
            "val": {r["step"]: round(r["val_loss"], 4) for r in ev},
            "first": round(tr[0]["train_loss"], 4), "last": round(tr[-1]["train_loss"], 4),
        },
        "gen_eval": {"before": before, "after": after, "n": gen_n if gen else 0},
        "adapter_checkpoint_bytes": ckpt_bytes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--micro", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--mps-mem-fraction", type=float, default=0.75)
    ap.add_argument("--pad-to-multiple-of", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--autocast", default="auto")
    ap.add_argument("--no-gen", action="store_true")
    ap.add_argument("--out", default=str(REPO / "results" / "p2" / "smoke.json"))
    args = ap.parse_args()
    result = run_smoke(steps=args.steps, micro=args.micro, accum=args.accum, lr=args.lr, device=args.device,
                       autocast=args.autocast, gen=not args.no_gen, grad_ckpt=args.grad_ckpt,
                       lora_dropout=args.lora_dropout, mps_mem_fraction=args.mps_mem_fraction,
                       pad_to_multiple_of=args.pad_to_multiple_of or None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({k: result[k] for k in ("timing", "memory", "gen_eval")}, indent=2, default=str))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
