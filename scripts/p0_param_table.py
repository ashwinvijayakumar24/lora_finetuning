"""P0: trainable parameters and training memory, full fine-tuning vs LoRA.

Reads the real Llama 3.2 1B `config.json` (no weights needed), computes the
analytic table from `playparse.lora.memory`, and cross-checks every analytic
LoRA count against `count_trainable` on a meta-device model with adapters
actually injected. Writes results/p0/param_table.{json,md}.

With --measure, also runs real LoRA training steps on MPS and records memory.
Full fine-tuning is not measured: its optimizer state alone exceeds this
machine's memory (see docs/benchmarks/p0-param-memory.md).

    python scripts/p0_param_table.py [--measure] [--seq 256] [--batch 1]
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import subprocess
import time
from pathlib import Path

import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM

from playparse.lora import LoRAConfig, count_total, count_trainable, inject_lora, trainable_parameters
from playparse.lora.memory import TARGET_SETS, full_param_count, load_config, param_table
from playparse.paths import REPO_ROOT, RESULTS, WEIGHTS

GiB = 1024**3
RANKS = (4, 8, 16, 64)


def git_info() -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()

    return {"sha": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def verify_counts(cfg: dict, rows: list[dict]) -> None:
    """Every analytic count must equal the parameter count of a real (meta) model."""
    def meta_model() -> LlamaForCausalLM:
        with torch.device("meta"):
            return LlamaForCausalLM(LlamaConfig(**cfg))

    actual_total = count_total(meta_model())
    assert rows[0]["trainable"] == actual_total, (rows[0]["trainable"], actual_total)
    rows[0]["verified_count"] = actual_total
    for row in rows[1:]:
        model = inject_lora(
            meta_model(),
            LoRAConfig(r=row["r"], alpha=2 * row["r"], target_modules=list(TARGET_SETS[row["targets"]])),
        )
        actual = count_trainable(model)
        assert actual == row["trainable"], (row, actual)
        row["verified_count"] = actual


def _mps_sample(samples: dict[str, dict[str, float]], label: str) -> None:
    torch.mps.synchronize()
    samples[label] = {
        "current_gib": torch.mps.current_allocated_memory() / GiB,
        "driver_gib": torch.mps.driver_allocated_memory() / GiB,
    }


def measure_lora_mps(r: int, batch: int, seq: int, steps: int = 2) -> dict[str, object]:
    """Peak memory of LoRA training steps on the real model (bf16 base, fp32 adapters).

    MPS has no max_memory_allocated(). We sample after each phase instead.
    `driver_allocated_memory` includes the caching allocator's pool, which only
    grows until empty_cache(), so its last value is a high-water mark for the
    process; `current_allocated_memory` is live tensors at the sample point.
    """
    from transformers import AutoModelForCausalLM

    gc.collect()
    torch.mps.empty_cache()
    samples: dict[str, dict[str, float]] = {}
    _mps_sample(samples, "start")
    model = AutoModelForCausalLM.from_pretrained(WEIGHTS, dtype=torch.bfloat16).to("mps")
    model.config.use_cache = False
    _mps_sample(samples, "weights_loaded")
    inject_lora(model, LoRAConfig(r=r, alpha=2 * r, dropout=0.05))
    model.train()
    params = trainable_parameters(model)
    opt = torch.optim.AdamW(params, lr=1e-4)
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, model.config.vocab_size, (batch, seq), generator=g).to("mps")
    step_times, losses = [], []
    for step in range(steps):
        t0 = time.perf_counter()
        loss = model(input_ids=ids, labels=ids).loss
        _mps_sample(samples, f"step{step}_after_forward")
        loss.backward()
        _mps_sample(samples, f"step{step}_after_backward")
        opt.step()
        opt.zero_grad(set_to_none=True)
        _mps_sample(samples, f"step{step}_after_optimizer")
        step_times.append(time.perf_counter() - t0)
        losses.append(loss.item())
    result = {
        "method": "lora",
        "targets": "all-linear",
        "r": r,
        "batch": batch,
        "seq": seq,
        "trainable": sum(p.numel() for p in params),
        "losses": losses,
        "samples": samples,
        "peak_current_gib": max(s["current_gib"] for s in samples.values()),
        "peak_driver_gib": max(s["driver_gib"] for s in samples.values()),
        "step_seconds": step_times,
        "recommended_max_gib": torch.mps.recommended_max_memory() / GiB,
    }
    del model, opt, params, loss
    gc.collect()
    torch.mps.empty_cache()
    return result


def fmt_gib(n: float) -> str:
    return f"{n / GiB:.2f}"


def write_markdown(path: Path, payload: dict) -> None:
    meta, rows = payload["metadata"], payload["rows"]
    lines = [
        "# P0 parameter and training-memory table (Llama 3.2 1B)",
        "",
        f"Generated {meta['date']} from git `{meta['git']['sha'][:10]}`"
        f"{' (dirty)' if meta['git']['dirty'] else ''} by `scripts/p0_param_table.py`.",
        f"Total parameters: {rows[0]['trainable']:,} (tied embeddings counted once).",
        "",
        "Memory columns are GiB, excluding activations. Full FT assumes mixed-precision AdamW",
        "(bf16 weights and grads, fp32 master, m, v = 16 B/param). LoRA assumes a frozen bf16",
        "base plus fp32 adapters under AdamW (16 B per adapter param). Every count was checked",
        "against `count_trainable` on a meta-device model with adapters injected.",
        "",
        "| Method | Targets | r | Trainable | % of total | Weights | Grads | Optimizer (m+v+master) | Total |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        m = row["memory_bytes"]
        opt = m["adam_m"] + m["adam_v"] + m["master"]
        lines.append(
            f"| {row['method']} | {row['targets']} | {row['r'] or '-'} | {row['trainable']:,} "
            f"| {row['trainable_pct']:.3f} | {fmt_gib(m['weights'])} | {fmt_gib(m['grads'])} "
            f"| {fmt_gib(opt)} | {fmt_gib(m['total'])} |"
        )
    measured = payload.get("measured")
    if measured:
        lines += ["", "## Measured on MPS", ""]
        for mres in measured:
            if "error" in mres:
                lines.append(f"- {mres['method']} r={mres.get('r')}: not measured ({mres['error']})")
                continue
            lines.append(
                f"- LoRA {mres['targets']} r={mres['r']}, batch {mres['batch']} x seq {mres['seq']}: "
                f"peak live tensors {mres['peak_current_gib']:.2f} GiB, "
                f"peak driver allocation {mres['peak_driver_gib']:.2f} GiB, "
                f"step time {mres['step_seconds'][-1]:.2f} s (after warm-up)."
            )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=WEIGHTS, help="config.json or its directory")
    ap.add_argument("--out", type=Path, default=RESULTS / "p0")
    ap.add_argument("--measure", action="store_true", help="also measure LoRA r=16 steps on MPS")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=256)
    args = ap.parse_args()

    cfg = load_config(args.config)
    rows = param_table(cfg, RANKS)
    verify_counts(cfg, rows)
    keys = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "vocab_size", "tie_word_embeddings")
    payload: dict[str, object] = {
        "metadata": {
            "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "git": git_info(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "config_source": str(args.config),
            "config": {k: cfg.get(k) for k in keys},
            "total_params": full_param_count(cfg),
            "assumptions": {
                "full": "mixed-precision AdamW: bf16 weights+grads, fp32 master, fp32 m and v",
                "lora": "frozen bf16 base; fp32 adapter weights, grads, m and v; no master copy",
                "excluded": "activations, temporary buffers, allocator fragmentation",
            },
        },
        "rows": rows,
    }
    if args.measure:
        measured: list[dict] = []
        if not torch.backends.mps.is_available():
            measured.append({"method": "lora", "r": 16, "error": "MPS not available"})
        else:
            measured.append(measure_lora_mps(16, args.batch, args.seq))
        full_total = rows[0]["memory_bytes"]["total"]
        measured.append({
            "method": "full",
            "r": None,
            "error": (
                f"weights+grads+AdamW state alone need {full_total / GiB:.1f} GiB, above the "
                f"16 GiB of unified memory and MPS's recommended working set"
            ),
        })
        payload["measured"] = measured

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "param_table.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_markdown(args.out / "param_table.md", payload)
    print((args.out / "param_table.md").read_text())


if __name__ == "__main__":
    main()
