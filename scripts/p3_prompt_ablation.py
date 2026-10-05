#!/usr/bin/env python
"""P3 prompt ablation: does a fine-tuned adapter still need the long system prompt?

The P3 local pilot trained with the full system prompt (186 tokens; 221 tokens of
fixed template per example). This re-runs the pilot with prompt_style "minimal"
(no system message) on the identical 2,400 training plays with identical
hyperparameters, then scores it on the identical 1,014-play eval set. The
full-prompt arm is NOT retrained: its committed results in results/p3_local_pilot/
are reused.

Stages (each can be rerun; training and eval resume):

    python scripts/p3_prompt_ablation.py data --src <dir with the pilot's train_pilot.jsonl, eval_pilot.jsonl>
    python scripts/p3_prompt_ablation.py tokens          # prompt/completion tokens, both styles
    python scripts/train.py --config configs/p3_prompt_ablation.yaml \
        --set data.val="\"$PLAYPARSE_PROCESSED_DIR/val.jsonl\"" [--resume latest]
    python scripts/p3_prompt_ablation.py eval --name lora_minimal --prompt-style minimal \
        --adapter runs/p3_prompt_ablation/train/checkpoints/step_0000300/adapter
    # speed control: both adapters on the first 256 eval plays, back to back
    python scripts/p3_prompt_ablation.py eval --name lora_full_timing --prompt-style full --limit 256 \
        --adapter <pilot adapter dir>
    python scripts/p3_prompt_ablation.py eval --name lora_minimal_timing --prompt-style minimal --limit 256 \
        --adapter runs/p3_prompt_ablation/train/checkpoints/step_0000300/adapter
    # training-throughput control: the full-prompt config for 30 steps, idle machine, no validation
    python scripts/train.py --config configs/p3_local_pilot.yaml --set \
        'data.train="runs/p3_prompt_ablation/data/train_pilot.jsonl"' data.val=null gen_eval_examples=0 \
        'train.output_dir="runs/p3_prompt_ablation/full_probe"' train.max_steps=30 train.epochs=null \
        train.save_every=1000 train.save_best=false train.save_final=false
    python scripts/p3_prompt_ablation.py report

`eval` wraps `python -m playparse.eval.run` (same harness, batch 16, bf16, greedy,
max_new_tokens 256, as the pilot) and polls the Metal driver's allocated memory
to record a peak, which the harness does not measure.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import statistics
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from playparse.paths import DATA_PROCESSED, WEIGHTS  # noqa: E402

RUN = REPO / "runs" / "p3_prompt_ablation"
DATA = RUN / "data"
RESULTS = REPO / "results" / "p3_prompt_ablation"
PILOT_RESULTS = REPO / "results" / "p3_local_pilot"
BUCKETS = ["fumble", "lateral", "challenge", "penalty_stands", "two_point", "interception", "td",
           "penalty_nullified", "normal"]
STYLES = ("full", "minimal")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- data


def cmd_data(args: argparse.Namespace) -> None:
    """Copy the pilot's subsets and prove they are byte-identical (sha256 + keys)."""
    manifest = json.loads((PILOT_RESULTS / "data_manifest.json").read_text())
    DATA.mkdir(parents=True, exist_ok=True)
    check = {}
    for name in ("train_pilot.jsonl", "eval_pilot.jsonl"):
        dst = DATA / name
        if not dst.exists():
            shutil.copy2(Path(args.src) / name, dst)
        want = manifest["files"][name]
        got_sha = sha256(dst)
        keys = [f"{r['game_id']}#{r['play_id']}" for r in read_jsonl(dst)]
        ok = got_sha == want["sha256"] and keys == want["keys"]
        check[name] = {"sha256": got_sha, "manifest_sha256": want["sha256"], "n": len(keys),
                       "keys_match_manifest": keys == want["keys"], "identical": ok}
        if not ok:
            sys.exit(f"{name} differs from the pilot manifest: {check[name]}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "data_check.json").write_text(json.dumps(check, indent=1) + "\n")
    print(json.dumps(check, indent=1))


# --------------------------------------------------------------------------- tokens


def _summ(xs: list[int]) -> dict:
    s = sorted(xs)
    return {"mean": round(statistics.mean(s), 1), "p50": s[len(s) // 2], "p99": s[int(0.99 * (len(s) - 1))],
            "max": s[-1], "sum": sum(s)}


def style_token_stats(recs: list[dict], tok, style: str, pad: int = 64) -> dict:
    from playparse.prompt import CHAT_DATE_STRING, build_messages
    from playparse.train.collate import encode_record

    prompt, comp, total, padded = [], [], [], []
    for r in recs:
        ex = encode_record(r, tok, prompt_style=style)
        prompt.append(ex.n_prompt)
        comp.append(ex.n_completion)
        total.append(len(ex))
        padded.append(-(-len(ex) // pad) * pad)
    # Fixed part: the rendered prompt with an empty user turn.
    msgs = build_messages(None, "", style=style)
    msgs[-1]["content"] = ""
    empty = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, date_string=CHAT_DATE_STRING)
    return {
        "n": len(recs),
        "prompt_tokens": _summ(prompt),
        "completion_tokens": _summ(comp),
        "total_tokens": _summ(total),
        f"padded_to_{pad}_tokens": _summ(padded),
        "fixed_template_tokens": len(tok(empty, add_special_tokens=False)["input_ids"]),
        "prompt_share": round(sum(prompt) / sum(total), 4),
        "completion_share": round(sum(comp) / sum(total), 4),
        "padding_overhead": round(sum(padded) / sum(total) - 1, 4),
    }


def cmd_tokens(args: argparse.Namespace) -> None:
    """Token counts for both styles: the pilot subsets and the full train split.

    The eval subset (2024, test season) is tokenized only to count lengths, as the
    pilot did; its play text is not read.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(WEIGHTS))
    out: dict = {}
    sets = {"train_pilot": read_jsonl(DATA / "train_pilot.jsonl"), "eval_pilot": read_jsonl(DATA / "eval_pilot.jsonl")}
    if args.full_train:
        sets["train_full"] = read_jsonl(Path(args.processed) / "train.jsonl")
    for name, recs in sets.items():
        out[name] = {}
        for style in STYLES:
            t0 = time.time()
            out[name][style] = style_token_stats(recs, tok, style)
            print(f"{name} {style}: {out[name][style]['total_tokens']['mean']} tokens/example "
                  f"({time.time() - t0:.0f}s)", flush=True)
        f, m = out[name]["full"], out[name]["minimal"]
        out[name]["minimal_over_full"] = {
            "total_tokens": round(m["total_tokens"]["sum"] / f["total_tokens"]["sum"], 4),
            "padded_to_64_tokens": round(m["padded_to_64_tokens"]["sum"] / f["padded_to_64_tokens"]["sum"], 4),
            "prompt_tokens": round(m["prompt_tokens"]["sum"] / f["prompt_tokens"]["sum"], 4),
        }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "token_stats.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v["minimal_over_full"] for k, v in out.items()}, indent=1))


def cmd_gpu_projection(args: argparse.Namespace) -> None:
    """Tokens one R5 epoch processes under the GPU config (micro-batch 16, each batch
    right-padded to its longest example, shuffled), for both styles.

    The minimal prompt is exactly the system prompt's tokens shorter for every
    example (tests/test_prompt_style.py), so only the full lengths are computed.
    """
    import random

    from transformers import AutoTokenizer

    from playparse.prompt import SYSTEM_PROMPT
    from playparse.train.collate import encode_record

    tok = AutoTokenizer.from_pretrained(str(WEIGHTS))
    delta = len(tok(SYSTEM_PROMPT, add_special_tokens=False)["input_ids"])
    full = [len(encode_record(r, tok)) for r in read_jsonl(Path(args.processed) / "train.jsonl")]
    out: dict = {"n": len(full), "system_prompt_tokens": delta, "micro_batch": args.micro_batch,
                 "n_shuffles": args.n_shuffles}
    for style, lens in (("full", full), ("minimal", [n - delta for n in full])):
        padded = []
        for s in range(args.n_shuffles):
            order = list(range(len(lens)))
            random.Random(s).shuffle(order)
            tot = 0
            for i in range(0, len(order), args.micro_batch):
                b = [lens[j] for j in order[i:i + args.micro_batch]]
                tot += max(b) * len(b)
            padded.append(tot)
        out[style] = {"real_tokens": sum(lens), "batch_padded_tokens_mean": statistics.mean(padded),
                      "padding_overhead": statistics.mean(padded) / sum(lens) - 1}
    out["minimal_over_full_real"] = out["minimal"]["real_tokens"] / out["full"]["real_tokens"]
    out["minimal_over_full_batch_padded"] = (out["minimal"]["batch_padded_tokens_mean"]
                                             / out["full"]["batch_padded_tokens_mean"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "gpu_projection.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


# --------------------------------------------------------------------------- eval


def _poll_mps_peak(stop: threading.Event, box: dict, every: float = 0.25) -> None:
    import torch

    while not stop.is_set():
        try:
            box["peak_driver_bytes"] = max(box.get("peak_driver_bytes", 0), torch.mps.driver_allocated_memory())
            box["peak_current_bytes"] = max(box.get("peak_current_bytes", 0), torch.mps.current_allocated_memory())
        except Exception:  # pragma: no cover - no MPS
            return
        stop.wait(every)


def cmd_eval(args: argparse.Namespace) -> None:
    from playparse.eval import run as eval_run

    out = RUN / ("extra" if args.data else "eval") / args.name
    data = args.data or str(DATA / "eval_pilot.jsonl")
    argv = ["--rung", "lora", "--adapter", args.adapter, "--data", data,
            "--out", str(out), "--dtype", "bfloat16", "--batch-size", str(args.batch_size),
            "--prompt-style", args.prompt_style]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    box: dict = {}
    stop = threading.Event()
    th = threading.Thread(target=_poll_mps_peak, args=(stop, box), daemon=True)
    th.start()
    t0 = time.time()
    try:
        eval_run.main(argv)
    finally:
        stop.set()
        th.join()
    box["wall_s"] = time.time() - t0
    box["note"] = ("peak of torch.mps driver/current allocated memory, polled every 0.25 s during this process "
                   "(model load + eval); this invocation's wall time includes model load and any resumed rows")
    prev = json.loads((out / "memory.json").read_text()) if (out / "memory.json").exists() else {}
    box["peak_driver_bytes"] = max(box.get("peak_driver_bytes", 0), prev.get("peak_driver_bytes", 0))
    box["peak_current_bytes"] = max(box.get("peak_current_bytes", 0), prev.get("peak_current_bytes", 0))
    (out / "memory.json").write_text(json.dumps(box, indent=1) + "\n")
    print(json.dumps(box))


def cmd_valdiag(args: argparse.Namespace) -> None:
    """Every val (2023) challenge and lateral play: the buckets where minimal leaned
    worse on eval. Diagnosis uses val text only; test-season text is not read."""
    recs = [r for r in read_jsonl(Path(args.processed) / "val.jsonl") if r["bucket"] in ("challenge", "lateral")]
    recs.sort(key=lambda r: (r["game_id"], r["play_id"]))
    path = DATA / "val_challenge_lateral.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    info = {"n": len(recs), "sha256": sha256(path),
            "by_bucket": {b: sum(r["bucket"] == b for r in recs) for b in ("challenge", "lateral")}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "val_diag_manifest.json").write_text(json.dumps(info, indent=1) + "\n")
    print(info)


def valdiag_report(n_boot: int, seed: int) -> dict | None:
    dirs = {s: RUN / "extra" / f"valdiag_{s}" for s in STYLES}
    if not all((d / "result.json").exists() for d in dirs.values()):
        return None
    records = read_jsonl(DATA / "val_challenge_lateral.jsonl")
    t, sc = {}, {}
    for s, d in dirs.items():
        t[s], sc[s], _ = stat_table(d / "predictions.jsonl", records)
    out: dict = {"paired_minimal_minus_full": paired(t["minimal"], t["full"], n_boot, seed), "buckets": {}}
    for b in ("challenge", "lateral"):
        idx = [i for i, r in enumerate(records) if r["bucket"] == b]
        out["buckets"][b] = {
            "n": len(idx),
            "full_exact": sum(sc["full"][i].exact for i in idx),
            "minimal_exact": sum(sc["minimal"][i].exact for i in idx),
            "minimal_right_full_wrong": sum(sc["minimal"][i].exact and not sc["full"][i].exact for i in idx),
            "full_right_minimal_wrong": sum(sc["full"][i].exact and not sc["minimal"][i].exact for i in idx),
            # Challenge plays whose text contains REVERSED: the one rule the full system
            # prompt states that the labels alone must otherwise teach.
            "reversed_n": sum("REVERSED" in records[i]["desc"] for i in idx),
            "reversed_full_exact": sum(sc["full"][i].exact for i in idx if "REVERSED" in records[i]["desc"]),
            "reversed_minimal_exact": sum(sc["minimal"][i].exact for i in idx if "REVERSED" in records[i]["desc"]),
        }
    return out


# --------------------------------------------------------------------------- report


def stat_table(pred_path: Path, records: list[dict]):
    """Rescore a predictions file into the harness's per-game stat matrices."""
    from playparse.eval.metrics import build_stat_table, score_example
    from playparse.ffscore.schema import PlayLabel

    opener = gzip.open if pred_path.suffix == ".gz" else open
    with opener(pred_path, "rt", encoding="utf-8") as f:
        preds = {json.loads(l)["key"]: json.loads(l) for l in f if l.strip()}
    keys = [f"{r['game_id']}#{r['play_id']}" for r in records]
    missing = [k for k in keys if k not in preds]
    if missing:
        raise SystemExit(f"{pred_path} lacks {len(missing)} eval plays (first {missing[0]})")
    golds = [PlayLabel.from_json(r["label"]) for r in records]
    scores = [score_example(g, preds[k]["raw"]) for g, k in zip(golds, keys)]
    table = build_stat_table([str(r["game_id"]) for r in records], [r["bucket"] for r in records], golds, scores)
    return table, scores, preds


def paired(table_a, table_b, n_boot: int, seed: int) -> dict:
    """Paired game-cluster bootstrap of A - B, overall and per bucket (same resamples)."""
    from playparse.eval.bootstrap import paired_diff_ci
    from playparse.eval.metrics import metrics_from_sums

    assert table_a.game_ids == table_b.game_ids
    out = {"OVERALL": paired_diff_ci(table_a.overall, table_b.overall, metrics_from_sums, n_boot=n_boot, seed=seed)}
    for b in BUCKETS:
        if b in table_a.by_bucket:
            out[b] = paired_diff_ci(table_a.by_bucket[b], table_b.by_bucket[b], metrics_from_sums,
                                    n_boot=n_boot, seed=seed)
    return out


def train_summary(train_dir: Path, max_step: int | None = None) -> dict:
    """Throughput, memory, and loss from a run's metrics.jsonl (optionally only steps <= max_step)."""
    rows = [json.loads(l) for l in (train_dir / "metrics.jsonl").read_text().splitlines() if l.strip()]
    if max_step is not None:
        rows = [r for r in rows if r.get("step", 0) <= max_step]
    tr = [r for r in rows if r["event"] == "train"]
    tps = [r["tokens_per_sec"] for r in tr]
    ttps = [r["target_tokens_per_sec"] for r in tr]
    step_t = [r["step_time_s"] for r in tr]
    starts = [r for r in rows if r["event"] == "start"]
    out = {
        "n_log_rows": len(tr),
        "first_time": rows[0]["time"], "last_time": rows[-1]["time"],
        "wall_h_first_to_last_log": (rows[-1]["time"] - rows[0]["time"]) / 3600,
        "n_starts": len(starts),
        "median_tokens_per_sec": statistics.median(tps),
        "p90_tokens_per_sec": sorted(tps)[int(0.9 * (len(tps) - 1))],
        "median_target_tokens_per_sec": statistics.median(ttps),
        "median_step_time_s": statistics.median(step_t),
        "tokens_seen": tr[-1]["tokens_seen"],
        "peak_mem_bytes": max(r.get("peak_mem_bytes", 0) for r in tr),
        "peak_alloc_bytes": max(r.get("peak_alloc_bytes", 0) for r in tr),
        "final_train_loss": tr[-1]["train_loss"],
        "first_train_loss": tr[0]["train_loss"],
    }
    # Sum of step times over the logged windows (log_every steps each, step_time_s is per step).
    log_every = starts[0]["config"]["log_every"] if starts else 5
    out["sum_step_time_h"] = sum(step_t) * log_every / 3600
    val = [r for r in rows if r["event"] == "eval"]
    out["val_loss"] = {r["step"]: r.get("val_loss") for r in val}
    cb = [r for r in rows if r["event"] == "val_callback"]
    out["val_gen"] = {r["step"]: {k: v for k, v in r.items() if k.startswith("val_exact_match")} for r in cb}
    return out


def cmd_report(args: argparse.Namespace) -> None:
    records = read_jsonl(DATA / "eval_pilot.jsonl")
    arms = {
        "full": PILOT_RESULTS / "eval" / "lora",
        "minimal": RUN / "eval" / "lora_minimal",
        "r0": PILOT_RESULTS / "eval" / "r0",
    }
    tables, scores, results = {}, {}, {}
    for arm, d in arms.items():
        pred = d / "predictions.jsonl"
        if not pred.exists():
            pred = d / "predictions.jsonl.gz"
        tables[arm], scores[arm], _ = stat_table(pred, records)
        results[arm] = json.loads((d / "result.json").read_text())
        # The rescored point estimate must match the harness's own result.json.
        got = tables[arm].overall[:, 3].sum() / len(records)
        assert abs(got - results[arm]["overall"]["exact_match"]["point"]) < 1e-12, arm

    summary: dict = {"n_eval": len(records), "n_boot": args.n_boot, "seed": args.seed,
                     "ci": "95% percentile, paired cluster bootstrap over games (playparse.eval.bootstrap.paired_diff_ci)",
                     "arms": {}, "paired": {}}
    for arm in arms:
        res = results[arm]
        summary["arms"][arm] = {
            "dir": str(arms[arm].relative_to(REPO)),
            "config_hash": res["meta"]["config_hash"],
            "overall": res["overall"],
            "buckets": res["buckets"],
            "latency": res.get("latency"),
            "usage": res.get("usage"),
        }
    summary["paired"]["minimal-minus-full"] = paired(tables["minimal"], tables["full"], args.n_boot, args.seed)
    summary["paired"]["minimal-minus-r0"] = paired(tables["minimal"], tables["r0"], args.n_boot, args.seed)
    summary["paired"]["full-minus-r0"] = paired(tables["full"], tables["r0"], args.n_boot, args.seed)

    # Discordant plays (the counts behind a McNemar-style comparison).
    disc: dict = {}
    for b in ["OVERALL"] + BUCKETS:
        idx = [i for i, r in enumerate(records) if b == "OVERALL" or r["bucket"] == b]
        mf = sum(1 for i in idx if scores["minimal"][i].exact and not scores["full"][i].exact)
        fm = sum(1 for i in idx if scores["full"][i].exact and not scores["minimal"][i].exact)
        same_out = sum(1 for i in idx if scores["full"][i].pred == scores["minimal"][i].pred)
        disc[b] = {"n": len(idx), "minimal_right_full_wrong": mf, "full_right_minimal_wrong": fm,
                   "identical_parsed_output": same_out}
    summary["discordant"] = disc

    # Mean input/output tokens per play at eval, from the harness's usage rows.
    tok_use = {}
    for arm in ("full", "minimal"):
        p = arms[arm] / "predictions.jsonl"
        rows = read_jsonl(p)
        tok_use[arm] = {"mean_input_tokens": statistics.mean(r["usage"]["input_tokens"] for r in rows),
                        "mean_output_tokens": statistics.mean(r["usage"]["output_tokens"] for r in rows)}
    summary["eval_tokens"] = tok_use

    # Timing control: both adapters on the same first plays, run back to back, so
    # thermal state and memory pressure are as similar as a laptop allows. The full
    # 1,014-play runs are not comparable on speed: the pilot's ran on a shared
    # machine and the minimal run slowed as the machine heated (see the write-up).
    timing = {}
    ctrl_f, ctrl_m = RUN / "eval" / "lora_full_timing", RUN / "eval" / "lora_minimal_timing"
    if (ctrl_f / "result.json").exists() and (ctrl_m / "result.json").exists():
        cf, cm = read_jsonl(ctrl_f / "predictions.jsonl"), read_jsonl(ctrl_m / "predictions.jsonl")
        f_by = {r["key"]: r for r in read_jsonl(arms["full"] / "predictions.jsonl")}
        m_by = {r["key"]: r for r in read_jsonl(arms["minimal"] / "predictions.jsonl")}

        def pps(rows):
            return len(rows) / sum(r["latency_s"] / r["batch_size"] for r in rows)

        def mem_of(d):
            return json.loads((d / "memory.json").read_text()) if (d / "memory.json").exists() else None

        timing = {
            "n_plays": len(cf),
            "full_plays_per_sec": pps(cf),
            "minimal_plays_per_sec": pps(cm),
            "full_mean_input_tokens": statistics.mean(r["usage"]["input_tokens"] for r in cf),
            "minimal_mean_input_tokens": statistics.mean(r["usage"]["input_tokens"] for r in cm),
            "full_mean_output_tokens": statistics.mean(r["usage"]["output_tokens"] for r in cf),
            "minimal_mean_output_tokens": statistics.mean(r["usage"]["output_tokens"] for r in cm),
            "full_pilot_plays_per_sec_same_plays": pps([f_by[r["key"]] for r in cf]),
            "minimal_main_run_plays_per_sec_same_plays": pps([m_by[r["key"]] for r in cm]),
            "full_rerun_identical_raw_to_pilot": sum(1 for r in cf if r["raw"] == f_by[r["key"]]["raw"]),
            "minimal_rerun_identical_raw_to_main_run": sum(1 for r in cm if r["raw"] == m_by[r["key"]]["raw"]),
            "memory_full": mem_of(ctrl_f),
            "memory_minimal": mem_of(ctrl_m),
        }
    summary["eval_timing_control"] = timing
    mem = arms["minimal"] / "memory.json"
    summary["eval_memory_minimal"] = json.loads(mem.read_text()) if mem.exists() else None

    summary["val_diag"] = valdiag_report(args.n_boot, args.seed)

    # Training.
    summary["train"] ={"minimal": train_summary(RUN / "train"), "full": train_summary(PILOT_RESULTS / "train")}
    probe = RUN / "full_probe"
    if (probe / "metrics.jsonl").exists():
        # Same examples in the same order (same seed), uncontended machine: a fair
        # throughput comparison against the minimal run's first steps.
        summary["train"]["full_probe"] = train_summary(probe)
        k = summary["train"]["full_probe"]["n_log_rows"] * 5
        summary["train"][f"minimal_first_{k}_steps"] = train_summary(RUN / "train", max_step=k)

    # Copy artifacts.
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name in ("lora_minimal", "lora_full_timing", "lora_minimal_timing"):
        src = RUN / "eval" / name
        if not (src / "result.json").exists():
            continue
        dst = RESULTS / "eval" / name
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("result.json", "memory.json"):
            if (src / f).exists():
                shutil.copy2(src / f, dst / f)
        p = src / "predictions.jsonl"
        if p.stat().st_size > 5 * 2**20:
            with open(p, "rb") as fi, gzip.open(dst / "predictions.jsonl.gz", "wb") as fo:
                shutil.copyfileobj(fi, fo)
        else:
            shutil.copy2(p, dst / "predictions.jsonl")
    for d in sorted((RUN / "extra").glob("*")) if (RUN / "extra").exists() else []:
        if (d / "result.json").exists():
            dst = RESULTS / "extra" / d.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in ("result.json", "predictions.jsonl", "memory.json"):
                if (d / f).exists():
                    shutil.copy2(d / f, dst / f)
    tdst = RESULTS / "train"
    tdst.mkdir(parents=True, exist_ok=True)
    for f in ("metrics.jsonl", "run_spec.json", "run_meta.json", "result.json"):
        shutil.copy2(RUN / "train" / f, tdst / f)
    if (RUN / "train" / "val_predictions").exists():
        shutil.copytree(RUN / "train" / "val_predictions", tdst / "val_predictions", dirs_exist_ok=True)
    final = sorted((RUN / "train" / "checkpoints").glob("step_*"))[-1] / "adapter"
    summary["adapter"] = {"path": str(final.relative_to(REPO)),
                          "adapter_model_sha256": sha256(final / "adapter_model.safetensors")}
    shutil.copy2(final / "adapter_config.json", tdst / "adapter_config.json")
    if (probe / "metrics.jsonl").exists():
        pdst = RESULTS / "full_probe"
        pdst.mkdir(parents=True, exist_ok=True)
        for f in ("metrics.jsonl", "run_spec.json"):
            shutil.copy2(probe / f, pdst / f)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")

    # Print the tables.
    def pct(ci):
        return f"{100 * ci['point']:.1f} [{100 * ci['lo']:.1f}, {100 * ci['hi']:.1f}]"

    def dpp(ci):
        return f"{100 * ci['point']:+.1f} [{100 * ci['lo']:+.1f}, {100 * ci['hi']:+.1f}]"

    print("| bucket | n | full | minimal | minimal − full | minimal − R0 | m✓f✗ / f✓m✗ |")
    print("|---|---|---|---|---|---|---|")
    for b in ["OVERALL"] + BUCKETS:
        blk = (lambda a: summary["arms"][a]["overall"] if b == "OVERALL" else summary["arms"][a]["buckets"][b])
        d1 = summary["paired"]["minimal-minus-full"][b]["exact_match"]
        d2 = summary["paired"]["minimal-minus-r0"][b]["exact_match"]
        dc = disc[b]
        print(f"| {b} | {blk('full')['n']} | {pct(blk('full')['exact_match'])} | {pct(blk('minimal')['exact_match'])} "
              f"| {dpp(d1)} | {dpp(d2)} | {dc['minimal_right_full_wrong']} / {dc['full_right_minimal_wrong']} |")
    for m in ("credit_f1", "valid_rate", "strict_valid_rate", "fp_mae"):
        d = summary["paired"]["minimal-minus-full"]["OVERALL"][m]
        f = summary["arms"]["full"]["overall"][m]
        mm = summary["arms"]["minimal"]["overall"][m]
        print(f"{m}: full {f['point']:.4f} minimal {mm['point']:.4f} diff {d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]")
    print(json.dumps({"eval_tokens": tok_use, "timing": timing, "eval_memory_minimal": summary["eval_memory_minimal"],
                      "val_diag": summary["val_diag"]}, indent=1, default=float))
    for arm, t in summary["train"].items():
        print(arm, {k: v for k, v in t.items() if k not in ("val_loss", "val_gen")})
        print("  val_loss", t["val_loss"])
        print("  val_gen", {s: round(v["val_exact_match"], 3) for s, v in t["val_gen"].items()})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--src", default=str(REPO / "runs" / "p3_local_pilot" / "data"))
    d.set_defaults(fn=cmd_data)
    t = sub.add_parser("tokens")
    t.add_argument("--processed", default=str(DATA_PROCESSED))
    t.add_argument("--no-full-train", dest="full_train", action="store_false")
    t.set_defaults(fn=cmd_tokens)
    e = sub.add_parser("eval")
    e.add_argument("--name", required=True)
    e.add_argument("--adapter", required=True)
    e.add_argument("--prompt-style", required=True, choices=STYLES)
    e.add_argument("--batch-size", type=int, default=16)
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--data", default=None, help="another JSONL (e.g. the val diagnostic); output goes to extra/")
    e.set_defaults(fn=cmd_eval)
    g = sub.add_parser("gpu_projection")
    g.add_argument("--processed", default=str(DATA_PROCESSED))
    g.add_argument("--micro-batch", type=int, default=16)
    g.add_argument("--n-shuffles", type=int, default=3)
    g.set_defaults(fn=cmd_gpu_projection)
    vd = sub.add_parser("valdiag")
    vd.add_argument("--processed", default=str(DATA_PROCESSED))
    vd.set_defaults(fn=cmd_valdiag)
    r = sub.add_parser("report")
    r.add_argument("--n-boot", type=int, default=2000)
    r.add_argument("--seed", type=int, default=0)
    r.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
