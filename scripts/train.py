#!/usr/bin/env python
"""Train a LoRA adapter on PlayParse JSONL data.

    python scripts/train.py --config configs/train_default.yaml \
        --set data.train=data/processed/train.jsonl train.output_dir=runs/r5

    # resume a killed run from its newest checkpoint
    python scripts/train.py --config runs/r5/run_spec.json --resume latest

--set overrides any field as section.key=value (value parsed as JSON when possible).
The resolved spec is written to <output_dir>/run_spec.json so every run can be
reproduced from its own directory.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playparse.train.build import (  # noqa: E402
    RunSpec,
    apply_lora,
    describe_device,
    load_base_model,
    read_jsonl,
)
from playparse.train.collate import encode_records, pad_token_id  # noqa: E402
from playparse.train.loop import resolve_device, seed_everything, train  # noqa: E402
from playparse.train.val_eval import harness_val_callback, stratified_subset  # noqa: E402


def apply_overrides(d: dict, sets: list[str]) -> dict:
    for item in sets:
        key, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = d
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return d


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=Path(__file__).resolve().parent).strip()
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML or JSON run spec (model/lora/data/train sections)")
    ap.add_argument("--set", nargs="*", action="extend", default=[], metavar="section.key=value",
                    help="may be repeated; later values win")
    ap.add_argument("--resume", default=None, help='"latest" or a checkpoint directory')
    ap.add_argument("--dry-run", action="store_true", help="build everything, encode data, then exit")
    args = ap.parse_args(argv)

    raw: dict = {}
    if args.config:
        p = Path(args.config)
        if p.suffix in (".yaml", ".yml"):
            import yaml

            raw = yaml.safe_load(p.read_text()) or {}
        else:
            raw = json.loads(p.read_text())
    spec = RunSpec.from_dict(apply_overrides(raw, args.set))
    if not spec.data.train:
        ap.error("data.train is required")

    out = Path(spec.train.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(spec.train.device)
    seed_everything(spec.train.seed)

    from transformers import AutoTokenizer

    from playparse.paths import WEIGHTS

    tok = AutoTokenizer.from_pretrained(spec.model.weights or str(WEIGHTS))
    t0 = time.time()
    train_recs = read_jsonl(spec.data.train, spec.data.limit_train)
    val_recs = read_jsonl(spec.data.val, spec.data.limit_val) if spec.data.val else []
    train_ex, train_rep = encode_records(train_recs, tok, spec.data.max_len, spec.data.mask_prompt,
                                         spec.data.on_overlength)
    val_ex, val_rep = encode_records(val_recs, tok, spec.data.max_len, spec.data.mask_prompt,
                                     spec.data.on_overlength) if val_recs else ([], None)
    print(f"encoded {train_rep.n_kept}/{train_rep.n_records} train (max {train_rep.max_tokens}, "
          f"mean {train_rep.mean_tokens:.0f} tokens, {len(train_rep.over_length)} over length) "
          f"and {len(val_ex)} val in {time.time() - t0:.1f}s", flush=True)

    meta = {"spec": spec.to_dict(), "git_sha": git_sha(), "argv": sys.argv, **describe_device(device),
            "train_report": vars(train_rep), "val_report": vars(val_rep) if val_rep else None}
    (out / "run_spec.json").write_text(json.dumps(spec.to_dict(), indent=2))
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    if args.dry_run:
        return 0

    model = load_base_model(spec.model, device)
    model, save_fn, load_fn, impl = apply_lora(model, spec.lora)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA impl={impl}, trainable params={n_train:,}", flush=True)

    cb = None
    if spec.gen_eval_examples and val_recs:
        gen_recs = stratified_subset(val_recs, spec.gen_eval_examples, spec.gen_eval_seed, spec.gen_eval_strategy)
        cb = harness_val_callback(tok, gen_recs, population=val_recs, max_new_tokens=spec.gen_max_new_tokens,
                                  batch_size=spec.train.eval_batch_size or 16, autocast=spec.train.autocast,
                                  predictions_dir=out / "val_predictions")
        print(f"generation eval: {len(gen_recs)} val plays every {spec.train.gen_every} steps, "
              f"buckets {dict(sorted(Counter(r['bucket'] for r in gen_recs).items()))}", flush=True)

    res = train(model, train_ex, val_ex or None, spec.train, pad_id=pad_token_id(tok), save_fn=save_fn,
                load_fn=load_fn, val_callback=cb, resume_from=args.resume, log_to_stdout=True)
    summary = {k: v for k, v in vars(res).items() if k != "history"}
    (out / "result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
