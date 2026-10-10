"""T6: label train plays with an OpenAI teacher and write the R8 / R9 training sets.

    python scripts/t6_label_openai.py --model gpt-5.4-mini --n-plays 100000 --max-usd 150
    python scripts/t6_label_openai.py --model gpt-5.4-mini --build-only       # rewrite sets from the cache

The plays are exactly the ones the ground-truth data-size runs trained on:
`nested_sample(train.jsonl, N, seed=0)`, whose first 1k / 5k / 20k / 100k records are
the datasize_* runs' training sets. So the three curves (ground truth, R8 unfiltered
teacher, R9 filtered teacher) differ only in where the labels came from.

The teacher sees the same prompt as R4 (system prompt + 8 fixed few-shot turns) and
returns k = 3 samples per play in one call (`n=3`, input billed once). Spend is capped
by `playparse.distill.teacher.Budget` against a persistent ledger, so resuming never
re-buys labels or resets the cap. Ground truth is never shown to the teacher or the
filter; it is used only afterwards, to measure the filter's precision.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playparse.distill.reject import FilterConfig, filter_record  # noqa: E402
from playparse.distill.teacher import CACHE_FILE, Budget, TeacherBatch, TeacherCache, Usage, label_dataset  # noqa: E402
from playparse.eval.baselines.frontier_openai import make_r4_openai  # noqa: E402
from playparse.ffscore.schema import PlayLabel  # noqa: E402
from playparse.paths import DATA_PROCESSED, REPO_ROOT  # noqa: E402
from playparse.train.build import nested_sample, read_jsonl  # noqa: E402

SIZES = (1000, 5000, 20000, 100000)


class OpenAITeacher:
    """TeacherClient: k samples per play from an OpenAI model, many plays in parallel."""

    def __init__(self, model: str, k: int, concurrency: int, reasoning_effort: str):
        self.pred = make_r4_openai(model, n=k, concurrency=concurrency, reasoning_effort=reasoning_effort)
        self.failures: list[dict] = []

    def _safe_sample(self, record):
        """One play's samples. A 403 that survives retries (seen once mid-run, transient
        for the account but fatal to the whole run) becomes k empty samples: schema-invalid,
        so both R8 and R9 drop the play, and it is counted as a teacher failure."""
        import time as _time

        import openai

        for attempt in range(3):
            try:
                return self.pred.sample(record)
            except openai.PermissionDeniedError as e:
                if attempt == 2:
                    self.failures.append({"game_id": record["game_id"], "play_id": record["play_id"],
                                          "error": str(e)[:300]})
                    print(f"  teacher failure on {record['game_id']}#{record['play_id']}: {str(e)[:200]}", flush=True)
                    return [""] * self.pred.n, {"input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 0}
                _time.sleep(5 * (attempt + 1))

    def __call__(self, records, n_samples):
        assert n_samples == self.pred.n, (n_samples, self.pred.n)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.pred.concurrency) as pool:
            results = list(pool.map(self._safe_sample, list(records)))
        u = Usage()
        for _, usage in results:
            u = u + Usage(calls=1, input_tokens=usage["input_tokens"] + usage["cache_read_input_tokens"],
                          output_tokens=usage["output_tokens"], cache_read_tokens=usage["cache_read_input_tokens"],
                          cost_usd=usage.get("cost_usd", 0.0))
        return TeacherBatch([texts for texts, _ in results], u)


def build_sets(records: list[dict], cache_dir: Path, out_dir: Path, teacher_id: str, k: int) -> dict:
    cache = TeacherCache(cache_dir / CACHE_FILE, teacher_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"teacher": teacher_id, "sizes": {}}
    r8_cfg, r9_cfg = FilterConfig.r8(), FilterConfig.r9()
    rows = []
    for rec in records:
        samples = cache.samples(rec["game_id"], rec["play_id"], k)
        if not samples or len(samples) < k:
            break  # nested order: stop at the first unlabeled play
        rows.append((rec, samples[:k]))
    report["labeled_plays"] = len(rows)
    for n in SIZES:
        if n > len(rows):
            continue
        stats = {"r8": 0, "r9": 0, "r8_correct": 0, "r9_correct": 0}
        with open(out_dir / f"r8_n{n}.jsonl", "w") as f8, open(out_dir / f"r9_n{n}.jsonl", "w") as f9:
            for rec, samples in rows[:n]:
                gt = PlayLabel.from_json(rec["label"])
                for name, cfg, fh in (("r8", r8_cfg, f8), ("r9", r9_cfg, f9)):
                    d = filter_record(rec, samples, cfg)
                    if not d.keep:
                        continue
                    stats[name] += 1
                    stats[f"{name}_correct"] += int(d.label.matches(gt))  # precision only; never used to filter
                    out = {**rec, "label": d.label.to_json(), "label_source": f"teacher_{name}"}
                    fh.write(json.dumps(out, ensure_ascii=False) + "\n")
        for name in ("r8", "r9"):
            stats[f"{name}_precision"] = stats[f"{name}_correct"] / max(stats[name], 1)
        report["sizes"][n] = stats
        print(n, stats)
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-plays", type=int, default=20000)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-usd", type=float, required=False)
    ap.add_argument("--est-usd-per-play", type=float, default=0.0016)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "t6"))
    a = ap.parse_args(argv)

    train = read_jsonl(DATA_PROCESSED / "train.jsonl")
    records = nested_sample(train, max(a.n_plays, max(SIZES)), seed=0)
    out = Path(a.out) / a.model
    teacher_id = f"openai:{a.model}:k{a.k}:effort={a.reasoning_effort}:fewshot_r2"
    if not a.build_only:
        if a.max_usd is None:
            sys.exit("--max-usd is required for a labeling run")
        teacher = OpenAITeacher(a.model, a.k, a.concurrency, a.reasoning_effort)
        rep = label_dataset(records[: a.n_plays], teacher,
                            out / "cache", n_samples=a.k, chunk_size=a.chunk_size, teacher_id=teacher_id,
                            budget=Budget(max_usd=a.max_usd, est_usd_per_call=a.est_usd_per_play),
                            on_chunk=lambda r: print(f"  labeled {r.n_labeled} this run, {r.n_remaining} left, "
                                                     f"total spent ${r.usage_total.cost_usd:.2f}", flush=True))
        print(json.dumps(rep.to_dict(), default=str))
        if teacher.failures:
            fpath = out / "teacher_failures.jsonl"
            with open(fpath, "a") as f:
                f.writelines(json.dumps(x) + "\n" for x in teacher.failures)
            print(f"{len(teacher.failures)} teacher failures -> {fpath}")
    build_sets(records, out / "cache", out / "sets", teacher_id, a.k)


if __name__ == "__main__":
    main()
