"""Teacher labeling orchestration for distillation (PRD §8.6, rungs R8 and R9).

This module does not talk to any API. It drives a :class:`TeacherClient` (anything
callable with the signature below), and owns the parts that must be right no matter
which client is plugged in:

- **Resumable cache.** Every teacher sample is appended to a JSONL file keyed by
  ``(game_id, play_id, sample_idx)`` as soon as its chunk returns. A crashed or
  budget-stopped run resumes by asking only for the samples that are missing.
- **Hard budget cap**, enforced in code. Spend is tracked in a ledger next to the
  cache (so the cap holds *across* resumes, not per process). A chunk is trimmed or
  skipped if its projected cost would cross the cap.
- **Training files.** R8 (sample 0, only parse-checked) and R9 (k samples, full
  filter) are built from the *same* cache, plus nested data-size subsets for the
  distillation curve.

Plugging in a client
--------------------
A client is a callable ``client(records, n_samples) -> TeacherBatch``:

- ``records`` is a list of dicts with only ``game_id``, ``play_id``, ``posteam`` and
  ``desc`` (the teacher never sees ground truth either);
- it returns ``TeacherBatch(outputs, usage)`` where ``outputs[i][j]`` is the raw text
  of sample ``j`` for ``records[i]``, and ``usage`` is a :class:`Usage` with the
  number of API calls made and their cost. Returning a plain ``(outputs, usage)``
  tuple also works.

Samples must be drawn independently at temperature > 0 for the agreement check to
mean anything (k identical greedy samples always agree). Render the prompt with
``playparse.prompt.build_messages`` plus R4's few-shot prefix, so the teacher sees
the same prompt as the R4 rung.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from playparse.distill.reject import FilterConfig, FilterDecision, drop_stats, filter_precision, filter_samples
from playparse.ffscore.schema import PlayLabel

CACHE_FILE = "teacher_cache.jsonl"
LEDGER_FILE = "usage.json"
TEACHER_FIELDS = ("game_id", "play_id", "posteam", "desc")
# Fields copied from a dataset record into a distillation training record. `label`
# is deliberately absent: it is replaced by the teacher's label. `bucket` is kept for
# per-bucket reporting only; neither the filter nor training reads it.
RECORD_FIELDS = ("game_id", "play_id", "season", "week", "season_type", "posteam", "desc", "bucket")
DEFAULT_SIZES = (1_000, 5_000, 20_000, 100_000)

Key = tuple[str, str]


# --------------------------------------------------------------------------- protocol


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(*(getattr(self, f) + getattr(other, f) for f in self.__dataclass_fields__))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Usage":
        return cls(**{k: d.get(k, 0) for k in cls.__dataclass_fields__})


@dataclass
class TeacherBatch:
    outputs: list[list[str]]
    usage: Usage = field(default_factory=Usage)


class TeacherClient(Protocol):
    def __call__(self, records: Sequence[Mapping[str, Any]], n_samples: int) -> TeacherBatch: ...


class TeacherProtocolError(RuntimeError):
    """The client returned the wrong number of records or samples."""


def play_key(game_id: Any, play_id: Any) -> Key:
    """Normalized cache key. ``play_id`` 55, 55.0 and "55" are the same play.

    nflverse play_id is an integer, but pandas turns an int column into float64 as
    soon as one value is missing, and JSON round trips keep the float. Without this,
    a resumed run would miss every cached sample and pay for it again.
    """
    pid = play_id
    if isinstance(pid, float) and pid.is_integer():
        pid = int(pid)
    elif isinstance(pid, str):
        with contextlib.suppress(ValueError):
            f = float(pid)
            if f.is_integer():
                pid = int(f)
    return (str(game_id), str(pid))


def teacher_view(record: Mapping[str, Any]) -> dict[str, Any]:
    """The only fields the teacher is shown."""
    return {k: record[k] for k in TEACHER_FIELDS if k in record}


# --------------------------------------------------------------------------- cache + ledger


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


class TeacherCache:
    """Append-only JSONL of teacher samples.

    Each line: ``{"game_id", "play_id", "sample_idx", "output", "teacher"}``. A torn
    last line (crash mid-append) is skipped on load. ``teacher_id`` (e.g. model name
    plus prompt hash) namespaces the cache: lines written by a different teacher are
    ignored, so changing the prompt can never silently reuse stale labels.
    """

    def __init__(self, path: str | Path, teacher_id: str | None = None):
        self.path = Path(path)
        self.teacher_id = teacher_id
        self._data: dict[tuple[str, str, int], str] = {}
        self.skipped_malformed = 0
        self.skipped_other_teacher = 0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                    key = (*play_key(e["game_id"], e["play_id"]), int(e["sample_idx"]))
                    out = e["output"]
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    self.skipped_malformed += 1
                    continue
                if self.teacher_id is not None and e.get("teacher") != self.teacher_id:
                    self.skipped_other_teacher += 1
                    continue
                self._data[key] = out

    def get(self, game_id: Any, play_id: Any, sample_idx: int) -> str | None:
        return self._data.get((*play_key(game_id, play_id), sample_idx))

    def samples(self, game_id: Any, play_id: Any, k: int) -> list[str] | None:
        """Samples 0..k-1, or None if any is missing."""
        out = [self.get(game_id, play_id, i) for i in range(k)]
        return None if any(o is None for o in out) else out  # type: ignore[return-value]

    def missing(self, game_id: Any, play_id: Any, k: int) -> list[int]:
        return [i for i in range(k) if self.get(game_id, play_id, i) is None]

    def append(self, entries: Iterable[tuple[Any, Any, int, str]]) -> None:
        lines = []
        for gid, pid, idx, out in entries:
            g, p = play_key(gid, pid)
            self._data[(g, p, idx)] = out
            lines.append(json.dumps({"game_id": g, "play_id": p, "sample_idx": idx, "output": out,
                                     "teacher": self.teacher_id}, ensure_ascii=False))
        if not lines:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def __len__(self) -> int:
        return len(self._data)


def load_ledger(cache_dir: str | Path) -> Usage:
    p = Path(cache_dir) / LEDGER_FILE
    if not p.exists():
        return Usage()
    return Usage.from_dict(json.loads(p.read_text(encoding="utf-8"))["total"])


def _save_ledger(cache_dir: Path, total: Usage) -> None:
    _atomic_write_json(cache_dir / LEDGER_FILE, {"total": asdict(total)})


# --------------------------------------------------------------------------- labeling


@dataclass(frozen=True)
class Budget:
    """Hard caps on cumulative teacher spend (across all runs sharing a cache dir).

    ``max_usd`` needs a per-call cost to project a chunk before it runs: the observed
    average once some calls have been made, ``est_usd_per_call`` before that (get it
    from the pilot). A chunk that would cross a cap is trimmed; if not even one play
    fits, the run stops. Because the estimate can be low, the realized spend can
    exceed ``max_usd`` by at most one chunk; keep chunks small near the cap.
    """

    max_calls: int | None = None
    max_usd: float | None = None
    est_usd_per_call: float | None = None

    def __post_init__(self) -> None:
        if self.max_usd is not None and self.est_usd_per_call is None:
            raise ValueError("max_usd requires est_usd_per_call (from the pricing pilot)")


@dataclass
class LabelRunReport:
    n_records: int
    n_already_cached: int
    n_labeled: int
    n_remaining: int
    stopped_reason: str | None  # None (finished), "max_calls", or "max_usd"
    usage_run: Usage
    usage_total: Usage
    overran_usd: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _dedupe(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    seen, out = set(), []
    for r in records:
        k = play_key(r["game_id"], r["play_id"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def label_dataset(
    records: Sequence[Mapping[str, Any]],
    client: TeacherClient | Callable[..., Any],
    cache_dir: str | Path,
    *,
    n_samples: int = 3,
    budget: Budget | None = None,
    chunk_size: int = 50,
    teacher_id: str | None = None,
    on_chunk: Callable[[LabelRunReport], None] | None = None,
) -> LabelRunReport:
    """Collect ``n_samples`` teacher outputs per play, resuming from the cache.

    Order of writes per chunk: the ledger is updated first (money spent is recorded
    even if the process dies before the cache append), then the samples are appended.
    A crash between the two over-counts spend, which errs on the safe side of the cap.
    """
    budget = budget or Budget()
    cache_dir = Path(cache_dir)
    cache = TeacherCache(cache_dir / CACHE_FILE, teacher_id)
    total = load_ledger(cache_dir)
    run = Usage()

    records = _dedupe(records)
    todo: dict[int, list[Mapping[str, Any]]] = {}
    already = 0
    for r in records:
        miss = cache.missing(r["game_id"], r["play_id"], n_samples)
        if miss:
            todo.setdefault(len(miss), []).append(r)
        else:
            already += 1

    labeled, stopped, overran = 0, None, False

    def report() -> LabelRunReport:
        remaining = sum(len(v) for v in todo.values())
        return LabelRunReport(len(records), already, labeled, remaining, stopped, run, total, overran)

    for m in sorted(todo):
        queue = todo[m]
        while queue and stopped is None:
            chunk = queue[:chunk_size]
            if budget.max_calls is not None:
                fit = (budget.max_calls - total.calls) // m
                if fit < 1:
                    stopped = "max_calls"
                    break
                chunk = chunk[:fit]
            if budget.max_usd is not None:
                per_call = total.cost_usd / total.calls if total.calls else budget.est_usd_per_call
                remaining_usd = budget.max_usd - total.cost_usd
                fit = int(remaining_usd // (per_call * m)) if per_call > 0 else len(chunk)
                if fit < 1:
                    stopped = "max_usd"
                    break
                chunk = chunk[:fit]

            res = client([teacher_view(r) for r in chunk], m)
            batch = res if isinstance(res, TeacherBatch) else TeacherBatch(*res)
            usage = batch.usage
            if usage.calls == 0:
                # A client that does not count calls is charged the worst case.
                usage = Usage(len(chunk) * m, usage.input_tokens, usage.output_tokens,
                              usage.cache_read_tokens, usage.cost_usd)
            total = total + usage
            run = run + usage
            _save_ledger(cache_dir, total)

            outs = batch.outputs
            if len(outs) != len(chunk) or any(len(o) != m for o in outs):
                raise TeacherProtocolError(
                    f"client returned {len(outs)} records x {[len(o) for o in outs]} samples; "
                    f"expected {len(chunk)} x {m}"
                )
            entries = []
            for r, samples in zip(chunk, outs):
                miss = cache.missing(r["game_id"], r["play_id"], n_samples)
                entries.extend((r["game_id"], r["play_id"], idx, s) for idx, s in zip(miss, samples))
            cache.append(entries)
            labeled += len(chunk)
            del queue[: len(chunk)]

            if budget.max_usd is not None and total.cost_usd > budget.max_usd:
                stopped, overran = "max_usd", True
            elif budget.max_calls is not None and total.calls >= budget.max_calls and queue:
                stopped = "max_calls"
            if on_chunk:
                on_chunk(report())
        if stopped:
            break
    return report()


# --------------------------------------------------------------------------- training sets


def subset_order(records: Sequence[Mapping[str, Any]], seed: int = 0) -> list[Mapping[str, Any]]:
    """Deterministic shuffle by a seeded hash of each play's key.

    A play's position depends only on its own key and the seed, not on which other
    plays are present. So the ground-truth, R8, and R9 curves all walk plays in the
    same order, and an N-play subset is a prefix of every larger subset (nested).
    """
    def h(r: Mapping[str, Any]) -> str:
        g, p = play_key(r["game_id"], r["play_id"])
        return hashlib.sha256(f"{seed}:{g}:{p}".encode()).hexdigest()
    return sorted(records, key=h)


def _training_record(record: Mapping[str, Any], label: PlayLabel, source: str) -> dict[str, Any]:
    out = {k: record[k] for k in RECORD_FIELDS if k in record}
    out["label"] = label.to_json()
    out["label_source"] = source
    return out


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


@dataclass
class BuildReport:
    r8_path: Path
    r9_path: Path
    n_r8: int
    n_r9: int
    subsets: dict[str, dict[int, Path]]
    skipped_sizes: dict[str, list[int]]
    r8_stats: dict[str, Any]
    r9_stats: dict[str, Any]
    n_missing_samples: dict[str, int]


def build_training_sets(
    records: Sequence[Mapping[str, Any]],
    cache_dir: str | Path,
    out_dir: str | Path,
    *,
    k: int = 3,
    r8_config: FilterConfig | None = None,
    r9_config: FilterConfig | None = None,
    sizes: Sequence[int] = DEFAULT_SIZES,
    seed: int = 0,
    teacher_id: str | None = None,
) -> BuildReport:
    """Write the R8 and R9 training files and their data-size subsets.

    Outputs in ``out_dir``:

    - ``r8_teacher_raw.jsonl``: sample 0 of every play whose sample 0 parses.
    - ``r9_teacher_filtered.jsonl``: plays whose k samples pass the full filter.
    - ``r8_teacher_raw.n{N}.jsonl`` / ``r9_teacher_filtered.n{N}.jsonl``: the first N
      *training examples* in :func:`subset_order`. Sizes larger than what is
      available are skipped and listed in the report (R9 keeps fewer plays than R8,
      so it may not reach 100k from a 100k-play teacher run).
    - ``r9_decisions.jsonl``: per-play keep/drop with each check's reason.
    - ``build_report.json``: counts and drop statistics.

    Records are the dataset's JSONL rows; their ``label`` is never read.
    """
    out_dir = Path(out_dir)
    cache = TeacherCache(Path(cache_dir) / CACHE_FILE, teacher_id)
    r8_cfg = r8_config or FilterConfig.r8()
    r9_cfg = r9_config or FilterConfig.r9()

    r8_rows, r9_rows, decisions_rows = [], [], []
    r8_dec: list[FilterDecision] = []
    r9_dec: list[FilterDecision] = []
    missing = {"r8": 0, "r9": 0}
    for r in subset_order(_dedupe(records), seed):
        gid, pid = r["game_id"], r["play_id"]
        s0 = cache.get(gid, pid, 0)
        if s0 is None:
            missing["r8"] += 1
        else:
            d8 = filter_samples(r["desc"], [s0], r8_cfg)
            r8_dec.append(d8)
            if d8.keep:
                r8_rows.append(_training_record(r, d8.label, "teacher_raw"))
        samples = cache.samples(gid, pid, k)
        if samples is None:
            missing["r9"] += 1
            continue
        d9 = filter_samples(r["desc"], samples, r9_cfg)
        r9_dec.append(d9)
        g, p = play_key(gid, pid)
        decisions_rows.append({"game_id": g, "play_id": p, "keep": d9.keep, "failed": d9.failed,
                               "checks": {c: o.reason for c, o in d9.checks.items()}})
        if d9.keep:
            r9_rows.append(_training_record(r, d9.label, "teacher_filtered"))

    r8_path = out_dir / "r8_teacher_raw.jsonl"
    r9_path = out_dir / "r9_teacher_filtered.jsonl"
    _write_jsonl(r8_path, r8_rows)
    _write_jsonl(r9_path, r9_rows)
    _write_jsonl(out_dir / "r9_decisions.jsonl", decisions_rows)

    subsets: dict[str, dict[int, Path]] = {"r8": {}, "r9": {}}
    skipped: dict[str, list[int]] = {"r8": [], "r9": []}
    for name, rows, base in (("r8", r8_rows, r8_path), ("r9", r9_rows, r9_path)):
        for n in sorted(sizes):
            if n > len(rows):
                skipped[name].append(n)
                continue
            p = base.with_name(f"{base.stem}.n{n}.jsonl")
            _write_jsonl(p, rows[:n])
            subsets[name][n] = p

    report = BuildReport(r8_path, r9_path, len(r8_rows), len(r9_rows), subsets, skipped,
                         drop_stats(r8_dec), drop_stats(r9_dec), missing)
    _atomic_write_json(out_dir / "build_report.json", {
        "k": k, "seed": seed, "teacher_id": teacher_id,
        "r8_config": asdict(r8_cfg), "r9_config": asdict(r9_cfg),
        "n_r8": report.n_r8, "n_r9": report.n_r9,
        "subsets": {n: {str(s): str(p) for s, p in d.items()} for n, d in subsets.items()},
        "skipped_sizes": skipped, "missing_samples": missing,
        "r8_stats": report.r8_stats, "r9_stats": report.r9_stats,
    })
    return report


def precision_report(train_path: str | Path, ground_truth_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score a distillation training file against the dataset's ground truth.

    This is the one place ground truth meets teacher labels, and it runs *after* the
    training file is written. Run it on both the R8 and R9 files: R8's precision is the
    raw teacher accuracy, and the difference is what filtering bought.
    """
    gt, buckets = {}, {}
    for r in ground_truth_records:
        key = play_key(r["game_id"], r["play_id"])
        gt[key] = r["label"]
        if "bucket" in r:
            buckets[key] = r["bucket"]
    kept = {}
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            kept[play_key(row["game_id"], row["play_id"])] = row["label"]
    return filter_precision(kept, gt, buckets or None)
