"""Rejection-sampling filter for teacher labels (PRD §8.6, rung R9).

Given one play's ``desc`` and k raw teacher outputs, decide whether to keep a label
for training. The checks, in PRD order:

1. ``schema``    - the output parses as a :class:`PlayLabel` (``PlayLabel.from_json``).
2. ``yardage``   - every yardage credit (pass_yds, rush_yds, rec_yds) has a value that
                   appears in ``desc`` as a gain phrase: "for 14 yards", "for 1 yard",
                   "for -3 yards", "for no gain" (0), "for a loss of 3 yards" (-3).
3. ``names``     - every credited player name appears in ``desc`` as a whole name.
4. ``agreement`` - all k samples parse and are the same label (``PlayLabel.matches``).

**Ground truth is never an input.** The filter sees only ``desc`` and the teacher's
strings; :func:`filter_record` reads ``record["desc"]`` and nothing else. In a real
distillation setting there is no ground truth, so a filter that peeked at it would
report a precision nobody could reproduce. :func:`filter_precision` is the separate,
after-the-fact measurement against ground truth that this dataset makes possible.

What "literally appears" means here
-----------------------------------
Yardage: only numbers inside a gain phrase count. Field positions ("to DAL 22") and
penalty distances ("Offensive Holding, 10 yards") are *not* gain phrases, so a
teacher that credits 22 or 10 yards is caught. Laterals produce several gain phrases
("for 5 yards. ... lateral to ... for 12 yards"); a value matching any of them passes.
The sign matters: 3 does not match "for -3 yards".

Names: matched as a whole token, not a raw substring. A plain ``name in desc`` would
accept "J.Brown" inside "A.J.Brown" and "Brown" inside "A.Brown"; both are wrong
names. A name must not be glued to a letter, apostrophe, or '.' on its left, nor to a
letter, apostrophe, or a hyphen-plus-letter on its right (so "D.Smith" does not match
inside "D.Smith-Schuster"). Jersey and team prefixes ("11-A.Brown", "DAL-55-D.Lawrence")
are fine because a digit-hyphen sits before the name.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Hashable, Mapping, Sequence

from playparse.ffscore.schema import PlayLabel, SchemaError

CHECKS: tuple[str, ...] = ("schema", "yardage", "names", "agreement")
YARDAGE_STATS = frozenset({"pass_yds", "rush_yds", "rec_yds"})

# Unicode minus and dashes sometimes show up in scraped text; treat them as '-'.
_DASHES = str.maketrans({"−": "-", "–": "-", "—": "-"})
_GAIN_RE = re.compile(
    r"\bfor\s+(?:a\s+(?P<kind>loss|gain)\s+of\s+)?(?P<num>-?\d+)\s+(?:yards?|yds?)\b",
    re.IGNORECASE,
)
_NO_GAIN_RE = re.compile(r"\bfor\s+no\s+gain\b", re.IGNORECASE)


def yardage_values(desc: str) -> set[int]:
    """Every signed yardage value stated as a gain phrase in ``desc``."""
    text = desc.translate(_DASHES)
    out: set[int] = set()
    for m in _GAIN_RE.finditer(text):
        n = int(m.group("num"))
        if (m.group("kind") or "").lower() == "loss":
            n = -abs(n)
        out.add(n)
    if _NO_GAIN_RE.search(text):
        out.add(0)
    return out


def name_in_desc(name: str, desc: str) -> bool:
    """Whole-name literal match (see module docstring for the boundary rule)."""
    if not name:
        return False
    pattern = r"(?<![A-Za-z.'])" + re.escape(name) + r"(?![A-Za-z']|-[A-Za-z])"
    return re.search(pattern, desc) is not None


@dataclass(frozen=True)
class FilterConfig:
    """Which checks are on. Defaults are the full R9 filter.

    ``schema=False`` relaxes only the *extra* samples: the label that would be kept
    (sample 0) must always parse, since an unparseable string cannot be a training
    target. With ``agreement=True`` every sample must parse anyway.
    """

    schema: bool = True
    yardage: bool = True
    names: bool = True
    agreement: bool = True

    @classmethod
    def r8(cls) -> "FilterConfig":
        """Unfiltered distillation: keep sample 0 if it parses at all."""
        return cls(schema=True, yardage=False, names=False, agreement=False)

    @classmethod
    def r9(cls) -> "FilterConfig":
        return cls()

    def enabled(self, check: str) -> bool:
        return bool(getattr(self, check))


@dataclass
class CheckOutcome:
    passed: bool | None  # None: disabled, or could not run (no parseable label)
    reason: str


@dataclass
class FilterDecision:
    keep: bool
    label: PlayLabel | None
    checks: dict[str, CheckOutcome] = field(default_factory=dict)

    @property
    def failed(self) -> list[str]:
        return [c for c in CHECKS if c in self.checks and self.checks[c].passed is False]

    @property
    def first_failure(self) -> str | None:
        f = self.failed
        return f[0] if f else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "keep": self.keep,
            "label": None if self.label is None else self.label.to_json(),
            "checks": {k: asdict(v) for k, v in self.checks.items()},
        }


_FENCE_RE = re.compile(r"^```[A-Za-z]*\s*\n?(?P<body>.*?)\n?\s*```$", re.DOTALL)


def strip_code_fence(text: str) -> str:
    """Remove one surrounding Markdown code fence (```json ... ```), if present.

    Chat models often wrap JSON in a fence even when told not to. That is formatting,
    not a wrong label; without this, the R8 "unfiltered" baseline would silently drop
    every fenced answer and stop being unfiltered. Anything else around the JSON
    (prose, a second object) is left alone and fails the schema check.
    """
    t = text.strip()
    m = _FENCE_RE.match(t)
    return m.group("body").strip() if m else t


def _parse(s: Any) -> tuple[PlayLabel | None, str]:
    if not isinstance(s, str):
        return None, f"not a string: {type(s).__name__}"
    try:
        return PlayLabel.from_json(strip_code_fence(s)), ""
    except SchemaError as e:
        return None, str(e)


def filter_samples(desc: str, samples: Sequence[str], config: FilterConfig | None = None) -> FilterDecision:
    """Decide keep/drop for one play from its ``desc`` and k teacher samples.

    All enabled checks are evaluated (not short-circuited), so drop statistics can
    say how often each check fires, not only which fired first.
    """
    cfg = config or FilterConfig()
    if not samples:
        return FilterDecision(False, None, {"schema": CheckOutcome(False, "no samples")})
    parsed = [_parse(s) for s in samples]
    labels = [p[0] for p in parsed]
    primary = labels[0]
    checks: dict[str, CheckOutcome] = {}

    # 1. schema
    must_parse = range(len(samples)) if cfg.schema else range(1)
    bad = [(i, parsed[i][1]) for i in must_parse if labels[i] is None]
    if bad:
        checks["schema"] = CheckOutcome(False, "; ".join(f"sample {i}: {err}" for i, err in bad))
    else:
        scope = f"all {len(samples)} samples" if cfg.schema else "sample 0 (schema check off for the rest)"
        checks["schema"] = CheckOutcome(True, f"{scope} parse")

    # 2. yardage
    if not cfg.yardage:
        checks["yardage"] = CheckOutcome(None, "disabled")
    elif primary is None:
        checks["yardage"] = CheckOutcome(None, "not run: sample 0 does not parse")
    else:
        allowed = yardage_values(desc)
        wrong = [c for c in primary.credits if c.stat in YARDAGE_STATS and c.value not in allowed]
        if wrong:
            checks["yardage"] = CheckOutcome(False, "values not stated in desc: " + ", ".join(
                f"{c.player} {c.stat}={c.value}" for c in wrong) + f" (desc states {sorted(allowed)})")
        else:
            checks["yardage"] = CheckOutcome(True, "all yardage values stated in desc")

    # 3. names
    if not cfg.names:
        checks["names"] = CheckOutcome(None, "disabled")
    elif primary is None:
        checks["names"] = CheckOutcome(None, "not run: sample 0 does not parse")
    else:
        missing = sorted({c.player for c in primary.credits if not name_in_desc(c.player, desc)})
        checks["names"] = (CheckOutcome(False, f"names not in desc: {missing}") if missing
                           else CheckOutcome(True, "all names in desc"))

    # 4. agreement
    if not cfg.agreement:
        checks["agreement"] = CheckOutcome(None, "disabled")
    elif len(samples) < 2:
        checks["agreement"] = CheckOutcome(True, "only one sample; agreement is vacuous")
    elif any(lab is None for lab in labels):
        n_bad = sum(lab is None for lab in labels)
        checks["agreement"] = CheckOutcome(False, f"{n_bad} of {len(samples)} samples do not parse")
    else:
        disagree = [i for i, lab in enumerate(labels[1:], start=1) if not lab.matches(primary)]
        checks["agreement"] = (
            CheckOutcome(False, f"samples {disagree} disagree with sample 0") if disagree
            else CheckOutcome(True, f"all {len(samples)} samples agree")
        )

    keep = primary is not None and all(o.passed is not False for o in checks.values())
    return FilterDecision(keep, primary.canonical() if keep else None, checks)


def filter_record(record: Mapping[str, Any], samples: Sequence[str],
                  config: FilterConfig | None = None) -> FilterDecision:
    """Filter using a dataset record. Reads only ``record["desc"]``; never the label."""
    return filter_samples(record["desc"], samples, config)


# --------------------------------------------------------------------------- statistics


def drop_stats(decisions: Sequence[FilterDecision]) -> dict[str, Any]:
    """How often each check fires.

    - ``failed_by_check``: a play counts once for every check it fails.
    - ``first_failure_by_check``: each dropped play is attributed to its first failing
      check in PRD order (these sum to ``dropped``).
    - ``sole_failure_by_check``: plays dropped *only* by that check, i.e. what turning
      that one check off would recover.
    """
    n = len(decisions)
    kept = sum(d.keep for d in decisions)
    failed, first, sole = Counter(), Counter(), Counter()
    for d in decisions:
        f = d.failed
        failed.update(f)
        if f:
            first[f[0]] += 1
        if len(f) == 1:
            sole[f[0]] += 1
    return {
        "n": n,
        "kept": kept,
        "dropped": n - kept,
        "keep_rate": kept / n if n else 0.0,
        "failed_by_check": {c: failed[c] for c in CHECKS},
        "first_failure_by_check": {c: first[c] for c in CHECKS},
        "sole_failure_by_check": {c: sole[c] for c in CHECKS},
    }


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _as_label(x: PlayLabel | str) -> PlayLabel:
    return x if isinstance(x, PlayLabel) else PlayLabel.from_json(x)


def filter_precision(
    kept: Mapping[Hashable, PlayLabel | str],
    ground_truth: Mapping[Hashable, PlayLabel | str],
    buckets: Mapping[Hashable, str] | None = None,
) -> dict[str, Any]:
    """Of the labels the filter kept, what fraction exactly match ground truth?

    ``kept`` and ``ground_truth`` map a play key (e.g. ``(game_id, play_id)``) to a
    label or its JSON. Pass the *unfiltered* teacher labels as ``kept`` to get the raw
    teacher accuracy, the baseline the filter is supposed to improve on.

    The CI is a Wilson 95% interval over plays. It ignores within-game correlation, so
    treat it as optimistic; the eval harness's game-level bootstrap is the rigorous one.
    """
    missing = [k for k in kept if k not in ground_truth]
    correct_by_bucket: Counter = Counter()
    n_by_bucket: Counter = Counter()
    n_correct = 0
    for key, lab in kept.items():
        if key not in ground_truth:
            continue
        ok = _as_label(lab).matches(_as_label(ground_truth[key]))
        n_correct += ok
        if buckets is not None:
            b = buckets.get(key, "unknown")
            n_by_bucket[b] += 1
            correct_by_bucket[b] += ok
    n_eval = len(kept) - len(missing)
    out: dict[str, Any] = {
        "n_kept": len(kept),
        "n_evaluated": n_eval,
        "n_correct": n_correct,
        "precision": n_correct / n_eval if n_eval else None,
        "precision_ci95": _wilson(n_correct, n_eval) if n_eval else None,
        "missing_ground_truth": len(missing),
    }
    if buckets is not None:
        out["per_bucket"] = {
            b: {"n": n_by_bucket[b], "correct": correct_by_bucket[b],
                "precision": correct_by_bucket[b] / n_by_bucket[b]}
            for b in sorted(n_by_bucket)
        }
    return out
