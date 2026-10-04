"""Assign every play exactly one bucket, so every metric can be reported per bucket.

A play can match several buckets at once (a lateral that ends in a touchdown after a
challenge). The PRD rule is that the *rarest* matching bucket wins, so rare hard
cases are never swallowed by common ones. `PRECEDENCE` is that order, fixed from the
standalone frequency of each condition over the training seasons (2015-2022), as
measured by `bucket_flag_rates` and recorded in docs/phases/P1-data.md:

    lateral (0.06%) < two_point (0.35%) < challenge (1.01%) < interception (1.19%)
      < fumble (1.52%) < penalty_stands (2.49%) < td (3.78%)
      < penalty_nullified (6.95%) < normal (83.6%)

(shares of the 294,016 training plays satisfying each condition on its own)

Bucket conditions (any play may satisfy several; precedence picks one):

* ``lateral``           nflverse ``lateral_reception``/``lateral_rush`` flag, or the
                        text says "Lateral to" / "Pass back to". Ground truth is noisy.
* ``two_point``         ``two_point_attempt == 1``, or a try wiped out by a penalty
                        (the text says "TWO-POINT CONVERSION ATTEMPT").
* ``challenge``         ``replay_or_challenge == 1`` (coach's challenge or booth review).
* ``interception``      ``interception == 1``.
* ``fumble``            ``fumble == 1``: any fumble, lost or recovered by the offense.
* ``td``                ``touchdown == 1`` (offensive or return).
* ``penalty_stands``    a penalty is in the play (``penalty == 1``, or the text says
                        "Penalty on", which also catches declined fouls) and the
                        play still counts.
* ``penalty_nullified`` the play was wiped out (`ground_truth.is_nullified`).
* ``normal``            none of the above.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from playparse.data.ground_truth import _flag, is_nullified

PRECEDENCE: tuple[str, ...] = (
    "lateral",
    "two_point",
    "challenge",
    "interception",
    "fumble",
    "penalty_stands",
    "td",
    "penalty_nullified",
    "normal",
)
BUCKETS = frozenset(PRECEDENCE)

_LATERAL_TEXT = re.compile(r"\bLateral to\b|\bPass back to\b", re.IGNORECASE)
_TWO_POINT_TEXT = re.compile(r"TWO-POINT CONVERSION ATTEMPT", re.IGNORECASE)
# nflverse sets penalty = 0 when every foul was declined, but the text still has to be
# read past ("Penalty on X, Defensive Holding, declined"), so the text counts too.
_PENALTY_TEXT = re.compile(r"\bpenalty on\b", re.IGNORECASE)


def bucket_flags(row: Mapping[str, Any]) -> dict[str, bool]:
    """Every bucket condition the play satisfies (before precedence)."""
    desc = str(row.get("desc") or "")
    nullified = is_nullified(row)
    flags = {
        "lateral": _flag(row, "lateral_reception") or _flag(row, "lateral_rush") or bool(_LATERAL_TEXT.search(desc)),
        "two_point": _flag(row, "two_point_attempt") or bool(_TWO_POINT_TEXT.search(desc)),
        "challenge": _flag(row, "replay_or_challenge"),
        "interception": _flag(row, "interception"),
        "fumble": _flag(row, "fumble"),
        "td": _flag(row, "touchdown"),
        "penalty_stands": (_flag(row, "penalty") or bool(_PENALTY_TEXT.search(desc))) and not nullified,
        "penalty_nullified": nullified,
    }
    flags["normal"] = not any(flags.values())
    return flags


def assign_bucket(row: Mapping[str, Any]) -> str:
    flags = bucket_flags(row)
    for name in PRECEDENCE:
        if flags[name]:
            return name
    raise AssertionError("unreachable: 'normal' is the fallback")


def bucket_flag_rates(rows) -> dict[str, int]:
    """Standalone count of each condition (used to justify PRECEDENCE)."""
    counts = {b: 0 for b in PRECEDENCE}
    for row in rows:
        for b, on in bucket_flags(row).items():
            counts[b] += int(on)
    return counts
