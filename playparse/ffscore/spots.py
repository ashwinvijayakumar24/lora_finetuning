"""Field spots as the play text writes them, and the arithmetic between them.

A *spot* is a place on the field, written the way nflverse's `desc` writes it:

    "DAL 22"   22 yards from DAL's own goal line, on DAL's half
    "50"       midfield (desc writes "to 50"; nflverse's `yrdln` column writes
               "MID 50", and a few 2015-2016 descs write "PIT 50"; all three parse)
    "DAL 0"    DAL's goal line. desc uses this form for the goal line too
               ("INTERCEPTED by 47-N.Gerry at PHI 0"). A touchdown ends at the
               defense's goal line, a safety at the offense's.

Only the canonical forms are produced: "<TEAM> N" for N in 0..49 and "50" for
midfield. The team is any 2-3 capital letters. Arithmetic needs only to know whether
a spot is on the offense's half, so every team other than `posteam` counts as the
opponent. That keeps the functions independent of team-name tables (desc and
nflverse use the same abbreviations; see docs/phases/T3b.md).

Distances use nflverse's `yardline_100` convention: yards from the spot to the
*opponent's* goal line, so 0 is a touchdown, 100 is the offense's own goal line,
and a gain makes the number smaller:

    spot_to_yardline100("PHI 30", "PHI") == 70
    spot_to_yardline100("DAL 22", "PHI") == 22
    yards_between("PHI 30", "DAL 22", "PHI") == 48
"""
from __future__ import annotations

import re

MIDFIELD = "50"

_SPOT = re.compile(r"^(?:(?P<team>[A-Z]{2,3}) )?(?P<n>\d{1,2})$")


class SpotError(ValueError):
    """A string that is not a field spot."""


def parse_spot(spot: str) -> tuple[str | None, int]:
    """Split a spot into (team or None, yards from that team's goal line).

    Midfield returns (None, 50) whatever its spelling. Raises SpotError otherwise.
    """
    if not isinstance(spot, str):
        raise SpotError(f"spot must be a string: {spot!r}")
    m = _SPOT.match(spot)
    if m is None:
        raise SpotError(f"not a field spot: {spot!r}")
    team, n = m.group("team"), int(m.group("n"))
    if n > 50 or (team is None and n != 50):
        raise SpotError(f"not a field spot: {spot!r}")
    if n == 50:
        return None, 50
    return team, n


def canonical_spot(spot: str) -> str:
    """The canonical spelling ("MID 50" and "PIT 50" become "50"; "DAL 07" becomes "DAL 7")."""
    team, n = parse_spot(spot)
    return MIDFIELD if team is None else f"{team} {n}"


def spot_to_yardline100(spot: str, posteam: str) -> int:
    """Yards from `spot` to the goal line `posteam` is attacking (0..100)."""
    if not posteam:
        raise SpotError("posteam is required to read a spot")
    team, n = parse_spot(spot)
    if team is None:
        return 50
    return 100 - n if team == posteam else n


def yardline100_to_spot(yardline_100: int, posteam: str, defteam: str) -> str:
    """Inverse of `spot_to_yardline100`: the canonical spot for a distance to the goal.

    `defteam` names the opponent's half (and its goal line, yardline_100 == 0).
    """
    if not isinstance(yardline_100, int) or isinstance(yardline_100, bool):
        raise SpotError(f"yardline_100 must be an int: {yardline_100!r}")
    if not 0 <= yardline_100 <= 100:
        raise SpotError(f"yardline_100 {yardline_100} is off the field")
    if not posteam or not defteam or posteam == defteam:
        raise SpotError(f"need two different teams, got {posteam!r} and {defteam!r}")
    if yardline_100 == 50:
        return MIDFIELD
    if yardline_100 < 50:
        return f"{defteam} {yardline_100}"
    return f"{posteam} {100 - yardline_100}"


def yards_between(from_spot: str, to_spot: str, posteam: str) -> int:
    """Yards gained by `posteam` moving the ball from `from_spot` to `to_spot`.

    Negative when the ball went backward (toward posteam's own goal line).
    """
    return spot_to_yardline100(from_spot, posteam) - spot_to_yardline100(to_spot, posteam)


def advance(from_spot: str, yards: int, posteam: str, defteam: str) -> str:
    """The spot `yards` beyond `from_spot` in posteam's direction. Raises if off the field."""
    return yardline100_to_spot(spot_to_yardline100(from_spot, posteam) - yards, posteam, defteam)


__all__ = [
    "MIDFIELD", "SpotError", "parse_spot", "canonical_spot", "spot_to_yardline100",
    "yardline100_to_spot", "yards_between", "advance",
]
