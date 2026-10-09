"""v2 ground truth (T3b): the frozen v1 label re-expressed as field spots.

The v1 label stays the source of truth. v2 never re-derives yards; it only says
*where* each credited run of the ball started and ended, chosen so that
`schema_v2.to_v1(label_v2, los, posteam)` gives back the frozen v1 label exactly
(the round-trip invariant, checked on every play by `build_dataset_v2`).

Inputs per play: the frozen v1 label, plus four nflverse columns joined on
(game_id, play_id): `yrdln` (the line of scrimmage as text, "PHI 30" or "MID 50"),
`yardline_100`, `posteam`, `defteam`, and the lateral columns.

Rules
-----
1. **Line of scrimmage** (`los`) is `yrdln` in desc's spelling: "MID 50" becomes "50".
   If `yrdln` is missing it is rebuilt from `yardline_100`; if both are missing it
   is None (only nullified plays and some two-point tries, which have no yardage).
2. **Most yardage credits start at the line of scrimmage.** `to` = `los` advanced
   by the credited yards; `from` is left out. This covers passes, catches, runs,
   scrambles, and every official-yards adjustment (spot fouls, backward fumbles):
   the official yards *are* "line of scrimmage to the spot of the foul / the
   recovery", so `to` is that spot.
3. **A lateral leg starts where the previous carrier's credit ended.** The player
   who took the (last) lateral gets `from` = `los` advanced by the previous
   carrier's credited yards (the receiver's `receiving_yards` on a
   `lateral_reception`, the rusher's `rushing_yards` on a `lateral_rush`), and
   `to` = `from` advanced by his own yards. That is how desc writes it ("to NYJ 33
   for 8 yards. Lateral to 81-Q.Enunwa to NYJ 35 ...") and how nflverse measures
   it. The passer's `pass_yds` (the whole gain) still runs from the line of
   scrimmage. If the player is both the previous carrier and the lateral taker
   (one merged v1 credit), the merged credit starts at the line of scrimmage.
4. **Goal lines** are "<TEAM> 0": a touchdown ends at "<defteam> 0", a safety at
   "<posteam> 0".
5. A spot that would land off the field (a data error) cannot be written; that
   play is reported as a round-trip exception rather than silently changed.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from playparse.data.ground_truth import desc_name
from playparse.ffscore.schema import PlayLabel
from playparse.ffscore.schema_v2 import PlayLabelV2, from_v1, to_v1
from playparse.ffscore.spots import MIDFIELD, SpotError, advance, canonical_spot, parse_spot, yardline100_to_spot

# Raw nflverse columns the v2 builder needs beyond the v1 record.
V2_COLUMNS: tuple[str, ...] = (
    "game_id", "play_id", "posteam", "defteam", "yrdln", "yardline_100",
    "receiver_player_name", "rusher_player_name", "receiving_yards", "rushing_yards",
    "lateral_reception", "lateral_rush", "lateral_receiver_player_name", "lateral_rusher_player_name",
)


def _missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _int(v: Any) -> int | None:
    return None if _missing(v) else int(round(float(v)))


def los_from_row(row: Mapping[str, Any]) -> str | None:
    """The line of scrimmage in desc's spelling, or None if nflverse has none."""
    yrdln = row.get("yrdln")
    if not _missing(yrdln) and str(yrdln).strip():
        s = str(yrdln).strip()
        if s == "MID 50":
            return MIDFIELD
        try:
            return canonical_spot(s)
        except SpotError:
            pass
    y = _int(row.get("yardline_100"))
    pos, dft = row.get("posteam"), row.get("defteam")
    if y is not None and not _missing(pos) and not _missing(dft):
        return yardline100_to_spot(y, str(pos), str(dft))
    return None


def lateral_from_spots(row: Mapping[str, Any], label: PlayLabel, los: str | None) -> dict[tuple[str, str], str]:
    """Rule 3: the start spot of the lateral taker's credit, keyed (player, stat)."""
    out: dict[tuple[str, str], str] = {}
    if los is None:
        return out
    desc = str(row.get("desc") or "")
    pos, dft = row.get("posteam"), row.get("defteam")
    if _missing(pos) or _missing(dft):
        return out
    legs = (("lateral_reception", "lateral_receiver_player_name", "rec_yds", "receiver_player_name", "receiving_yards"),
            ("lateral_rush", "lateral_rusher_player_name", "rush_yds", "rusher_player_name", "rushing_yards"))
    stats_present = {(c.player, c.stat) for c in label.credits}
    for flag, lat_col, stat, prev_col, prev_yds_col in legs:
        if _missing(row.get(flag)) or float(row.get(flag)) != 1.0:
            continue
        lat = row.get(lat_col)
        if _missing(lat):
            continue
        lat = desc_name(str(lat), desc)
        prev = row.get(prev_col)
        prev = desc_name(str(prev), desc) if not _missing(prev) else None
        if lat == prev or (lat, stat) not in stats_present:
            continue  # merged with his own first leg (rule 3), or no credit to place
        prev_yds = _int(row.get(prev_yds_col))
        if prev_yds is None and flag == "lateral_rush":
            # a lateral after a catch, credited as a rush: the previous leg is the catch
            prev_yds = _int(row.get("receiving_yards"))
        try:
            out[(lat, stat)] = advance(los, prev_yds or 0, str(pos), str(dft))
        except SpotError:
            continue
    return out


@dataclass(frozen=True)
class V2Result:
    label_v2: PlayLabelV2 | None
    los: str | None
    round_trip: bool
    error: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


def build_label_v2(row: Mapping[str, Any], label_v1: PlayLabel) -> V2Result:
    """The v2 label for one play, plus whether it converts back to `label_v1`."""
    los = los_from_row(row)
    pos = None if _missing(row.get("posteam")) else str(row.get("posteam"))
    dft = None if _missing(row.get("defteam")) else str(row.get("defteam"))
    froms = lateral_from_spots(row, label_v1, los)
    try:
        lab2 = from_v1(label_v1, los, pos, dft, froms)
    except SpotError as e:
        if froms:  # retry with every leg at the line of scrimmage before giving up
            try:
                lab2 = from_v1(label_v1, los, pos, dft, {})
            except SpotError:
                return V2Result(None, los, False, f"spot off the field: {e}")
        else:
            return V2Result(None, los, False, f"spot off the field: {e}")
    back = to_v1(lab2, los, pos)
    ok = back.canonical() == label_v1.canonical()
    return V2Result(lab2, los, ok, None if ok else f"round trip gave {back.to_json()}")


# --------------------------------------------------------------- literal spots


def spot_in_desc(spot: str, desc: str) -> bool:
    """Does `spot` appear in `desc` as a field spot, spelled the way desc spells it?"""
    team, n = parse_spot(spot)
    if team is None:  # midfield: "to 50", "at 50", "MID 50", or "PIT 50"
        return re.search(r"\b(?:to|at|from|on|MID|[A-Z]{2,3}) 50\b", desc) is not None
    return re.search(rf"(?<![A-Za-z0-9-]){team} {n}(?!\d)", desc) is not None


def literal_kind(spot: str, desc: str) -> str:
    """'literal' if desc prints the spot, 'goal_line' for an unprinted goal line
    (a touchdown or safety, which desc signals with words), else 'computed'."""
    if spot_in_desc(spot, desc):
        return "literal"
    team, n = parse_spot(spot)
    if team is not None and n == 0:
        return "goal_line"
    return "computed"


__all__ = ["V2_COLUMNS", "los_from_row", "lateral_from_spots", "V2Result", "build_label_v2",
           "spot_in_desc", "literal_kind"]
