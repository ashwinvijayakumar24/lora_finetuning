"""Ground-truth builder: one nflverse play-by-play row -> one `PlayLabel`.

Every downstream number (training targets, exact match, game MAE) depends on this
module, so each rule is written out here and tested in tests/test_ground_truth.py
with real `desc` strings.

Labeling rules (v1)
-------------------
Names are nflverse's abbreviated `*_player_name` values (``A.Brown``), which are
the same abbreviations printed in `desc` (``11-A.Brown``). The builder checks that
each credited name literally appears in `desc` and reports the rate.

1. **Nullified.** ``play_type == 'no_play'`` (a penalty wiped the play out, or a
   pre-snap foul meant no play happened) -> ``nullified=True``, no credits.
   Exception: a two-point try followed by a dead-ball foul "enforced between downs"
   is filed as ``no_play`` by nflverse but still counts (see rule 2).
2. **Two-point attempt** (``two_point_attempt == 1``). Only the conversion counts:
   on ``two_point_conv_result == 'success'`` a pass credits ``two_pt`` to *both*
   the passer and the receiver (the fantasy convention, and what the official
   ``passing_2pt_conversions`` / ``receiving_2pt_conversions`` columns do), a run
   credits ``two_pt`` to the rusher. Yards on a try never count. A failed try has
   no credits. A try whose ``two_point_conv_result`` is filled in stands even when
   ``play_type == 'no_play'``; a try truly wiped out ("- No Play.") has it empty.
3. **Yardage.** Taken from the structured columns, not re-derived from text:
   ``passing_yards`` -> passer ``pass_yds``; ``receiving_yards`` -> receiver
   ``rec_yds``; ``rushing_yards`` -> rusher ``rush_yds``. nflverse leaves these
   empty on sacks (no rushing yards for the QB, and sacks do not reduce
   ``pass_yds``), on incompletions, and on interceptions. Scrambles are runs, so
   they appear as ``rushing_yards`` for the QB. Penalty yards are never included.
4. **Receptions.** ``complete_pass == 1`` -> receiver ``rec = 1``.
5. **Interceptions.** ``interception == 1`` -> passer ``int = 1``.
6. **Touchdowns.** ``pass_touchdown == 1`` -> passer ``pass_td`` and
   ``td_player_name`` ``rec_td``; ``rush_touchdown == 1`` -> ``td_player_name``
   ``rush_td``. ``td_player_name`` is the player who crossed the goal line, so on
   a lateral touchdown it is the lateral receiver. Return touchdowns and offensive
   fumble-recovery touchdowns are out of scope and get no credit.
7. **Laterals.** The player who took the last lateral gets
   ``lateral_receiving_yards`` as ``rec_yds`` (no ``rec``) or
   ``lateral_rushing_yards`` as ``rush_yds``. nflverse records only the *last*
   lateral, so plays with several laterals are knowingly wrong; every lateral play
   is flagged ``noisy`` and lands in the ``lateral`` bucket.
8. **Fumbles lost.** When ``fumble_lost == 1``, credit ``fumble_lost = 1`` to the
   offensive player whose fumble the defense recovered: the first of
   ``fumbled_1`` / ``fumbled_2`` that belongs to `posteam` and whose matching
   ``fumble_recovery_k_team`` is the other team. A player who fumbles twice on one
   play is listed only once (slot 1), so an empty ``fumbled_2`` with a filled
   ``fumble_recovery_2_team`` is read as the slot-1 player's second fumble. If no
   recovery team is recorded (ball out of the end zone, a touchback) the last
   offensive fumbler is used. If no offensive player fumbled (a defender fumbled
   back after an interception), nobody on offense is charged. Sack fumbles count,
   as in fantasy scoring. Every lost fumble is charged, including ones nflverse's
   fantasy formula leaves out (a lineman's botched snap, a fumble on a lateral or
   after a recovery); those are in the official ``fumbles_lost_total``.
9. **Zero values are dropped.** A credit whose yardage is 0 scores nothing, so
   emitting it would be a pure formatting convention the model has to guess. A
   catch for no gain is therefore just ``rec = 1``.
10. **Replay reversals.** nflverse's structured columns describe the final ruling,
    and `desc` prints the original call followed by "the play was REVERSED" and the
    final call. The label follows the final call. A reversal to an incompletion is
    *not* nullified: the play happened, it just produced no credits.
11. **Same player, same yardage stat twice** (e.g. a receiver who also takes a
    lateral back) is merged into one credit with the summed value.
12. **Name spelling follows `desc`.** If nflverse's name has a space after the
    initial ("D. Thomas") but `desc` prints "D.Thomas", the `desc` spelling is used.
    When `desc` itself spells one player two ways in a game ("Di.Johnson" and
    "Dio.Johnson"), each play keeps its own spelling; that is what the text says.

Yardage follows the official stat rules, which sometimes differ from the "for N
yards" phrase in `desc`: after an offensive foul enforced from a spot downfield the
gain is only credited to the spot of the foul, and when a fumble goes backward the
gain is measured to the recovery spot. The cross-check confirms these labels match
the official totals; build_dataset measures how often the text disagrees.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from playparse.ffscore.schema import Credit, PlayLabel

YARDAGE_STATS = frozenset({"pass_yds", "rush_yds", "rec_yds"})


@dataclass(frozen=True)
class LabelResult:
    """The label plus the diagnostics the builder found while making it."""

    label: PlayLabel
    # GSIS id per credit (same order as label.credits), for cross-check diagnostics.
    player_ids: tuple[str | None, ...] = ()
    noisy: bool = False
    # Credited names that do not literally appear in desc.
    names_missing_from_desc: tuple[str, ...] = ()
    # Credits the row implied but could not be emitted (e.g. no player name).
    dropped: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)


def _missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _flag(row: Mapping[str, Any], col: str) -> bool:
    v = row.get(col)
    return not _missing(v) and float(v) == 1.0


def _num(row: Mapping[str, Any], col: str) -> int | None:
    v = row.get(col)
    if _missing(v):
        return None
    return int(round(float(v)))


def _name(row: Mapping[str, Any], col: str) -> str | None:
    v = row.get(col)
    if _missing(v) or not str(v).strip():
        return None
    return str(v)


_INITIAL_SPACE = re.compile(r"^([A-Za-z]+)\.\s+")


def normalize_name(name: str) -> str:
    """Collapse "D. Thomas" to "D.Thomas", the form `desc` prints."""
    return _INITIAL_SPACE.sub(r"\1.", name, count=1)


def desc_name(name: str, desc: str) -> str:
    """The credited name, in the spelling that appears in `desc` when possible.

    nflverse occasionally stores an abbreviated name with a space after the initial
    ("D. Thomas", Denver 2017) while `desc` prints "88-D.Thomas". The label must use
    the name as written in the play. See docs/issues/p1-name-spacing.md.
    """
    if name in desc:
        return name
    alt = normalize_name(name)
    if alt != name and alt in desc:
        return alt
    return name


class _Builder:
    def __init__(self, row: Mapping[str, Any]):
        self.row = row
        self.desc = str(row.get("desc") or "")
        self.items: list[tuple[str, str, int, str | None]] = []  # player, stat, value, id
        self.dropped: list[str] = []

    def add(self, name_col: str, stat: str, value: int | None, id_col: str | None = None) -> None:
        if value is None:
            return
        if stat in YARDAGE_STATS and value == 0:
            return
        player = _name(self.row, name_col)
        if player is None:
            self.dropped.append(f"{stat}:{name_col} missing")
            return
        player = desc_name(player, self.desc)
        pid = _name(self.row, id_col) if id_col else None
        self.items.append((player, stat, value, pid))

    def credits(self) -> tuple[tuple[Credit, ...], tuple[str | None, ...]]:
        merged: dict[tuple[str, str], list] = {}
        out: list[list] = []
        for player, stat, value, pid in self.items:
            key = (player, stat)
            if stat in YARDAGE_STATS and key in merged:
                merged[key][2] += value
                continue
            entry = [player, stat, value, pid]
            if stat in YARDAGE_STATS:
                merged[key] = entry
            out.append(entry)
        # merging can produce a zero (e.g. +3 then -3); drop it like any zero yardage
        out = [e for e in out if not (e[1] in YARDAGE_STATS and e[2] == 0)]
        return tuple(Credit(e[0], e[1], int(e[2])) for e in out), tuple(e[3] for e in out)


def _fumble_loser(row: Mapping[str, Any]) -> tuple[str, str] | None:
    """(name column, id column) of the offensive player charged with the lost fumble."""
    posteam = row.get("posteam")
    offensive: list[tuple[int, int]] = []  # (fumble slot k, slot holding the fumbler's name)
    for k in (1, 2):
        who = k
        if _name(row, f"fumbled_{k}_player_name") is None:
            # A player who fumbles, recovers, and fumbles again appears once, in slot 1,
            # while the second recovery is still recorded in fumble_recovery_2_*.
            # See docs/issues/p1-same-player-fumbles-twice.md.
            if k == 2 and _name(row, "fumbled_1_player_name") is not None and not _missing(
                row.get("fumble_recovery_2_team")
            ):
                who = 1
            else:
                continue
        if row.get(f"fumbled_{who}_team") != posteam:
            continue
        offensive.append((k, who))
        rec_team = row.get(f"fumble_recovery_{k}_team")
        if not _missing(rec_team) and rec_team != posteam:
            return f"fumbled_{who}_player_name", f"fumbled_{who}_player_id"
    if not offensive:
        return None
    # Lost with no recorded recovery team (out of the end zone): last offensive fumbler.
    if all(_missing(row.get(f"fumble_recovery_{k}_team")) for k, _ in offensive):
        who = offensive[-1][1]
        return f"fumbled_{who}_player_name", f"fumbled_{who}_player_id"
    return None


def is_two_point_that_stands(row: Mapping[str, Any]) -> bool:
    """A two-point try whose result counts, even if nflverse filed it as `no_play`.

    A foul after a try (taunting, unsportsmanlike conduct) is "enforced between downs"
    or on the kickoff: the try still counts, but nflverse sets ``play_type =
    'no_play'``. Those rows keep ``two_point_conv_result``; a try that was truly wiped
    out ("... - No Play.") has it empty. See docs/issues/p1-two-point-dead-ball-penalty.md.
    """
    return _flag(row, "two_point_attempt") and not _missing(row.get("two_point_conv_result"))


def is_nullified(row: Mapping[str, Any]) -> bool:
    return row.get("play_type") == "no_play" and not is_two_point_that_stands(row)


def is_lateral(row: Mapping[str, Any]) -> bool:
    return _flag(row, "lateral_reception") or _flag(row, "lateral_rush")


def build_label(row: Mapping[str, Any]) -> LabelResult:
    """Convert one play-by-play row (a dict or pandas row) into its ground-truth label."""
    desc = str(row.get("desc") or "")

    if is_nullified(row):
        return LabelResult(PlayLabel(True, ()))

    b = _Builder(row)
    noisy = False
    notes: list[str] = []

    if _flag(row, "two_point_attempt"):
        if row.get("two_point_conv_result") == "success":
            if _name(row, "receiver_player_name") is not None:
                b.add("passer_player_name", "two_pt", 1, "passer_player_id")
                b.add("receiver_player_name", "two_pt", 1, "receiver_player_id")
            else:
                b.add("rusher_player_name", "two_pt", 1, "rusher_player_id")
    else:
        b.add("passer_player_name", "pass_yds", _num(row, "passing_yards"), "passer_player_id")
        if _flag(row, "complete_pass"):
            b.add("receiver_player_name", "rec", 1, "receiver_player_id")
        b.add("receiver_player_name", "rec_yds", _num(row, "receiving_yards"), "receiver_player_id")
        b.add("rusher_player_name", "rush_yds", _num(row, "rushing_yards"), "rusher_player_id")
        if _flag(row, "lateral_reception"):
            b.add("lateral_receiver_player_name", "rec_yds", _num(row, "lateral_receiving_yards"),
                  "lateral_receiver_player_id")
        if _flag(row, "lateral_rush"):
            b.add("lateral_rusher_player_name", "rush_yds", _num(row, "lateral_rushing_yards"),
                  "lateral_rusher_player_id")
        if _flag(row, "interception"):
            b.add("passer_player_name", "int", 1, "passer_player_id")
        if _flag(row, "pass_touchdown"):
            b.add("passer_player_name", "pass_td", 1, "passer_player_id")
            b.add("td_player_name", "rec_td", 1, "td_player_id")
        if _flag(row, "rush_touchdown"):
            b.add("td_player_name", "rush_td", 1, "td_player_id")
        if is_lateral(row):
            noisy = True
            notes.append("lateral: only the last lateral is recorded")

    if _flag(row, "fumble_lost"):
        loser = _fumble_loser(row)
        if loser is not None:
            b.add(loser[0], "fumble_lost", 1, loser[1])
        else:
            notes.append("fumble_lost=1 but no offensive fumbler was charged")

    credits, ids = b.credits()
    missing = tuple(sorted({c.player for c in credits if c.player not in desc}))
    return LabelResult(
        label=PlayLabel(False, credits),
        player_ids=ids,
        noisy=noisy,
        names_missing_from_desc=missing,
        dropped=tuple(b.dropped),
        notes=tuple(notes),
    )
