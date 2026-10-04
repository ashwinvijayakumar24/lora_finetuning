"""R0: a hand-written regex parser for play descriptions.

This rung answers "do you need ML at all?". It had a one-day effort budget (PRD §7),
so it handles the common shapes well and the rare ones not at all:

handled   complete / incomplete passes, runs and scrambles, sacks, interceptions,
          rushing and passing touchdowns, fumbles (lost vs recovered by the offense),
          penalties that wipe out the play ("No Play"), penalties that stand
          (penalty text is ignored), replay reversals (the post-reversal play is
          parsed), two-point attempts, simple laterals, botched snaps.
          Official yardage, not the stated gain, in two spot-based cases (needs
          posteam): offensive fouls enforced short of the end of the run, and
          fumbles that bounce backward (see `_adjust_yards`).
not       multiple fumbles, fumbles out of the end zone, fumble-recovery
          advances, handoffs/pitches after a bobbled snap, multi-word surnames
          ("H.Krieger Coble"), lateral yardage conventions.

Conventions match the ground truth (see `docs/phases/P1-eval.md`): zero-yard gains
still emit a yardage credit (a completion for no gain is pass_yds 0, rec 1,
rec_yds 0), sacks and incompletions give no credits, and a successful two-point try
gives only `two_pt` credits.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from playparse.ffscore.schema import Credit, PlayLabel

# A player as written in desc: jersey, dash, abbreviated name, optional suffix.
# Examples: 11-A.Brown, 20-Ju.Reid, 14-A.St. Brown, 15-G.Minshew II, 22-W.Gallman Jr.
_NAME_CORE = r"[A-Z][A-Za-z']*\. ?(?:St\. ?)?[A-Za-z][A-Za-z'\-]*(?: (?:Jr|Sr)\.?| (?:II|III|IV|V)(?![A-Za-z]))?"
# The lookbehind rejects team-prefixed names (RECOVERED by SF-99-D.Buckner), which are
# always defenders or penalized players, never the offensive ball carrier.
NAME = rf"(?<![\w-])\d{{1,2}}-(?P<{{g}}>{_NAME_CORE})"


def _name(group: str) -> str:
    return NAME.replace("{g}", group)


_YDS = r"(?:(?P<yds>-?\d+) yards?|(?P<nogain>no gain))"
_FIRST_NAME = re.compile(_name("p"))
_PASS_TO = re.compile(rf"^ pass[a-z ]*? to {_name('r')}[^.]*? for {_YDS}")
_RUN = re.compile(rf"^(?P<mid>[^.]*?) for {_YDS}")
_LATERAL = re.compile(rf"Lateral to {_name('l')}[^.]*? for {_YDS}")
_FUMBLER = re.compile(rf"{_name('f')} FUMBLES")
_RECOVERY = re.compile(r"(?P<word>RECOVERED|recovered) by (?P<team>[A-Z]{2,3})-")
_AFTER_ABORT = re.compile(rf"\. (?P<play>\d{{1,2}}-{_NAME_CORE} (?:pass|sacked))")
_TWO_PT_PASS = re.compile(rf"{_name('p')} pass to {_name('r')}")
_TWO_PT_RUN = re.compile(rf"{_name('p')} (?:rushes|scrambles|left|right|up)")

# Preamble sentences that name a player before the real action.
_PREAMBLE = [
    re.compile(rf"Direct snap to \d{{1,2}}-{_NAME_CORE}\.?\s*"),
    re.compile(rf"\d{{1,2}}-{_NAME_CORE} in at QB(?: for [A-Z]{{2,3}})?\.?\s*"),
    re.compile(r"\d{1,2}-[A-Z][\w'.\- ]*? reported in as eligible\.?\s*"),
]


_SPOT = r"(?:(?P<{g}t>[A-Z]{{2,3}}) )?(?P<{g}n>\d{{1,2}})"


def _spot(group: str) -> str:
    return _SPOT.format(g=group)


# "to KC 17 for 26 yards": the spot where the ball carrier was downed.
_END_SPOT = re.compile(rf" (?:to|at) {_spot('e')} for {_YDS}$")
# Where a fumble was recovered (by either team) or went out of bounds.
_RECOVERY_SPOT = re.compile(
    rf"(?:recovers at|RECOVERED by [A-Z]{{2,3}}-\d{{1,2}}-{_NAME_CORE} at|"
    rf"recovered by [A-Z]{{2,3}}-\d{{1,2}}-{_NAME_CORE} at|ball out of bounds at) {_spot('r')}"
)
# An offensive foul enforced at a spot: "PENALTY on PIT-67-B.Finney, ..., enforced at KC 41."
_SPOT_FOUL = re.compile(rf"PENALTY on (?P<team>[A-Z]{{2,3}})\b.*?enforced at {_spot('f')}")


def _yards(m: re.Match) -> int:
    return 0 if m.group("nogain") else int(m.group("yds"))


def _pos(m: re.Match, g: str, posteam: str) -> int:
    """Yards from the offense's own goal line for a spot like 'KC 17' or '50'."""
    team, num = m.group(f"{g}t"), int(m.group(f"{g}n"))
    if team is None:
        return num
    return num if team == posteam else 100 - num


def _adjust_yards(segment: str, gained: int, after: str, full: str, posteam: str | None) -> int:
    """Correct the stated gain the way the official stats do.

    The stated "for N yards" is where the carrier was downed. The official yardage
    differs in two common cases, both computable from spots in the text:
    - a fumble: the gain is measured to where the ball was recovered;
    - an offensive foul enforced at a spot short of the end of the run (holding,
      illegal block): the gain is measured to the foul's spot.
    `segment` ends with the "... to SPOT for N yards" clause; `after` is the text
    following it (up to any penalty); `full` includes the penalty text.
    """
    if not posteam:
        return gained
    em = _END_SPOT.search(segment)
    if em is not None:
        end = _pos(em, "e", posteam)
    elif after.lstrip(", ").startswith("TOUCHDOWN"):
        end = 100  # "for 62 yards, TOUCHDOWN [NULLIFIED ...]": the run ended in the end zone
    else:
        return gained
    start = end - gained
    if "FUMBLES" in after and "Aborted" not in after:
        rm = _RECOVERY_SPOT.search(after)
        if rm is not None:
            delta = _pos(rm, "r", posteam) - end
            # A fumble that moves backward is charged to the carrier; a forward bounce
            # only counts if he recovers it himself.
            if not rm.group(0).startswith("recovers at"):
                delta = min(0, delta)
            return gained + delta
    fm = _SPOT_FOUL.search(full)
    if fm is not None and fm.group("team") == posteam:
        foul = _pos(fm, "f", posteam)
        if start <= foul < end:
            return foul - start
    return gained


def _fumble(rest: str, carrier: str | None, posteam: str | None) -> tuple[bool, str | None]:
    """(lost?, fumbler) for the first fumble in `rest`."""
    idx = rest.find("FUMBLES")
    if idx == -1:
        return False, None
    m = None
    for m in _FUMBLER.finditer(rest[: idx + len("FUMBLES")]):
        pass
    fumbler = m.group("f") if m else carrier
    rec = _RECOVERY.search(rest, idx)
    if rec is None:
        return False, fumbler  # out of bounds, or "and recovers"
    if posteam:
        lost = rec.group("team") != posteam
    else:
        lost = rec.group("word") == "RECOVERED"  # desc capitalizes change-of-possession recoveries
    return lost, fumbler


def parse_desc(desc: str, posteam: str | None = None) -> PlayLabel:
    text = desc or ""

    # Replay reversal: the text after REVERSED is the play that counts.
    rev = text.rfind("the play was REVERSED.")
    if rev != -1:
        text = text[rev + len("the play was REVERSED.") :]

    if re.search(r"\bNo Play\b", text):
        return PlayLabel(True, ())

    # Penalties that stand don't change the stats; drop their text (it names players).
    cut = re.search(r"\bpenalty on\b", text, flags=re.IGNORECASE)
    main = text[: cut.start()] if cut else text

    if "TWO-POINT CONVERSION ATTEMPT" in main:
        main = main.split("DEFENSIVE TWO-POINT ATTEMPT")[0]  # a defensive return is not the offense's
        if "ATTEMPT SUCCEEDS" not in main:
            return PlayLabel(False, ())
        m = _TWO_PT_PASS.search(main)
        if m:
            return PlayLabel(False, (Credit(m.group("p"), "two_pt", 1), Credit(m.group("r"), "two_pt", 1)))
        m = _TWO_PT_RUN.search(main)
        if m:
            return PlayLabel(False, (Credit(m.group("p"), "two_pt", 1),))
        return PlayLabel(False, ())

    for pat in _PREAMBLE:
        main = pat.sub("", main)

    first = _FIRST_NAME.search(main)
    if first is None:
        return PlayLabel(False, ())
    player = first.group("p")
    rest = main[first.end() :]
    credits: list[Credit] = []
    tdm = re.search(r"TOUCHDOWN(?! NULLIFIED)", rest)
    # A touchdown after a fumble was scored on the recovery, not by the passer/rusher.
    td = tdm is not None and "FUMBLES" not in rest[: tdm.start()]

    if "FUMBLES" in rest and not rest.startswith(" pass"):
        # A bobbled snap ("3-R.Wilson to LV 25 for -6 yards. FUMBLES, and recovers")
        # followed by a real pass or sack: the later play is the one that counts.
        again = _AFTER_ABORT.search(rest, rest.find("FUMBLES"))
        if again is not None:
            return parse_desc(rest[again.start("play") :], posteam)

    if rest.startswith(" pass"):
        if "INTERCEPTED" in rest:
            return PlayLabel(False, (Credit(player, "int", 1),))
        m = _PASS_TO.match(rest)
        if m is None:  # incomplete, spike, or a shape we don't know
            return PlayLabel(False, ())
        receiver = m.group("r")
        tail = rest[m.end() :]
        yds = _adjust_yards(rest[: m.end()], _yards(m), tail, text, posteam)
        carrier = receiver
        total = yds
        credits += [Credit(receiver, "rec", 1), Credit(receiver, "rec_yds", yds)]
        for lm in _LATERAL.finditer(tail):
            ly = _yards(lm)
            credits.append(Credit(lm.group("l"), "rec_yds", ly))
            total += ly
            carrier = lm.group("l")
        credits.insert(0, Credit(player, "pass_yds", total))
        lost, fumbler = _fumble(rest, carrier, posteam)
        if lost and fumbler:
            credits.append(Credit(fumbler, "fumble_lost", 1))
        elif td:
            credits += [Credit(player, "pass_td", 1), Credit(carrier, "rec_td", 1)]
        return PlayLabel(False, tuple(credits))

    if rest.startswith(" sacked"):
        lost, fumbler = _fumble(rest, player, posteam)
        return PlayLabel(False, (Credit(fumbler, "fumble_lost", 1),) if lost and fumbler else ())

    if rest.startswith(" FUMBLES (Aborted)") or rest.startswith(" Aborted"):
        # It is scored as a 0-yard run by the player who fumbled it.
        lost, fumbler = _fumble(rest, player, posteam)
        credits.append(Credit(player, "rush_yds", 0))
        if lost and fumbler:
            credits.append(Credit(fumbler, "fumble_lost", 1))
        return PlayLabel(False, tuple(credits))

    if rest.startswith(" spiked"):
        return PlayLabel(False, ())

    # Otherwise a run (including scrambles and kneels).
    m = _RUN.match(rest)
    carrier = player
    if m is not None:
        yds = _adjust_yards(rest[: m.end()], _yards(m), rest[m.end() :], text, posteam)
        credits.append(Credit(player, "rush_yds", yds))
        for lm in _LATERAL.finditer(rest[m.end() :]):
            credits.append(Credit(lm.group("l"), "rush_yds", _yards(lm)))
            carrier = lm.group("l")
    lost, fumbler = _fumble(rest, carrier, posteam)
    if lost and fumbler:
        credits.append(Credit(fumbler, "fumble_lost", 1))
    elif td and m is not None:
        credits.append(Credit(carrier, "rush_td", 1))
    return PlayLabel(False, tuple(credits))


class RegexPredictor:
    """Harness adapter for R0. Deterministic, CPU-only, effectively free."""

    name = "regex"
    batch_size = 256

    def __init__(self, use_posteam: bool = True):
        self.use_posteam = use_posteam

    def config(self) -> dict[str, Any]:
        return {"use_posteam": self.use_posteam, "version": 1}

    def predict_batch(self, records: Sequence[dict]) -> list[str]:
        return [
            parse_desc(r["desc"], r.get("posteam") if self.use_posteam else None).to_json() for r in records
        ]
