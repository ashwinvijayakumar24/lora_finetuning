"""R0 + LOS: the R0 regex parser, also given the line of scrimmage (T3b fairness arm).

T3b's adapter sees the line of scrimmage (`los`, nflverse `yrdln`); R0 never did. So
the pre-registration (PRD section 17) gives the regex the same field, with a fixed
2-hour engineering budget, developed on train and val only
(scripts/t3b_r0los_dev.py; the time log is in docs/phases/T3b.md). R0 itself is
unchanged; this module runs it and then corrects the ball carrier's yardage.

What the line of scrimmage buys a regex: R0 has to *infer* where the play started
(the end spot minus the stated gain). With the start known, the official yards are
"spot minus line of scrimmage" for whichever spot the official rules pick:

* the end of the run (normally the same as the stated gain);
* where a fumble was touched or recovered (backward only, and never past the line
  of scrimmage when the carrier was already behind it);
* the last spot of a run that continues after a botched snap the runner recovered
  himself ("FUMBLES (Aborted) at X, and recovers at Y. P to Z for N yards" counts
  from the line of scrimmage, not from Y);
* the spot of an offensive foul, or the spot named by "Rush credited to X".

Text that states the official number outright ("Officially, a rush for 3 yards")
overrides everything. Plays without a usable line of scrimmage keep R0's answer.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from playparse.eval.baselines.regex_parser import (
    _AFTER_ABORT,
    _FIRST_NAME,
    _NAME_CORE,
    _PASS_TO,
    _PREAMBLE,
    _RUN,
    _YDS,
    parse_desc,
)
from playparse.ffscore.schema import Credit, PlayLabel
from playparse.ffscore.spots import SpotError, spot_to_yardline100

_SPOT_RX = r"(?:[A-Z]{2,3} )?\d{1,2}"
_END = re.compile(rf" (?:to|at) (?P<s>{_SPOT_RX}) for {_YDS}$")
_TOUCHED = re.compile(rf"touched at (?P<s>{_SPOT_RX})")
_REC = re.compile(
    rf"(?:(?P<self>and recovers at)|(?P<word>RECOVERED|recovered) by (?P<team>[A-Z]{{2,3}})-\d{{1,2}}-{_NAME_CORE} at"
    rf"|(?P<oob>ball out of bounds at)) (?P<s>{_SPOT_RX})"
)
_FOUL = re.compile(rf"PENALTY on (?P<team>[A-Z]{{2,3}})\b.*?enforced at (?P<s>{_SPOT_RX})")
_CREDITED_TO = re.compile(rf"(?:Rush|Reception|Pass) credited to (?P<s>{_SPOT_RX})", re.IGNORECASE)
_OFFICIAL = re.compile(
    r"(?:(?:Officially,? a|charged with a|credited wit?h? a) (?P<n>-?\d+)[ -]yard (?:rush|run|reception|catch|gain)"
    r"|Officially,? a (?:rush|run|reception|catch|pass) (?:of|for) (?P<m>-?\d+) yards?)",
    re.IGNORECASE)


def _pos(spot: str, posteam: str) -> int:
    """Yards from the offense's own goal line (R0's convention)."""
    return 100 - spot_to_yardline100(spot.strip(), posteam)


def _carrier_yards(seg_end: str | None, after: str, full: str, posteam: str, los: int, td: bool) -> int | None:
    """Official yards for the primary ball carrier, or None to keep R0's."""
    m = _END.search(seg_end) if seg_end is not None else None
    if m is not None:
        end = _pos(m.group("s"), posteam)
    elif td:
        end = 100
    else:
        return None
    yards = end - los
    fum = after.find("FUMBLES")
    if fum != -1 and "Aborted" not in after[: fum + 20]:
        rest = after[fum:]
        rec = _REC.search(rest)
        touched = _TOUCHED.search(rest)
        own = rec is not None and rec.group("self")
        if touched is not None and not own and (rec is None or touched.start() < rec.start()):
            spot = _pos(touched.group("s"), posteam)  # the ball is spotted where it was first touched
        elif rec is not None:
            spot = _pos(rec.group("s"), posteam)
        else:
            spot = None
        if own:
            # he picked it up himself; if he ran on, his last spot counts
            tail = rest[rec.end():]
            if re.match(rf"^ ?{_SPOT_RX}\. \d{{1,2}}-{_NAME_CORE} (?:pass|sacked)", tail[0:0] + rest[rec.start("s"):]):
                return None  # the play went on as a pass or a sack: R0 handles that
            cont = re.search(rf"^\. \d{{1,2}}-{_NAME_CORE} [^.]*?(?:to|at) (?P<s>{_SPOT_RX}) for {_YDS}", tail)
            yards = (_pos(cont.group("s"), posteam) if cont else spot) - los
        elif spot is not None and spot <= end:
            yards = spot - los  # a backward fumble is measured to where the ball went
        elif spot is not None and end < los and not (rec is not None and rec.group("oob") and touched is None):
            yards = min(spot, los) - los  # forward bounce, but the carrier was behind the line
        lost = rec is not None and rec.group("team") is not None and rec.group("team") != posteam
    else:
        lost = False
    fm = _FOUL.search(full)
    # (after a lost fumble the foul is usually a post-possession one)
    if fm is not None and fm.group("team") == posteam and "No Play" not in full and not lost:
        foul = _pos(fm.group("s"), posteam)
        if los <= foul < los + yards:
            yards = foul - los
    cm = _CREDITED_TO.search(full)
    if cm is not None:
        yards = _pos(cm.group("s"), posteam) - los
    om = _OFFICIAL.search(full)
    if om is not None:
        yards = int(om.group("n") or om.group("m"))
    return yards


_LAT = re.compile(
    rf"(?:Lateral|Pass back) to \d{{1,2}}-(?P<l>{_NAME_CORE})(?P<mid>(?: [a-z]+)*?(?: (?:to|at) (?P<s>{_SPOT_RX}))?)"
    rf" for {_YDS}")
_HANDOFF = re.compile(rf"Handoff to \d{{1,2}}-(?P<p>{_NAME_CORE}) [^.]*?(?:to|at) (?P<s>{_SPOT_RX}) for {_YDS}")
_SELF_RUN = re.compile(rf"and recovers at {_SPOT_RX}\. \d{{1,2}}-(?P<p>{_NAME_CORE}) [^.]*?(?:to|at) "
                       rf"(?P<s>{_SPOT_RX}) for {_YDS}")


def _lateral_chain(primary: str, seg: str, after: str, full: str, posteam: str, los: int):
    """Yards on a play with laterals, the way nflverse credits them.

    Only the first carrier and the *last* lateral taker get yards (nflverse records
    one lateral): the first carrier his own leg, floored at 0; the whole play's gain
    (line of scrimmage to the final spot, fumble rules applied) goes to the passer;
    the last taker gets the rest, net of the uncredited middle legs.
    Returns (primary_yards, last_taker, last_yards, total) or None to keep R0's answer.
    """
    first = _LAT.search(after)
    if first is None:
        return None
    m0 = _END.search(seg)
    if m0 is None:
        return None
    e0 = _pos(m0.group("s"), posteam)
    head = after[: first.start()]
    if "FUMBLES" in head:
        cont = _SELF_RUN.search(head)
        if cont is None or cont.group("p") != primary:
            return None
        e0 = _pos(cont.group("s"), posteam)
    legs = [first]
    while True:
        nxt = _LAT.search(after, legs[-1].end())
        if nxt is None:
            break
        between = after[legs[-1].end(): nxt.start()]
        if "FUMBLES" in between or " pass" in between:
            return None
        legs.append(nxt)
    tail = after[legs[-1].end():]
    if " pass " in tail or "Lateral" in tail:
        return None
    ends = [e0]
    for lg in legs[:-1]:
        if lg.group("s") is None:
            return None
        ends.append(_pos(lg.group("s"), posteam))
    td = re.match(r"^,? TOUCHDOWN(?! NULLIFIED)", tail) is not None
    total = _carrier_yards(legs[-1].group(0), tail, full, posteam, los, td)
    if total is None:
        return None
    # A first leg behind the line is floored at 0 (the lateral taker gets the play)
    # unless the laterals gained nothing, when it stands as it was.
    prim = e0 - los if total == e0 - los else max(0, e0 - los)
    middle = sum(ends[i] - ends[i - 1] for i in range(1, len(ends)))
    return prim, legs[-1].group("l"), total - prim - middle, total


# Substitution notes that R0's one-day preamble list missed (they name a player who
# is not the ball carrier). Not a line-of-scrimmage rule: part of the arm's 2 hours.
_EXTRA_PREAMBLE = [
    re.compile(rf"#?\d{{1,2}}-{_NAME_CORE} (?:back )?(?:in )?at quarterback\.\s*"),
    re.compile(r"(?:\d{1,2}-[A-Z][\w'.\-]*(?: [A-Z][\w'.\-]*)?(?:, | and ))+\d{1,2}-[A-Z][\w'.\- ]*? "
               r"reported in as eligible\.?\s*"),
]


def parse_desc_los(desc: str, posteam: str | None, los: str | None) -> PlayLabel:
    for pat in _EXTRA_PREAMBLE:
        desc = pat.sub("", desc or "")
    base = parse_desc(desc, posteam)
    if base.nullified or not posteam or not los:
        return base
    try:
        L = _pos(los, posteam)
    except SpotError:
        return base
    text = desc or ""
    rev = text.rfind("the play was REVERSED.")
    if rev != -1:
        text = text[rev + len("the play was REVERSED."):]
    if "TWO-POINT CONVERSION ATTEMPT" in text:
        return base
    cut = re.search(r"\bpenalty on\b", text, flags=re.IGNORECASE)
    main = text[: cut.start()] if cut else text
    for pat in _PREAMBLE:
        main = pat.sub("", main)
    first = _FIRST_NAME.search(main)
    if first is None:
        return base
    player = first.group("p")
    rest = main[first.end():]
    credits = list(base.credits)

    def setyards(name: str, stat: str, new: int) -> int:
        """Replace (or add) name's `stat` credit; return the old value."""
        for i, c in enumerate(credits):
            if c.player == name and c.stat == stat:
                credits[i] = Credit(name, stat, new)
                return c.value
        credits.append(Credit(name, stat, new))
        return 0

    tdm = re.search(r"TOUCHDOWN(?! NULLIFIED)", rest)
    if "FUMBLES" in rest and _AFTER_ABORT.search(rest, rest.find("FUMBLES")) is not None:
        return base  # a bobbled snap followed by the real play (a pass or sack): R0 parses that
    if rest.startswith(" pass"):
        if "INTERCEPTED" in rest:
            return base
        m = _PASS_TO.match(rest)
        if m is None:
            return base
        receiver = m.group("r")
        if not any(c.player == receiver and c.stat == "rec" for c in credits):
            return base
        after = rest[m.end():]
        if _LAT.search(after):
            chain = _lateral_chain(receiver, rest[: m.end()], after, text, posteam, L)
            if chain is None:
                return base
            prim, taker, last, total = chain
            _replace_lateral_credits(credits, [receiver, player], "rec_yds")
            setyards(receiver, "rec_yds", prim)
            old = next((c.value for c in credits if c.player == taker and c.stat == "rec_yds"), 0)
            setyards(taker, "rec_yds", old + last)
            setyards(player, "pass_yds", total)
            return _finish(credits)
        y = _carrier_yards(rest[: m.end()], after, text, posteam, L,
                           tdm is not None and "FUMBLES" not in rest[: tdm.start()])
        if y is None:
            return base
        _touchback_fumble(credits, receiver, after)
        old = setyards(receiver, "rec_yds", y)
        passer_old = next((c.value for c in credits if c.player == player and c.stat == "pass_yds"), 0)
        setyards(player, "pass_yds", passer_old - old + y)
    elif rest.startswith(" FUMBLES (Aborted)") or rest.startswith(" Aborted"):
        # A botched snap the runner recovered and ran with: rush from the line of scrimmage.
        if re.search(r"\. \d{1,2}-" + _NAME_CORE + r" (?:pass|sacked)", rest):
            return base
        cont = re.search(rf"and recovers at {_SPOT_RX}\. \d{{1,2}}-(?P<p>{_NAME_CORE}) [^.]*?(?:(?:to|at) "
                         rf"(?P<s>{_SPOT_RX}) )?for {_YDS}(?P<td>, TOUCHDOWN)?", rest)
        if cont is None or cont.group("p") != player or (cont.group("s") is None and not cont.group("td")):
            return base
        if cont.group("td"):
            setyards(player, "rush_yds", 100 - L)
            return _finish(credits)
        if cont.group("nogain") or int(cont.group("yds")) <= 0:
            return base  # a botched snap that gains nothing is a 0-yard rush (R0's answer)
        # counted from the line of scrimmage, and never below 0
        setyards(player, "rush_yds", max(0, _pos(cont.group("s"), posteam) - L))
    elif rest.startswith(" sacked") or rest.startswith(" spiked"):
        return base
    else:
        m = _RUN.match(rest)
        if m is None:
            return base
        ho = _HANDOFF.search(rest, m.end())
        if ho is not None and "FUMBLES" in rest[m.end(): ho.start()]:
            # botched exchange, then a handoff: only the final runner is credited, from the line
            credits[:] = [c for c in credits if c.stat != "rush_yds"]
            setyards(ho.group("p"), "rush_yds", _pos(ho.group("s"), posteam) - L)
            return _finish(credits)
        if _LAT.search(rest[m.end():]):
            chain = _lateral_chain(player, rest[: m.end()], rest[m.end():], text, posteam, L)
            if chain is None:
                return base
            prim, taker, last, total = chain
            _replace_lateral_credits(credits, [player], "rush_yds")
            setyards(player, "rush_yds", prim)
            old = next((c.value for c in credits if c.player == taker and c.stat == "rush_yds"), 0)
            setyards(taker, "rush_yds", old + last)
            return _finish(credits)
        y = _carrier_yards(rest[: m.end()], rest[m.end():], text, posteam, L,
                           tdm is not None and "FUMBLES" not in rest[: tdm.start()])
        if y is None:
            return base
        _touchback_fumble(credits, player, rest[m.end():])
        setyards(player, "rush_yds", y)
    return _finish(credits)


def _touchback_fumble(credits: list[Credit], carrier: str, after: str) -> None:
    """A fumble out of the opponent's end zone is a touchback: possession lost."""
    if re.search(r"FUMBLES(?:(?!RECOVERED|recovered).)*?ball out of bounds in End Zone, Touchback", after) and not any(
            c.stat == "fumble_lost" for c in credits):
        credits.append(Credit(carrier, "fumble_lost", 1))


def _replace_lateral_credits(credits: list[Credit], keep: list[str], stat: str) -> None:
    """Drop R0's yardage credits for lateral takers (everyone but `keep`), in place."""
    credits[:] = [c for c in credits if not (c.stat == stat and c.player not in keep)]


def _finish(credits: list[Credit]) -> PlayLabel:
    return PlayLabel(False, tuple(c for c in credits
                                  if not (c.stat in ("pass_yds", "rush_yds", "rec_yds") and c.value == 0)))


class RegexLOSPredictor:
    """Harness adapter for R0 + LOS. Needs records with a `los` field (v2 data files)."""

    name = "regex_los"
    batch_size = 256

    def config(self) -> dict[str, Any]:
        return {"use_posteam": True, "use_los": True, "version": 1}

    def predict_batch(self, records: Sequence[dict]) -> list[str]:
        return [parse_desc_los(r["desc"], r.get("posteam"), r["los"]).to_json() for r in records]
