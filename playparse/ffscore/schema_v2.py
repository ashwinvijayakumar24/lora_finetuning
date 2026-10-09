"""Output contract v2 (T3b): yardage credits carry field spots, code does the arithmetic.

v1 (`playparse.ffscore.schema`) asks the model for a number of yards. On most plays
that number is printed in the text ("for 14 yards"), but on fumbles, spot fouls and
laterals the official yards come from field spots the text names and never
subtracts (docs/issues/p1-official-yards-vs-text.md). v2 asks the model only to
*read*: each yardage credit names the spot where the credited run of the ball
ended (`to`) and, when it did not start at the line of scrimmage, where it started
(`from`). `to_v1` then computes the yards with `playparse.ffscore.spots`, so every
v2 output is scored as the v1 label it converts to, by the unchanged v1 harness.

Wire format (canonical, compact, used as the training completion)::

    {"nullified":false,"credits":[
      {"player":"J.Hurts","stat":"pass_yds","to":"DAL 22"},
      {"player":"A.Brown","stat":"rec","value":1},
      {"player":"A.Brown","stat":"rec_yds","to":"DAL 22"},
      {"player":"Q.Enunwa","stat":"rec_yds","from":"NYJ 33","to":"NYJ 21"}]}

* Yardage stats (pass_yds, rush_yds, rec_yds) have exactly {player, stat, to} or
  {player, stat, from, to}. `from` defaults to the line of scrimmage (the `los`
  input field) and is written only when it differs from it.
* Every other stat keeps v1's {player, stat, value}.
* Credits are sorted like v1 (stat vocabulary order, then player), then by spots.

Conversion rules (`to_v1`): yards = yards_between(from or los, to, posteam).
Credits for the same (player, yardage stat) are summed, as v1 rule 11 does, and a
yardage credit worth 0 is dropped (v1 rule 9). So a label that says a player went
from the line of scrimmage back to it simply has no yardage credit.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from playparse.ffscore.schema import _STAT_INDEX, STAT_VOCAB, Credit, PlayLabel, SchemaError
from playparse.ffscore.spots import SpotError, canonical_spot, spot_to_yardline100, yards_between

YARDAGE_STATS = frozenset({"pass_yds", "rush_yds", "rec_yds"})
_YARD_KEYS = ({"player", "stat", "to"}, {"player", "stat", "from", "to"})
_COUNT_KEYS = {"player", "stat", "value"}


@dataclass(frozen=True)
class CreditV2:
    """One credit. Yardage stats set `to` (and maybe `from_`); the others set `value`."""

    player: str
    stat: str
    value: int | None = None
    to: str | None = None
    from_: str | None = None

    @property
    def is_yardage(self) -> bool:
        return self.stat in YARDAGE_STATS

    def sort_key(self) -> tuple:
        return (_STAT_INDEX[self.stat], self.player, self.value if self.value is not None else 0,
                self.from_ or "", self.to or "")

    def to_obj(self) -> dict:
        if self.is_yardage:
            obj = {"player": self.player, "stat": self.stat}
            if self.from_ is not None:
                obj["from"] = self.from_
            obj["to"] = self.to
            return obj
        return {"player": self.player, "stat": self.stat, "value": self.value}


@dataclass(frozen=True)
class PlayLabelV2:
    nullified: bool
    credits: tuple[CreditV2, ...] = field(default_factory=tuple)

    def canonical(self) -> "PlayLabelV2":
        return PlayLabelV2(self.nullified, tuple(sorted(self.credits, key=CreditV2.sort_key)))

    def to_json(self) -> str:
        """Compact, canonical serialization used as the v2 training completion."""
        c = self.canonical()
        return json.dumps({"nullified": c.nullified, "credits": [x.to_obj() for x in c.credits]},
                          separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "PlayLabelV2":
        """Parse and validate (as strictly as v1's `PlayLabel.from_json`).

        Spots are canonicalized ("MID 50" -> "50"). Raises SchemaError on any deviation.
        """
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, TypeError) as e:
            raise SchemaError(f"not valid JSON: {e}") from e
        return cls.from_obj(obj)

    @classmethod
    def from_obj(cls, obj) -> "PlayLabelV2":
        if not isinstance(obj, dict) or set(obj) != {"nullified", "credits"}:
            raise SchemaError("top level must be an object with exactly 'nullified' and 'credits'")
        if not isinstance(obj["nullified"], bool):
            raise SchemaError("'nullified' must be a bool")
        if not isinstance(obj["credits"], list):
            raise SchemaError("'credits' must be a list")
        credits = []
        for item in obj["credits"]:
            if not isinstance(item, dict):
                raise SchemaError(f"bad credit: {item!r}")
            player, stat = item.get("player"), item.get("stat")
            if not isinstance(player, str) or not player:
                raise SchemaError(f"bad player: {player!r}")
            if stat not in _STAT_INDEX:
                raise SchemaError(f"unknown stat: {stat!r}")
            keys = set(item)
            if stat in YARDAGE_STATS:
                if keys not in _YARD_KEYS:
                    raise SchemaError(f"a {stat} credit needs 'to' (and optionally 'from'), not {sorted(keys)}")
                try:
                    to = canonical_spot(item["to"])
                    frm = canonical_spot(item["from"]) if "from" in item else None
                except SpotError as e:
                    raise SchemaError(str(e)) from e
                credits.append(CreditV2(player, stat, to=to, from_=frm))
            else:
                if keys != _COUNT_KEYS:
                    raise SchemaError(f"bad credit: {item!r}")
                value = item["value"]
                if not isinstance(value, int) or isinstance(value, bool):
                    raise SchemaError(f"value must be an int: {value!r}")
                credits.append(CreditV2(player, stat, value=value))
        label = cls(obj["nullified"], tuple(credits))
        if label.nullified and label.credits:
            raise SchemaError("a nullified play must have no credits")
        return label


def to_v1(label: PlayLabelV2, los: str | None, posteam: str | None) -> PlayLabel:
    """Compute the yards and return the v1 label this v2 label stands for.

    Raises SchemaError if a spot cannot be read: no `posteam`, or a credit without
    `from` on a play whose line of scrimmage is unknown.
    """
    if label.nullified:
        return PlayLabel(True, ())
    yards: dict[tuple[str, str], int] = {}
    out: list[Credit] = []
    for c in label.credits:
        if not c.is_yardage:
            out.append(Credit(c.player, c.stat, int(c.value)))
            continue
        start = c.from_ if c.from_ is not None else los
        if start is None:
            raise SchemaError(f"{c.player} {c.stat}: no 'from' and the line of scrimmage is unknown")
        if not posteam:
            raise SchemaError("posteam is required to turn spots into yards")
        try:
            gained = yards_between(start, c.to, posteam)
        except SpotError as e:
            raise SchemaError(str(e)) from e
        key = (c.player, c.stat)
        if key not in yards:
            out.append(Credit(c.player, c.stat, 0))  # placeholder keeps first-seen order
        yards[key] = yards.get(key, 0) + gained
    final = []
    for cr in out:
        if cr.stat in YARDAGE_STATS:
            v = yards[(cr.player, cr.stat)]
            if v == 0:
                continue
            cr = Credit(cr.player, cr.stat, v)
        final.append(cr)
    # A placeholder for a repeated key appears once, so no duplicates survive.
    return PlayLabel(False, tuple(final))


def from_v1(label: PlayLabel, los: str | None, posteam: str | None, defteam: str | None,
            from_spots: dict[tuple[str, str], str] | None = None) -> PlayLabelV2:
    """The v2 label for a v1 label: each yardage credit's `to` is its start advanced
    by the credited yards. `from_spots` gives the start for credits that do not
    begin at the line of scrimmage (lateral legs); everything else starts at `los`.

    Raises SpotError when a spot would fall off the field.
    """
    from playparse.ffscore.spots import advance

    if label.nullified:
        return PlayLabelV2(True, ())
    from_spots = from_spots or {}
    out = []
    for c in label.credits:
        if c.stat not in YARDAGE_STATS:
            out.append(CreditV2(c.player, c.stat, value=c.value))
            continue
        frm = from_spots.get((c.player, c.stat))
        start = frm if frm is not None else los
        if start is None or not posteam or not defteam:
            raise SpotError(f"cannot place {c.player} {c.stat}: los={los!r} posteam={posteam!r}")
        to = advance(start, c.value, posteam, defteam)
        if frm is not None and los is not None and spot_to_yardline100(frm, posteam) == spot_to_yardline100(
                los, posteam):
            frm = None  # the start is the line of scrimmage after all: leave it implicit
        out.append(CreditV2(c.player, c.stat, to=to, from_=frm))
    return PlayLabelV2(False, tuple(out)).canonical()


def v2_text_to_v1_text(raw: str, los: str | None, posteam: str | None) -> tuple[str, str | None]:
    """Turn raw v2 model output into text the unchanged v1 scorer reads.

    Returns (text, error). On success `text` is the v1 JSON, with whatever the
    model wrote around its JSON object kept around it, so the v1 scorer's
    `strict_valid` (nothing around the object) means the same thing for v2 runs.
    On failure `text` holds no JSON object at all, so the v1 scorer counts the play
    as invalid: an output that is not valid v2 is never rescued by happening to be
    valid v1 (for example a yardage credit with a "value").
    """
    from playparse.eval.extract import extract_first_json_object

    candidate = extract_first_json_object(raw)
    if candidate is None:
        return "INVALID_V2: no JSON object found", "no JSON object found"
    try:
        v1 = to_v1(PlayLabelV2.from_json(candidate), los, posteam)
    except SchemaError as e:
        return f"INVALID_V2: {e}".replace("{", "(").replace("}", ")"), str(e)
    start = raw.find(candidate)
    return raw[:start] + v1.to_json() + raw[start + len(candidate):], None


__all__ = ["YARDAGE_STATS", "STAT_VOCAB", "CreditV2", "PlayLabelV2", "to_v1", "from_v1", "v2_text_to_v1_text"]
