"""The output contract: what the model emits for one play.

Every component (ground-truth builder, baselines, training targets, eval, serving)
goes through this module, so the serialized form of a label is defined exactly once.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

STAT_VOCAB: tuple[str, ...] = (
    "pass_yds",
    "pass_td",
    "int",
    "rush_yds",
    "rush_td",
    "rec",
    "rec_yds",
    "rec_td",
    "fumble_lost",
    "two_pt",
)
_STAT_INDEX = {s: i for i, s in enumerate(STAT_VOCAB)}

# Stats whose value is a count of 1 per occurrence (yardage stats may be any int).
COUNT_STATS = frozenset({"pass_td", "int", "rush_td", "rec", "rec_td", "fumble_lost", "two_pt"})


class SchemaError(ValueError):
    """Raised when model output does not conform to the label schema."""


@dataclass(frozen=True, order=True)
class Credit:
    player: str
    stat: str
    value: int

    def sort_key(self) -> tuple[int, str, int]:
        return (_STAT_INDEX[self.stat], self.player, self.value)


@dataclass(frozen=True)
class PlayLabel:
    nullified: bool
    credits: tuple[Credit, ...] = field(default_factory=tuple)

    def canonical(self) -> "PlayLabel":
        """Credits sorted by (stat vocabulary order, player). Training targets use this."""
        return PlayLabel(self.nullified, tuple(sorted(self.credits, key=Credit.sort_key)))

    def to_json(self) -> str:
        """Compact, canonical serialization used as the training completion."""
        c = self.canonical()
        return json.dumps(
            {
                "nullified": c.nullified,
                "credits": [{"player": x.player, "stat": x.stat, "value": x.value} for x in c.credits],
            },
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, text: str) -> "PlayLabel":
        """Parse and validate. Raises SchemaError on any deviation from the schema."""
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, TypeError) as e:
            raise SchemaError(f"not valid JSON: {e}") from e
        if not isinstance(obj, dict) or set(obj) != {"nullified", "credits"}:
            raise SchemaError("top level must be an object with exactly 'nullified' and 'credits'")
        if not isinstance(obj["nullified"], bool):
            raise SchemaError("'nullified' must be a bool")
        if not isinstance(obj["credits"], list):
            raise SchemaError("'credits' must be a list")
        credits = []
        for item in obj["credits"]:
            if not isinstance(item, dict) or set(item) != {"player", "stat", "value"}:
                raise SchemaError(f"bad credit: {item!r}")
            player, stat, value = item["player"], item["stat"], item["value"]
            if not isinstance(player, str) or not player:
                raise SchemaError(f"bad player: {player!r}")
            if stat not in _STAT_INDEX:
                raise SchemaError(f"unknown stat: {stat!r}")
            if not isinstance(value, int) or isinstance(value, bool):
                raise SchemaError(f"value must be an int: {value!r}")
            credits.append(Credit(player, stat, value))
        label = cls(obj["nullified"], tuple(credits))
        if label.nullified and label.credits:
            raise SchemaError("a nullified play must have no credits")
        return label

    def credit_multiset(self) -> dict[Credit, int]:
        out: dict[Credit, int] = {}
        for c in self.credits:
            out[c] = out.get(c, 0) + 1
        return out

    def matches(self, other: "PlayLabel") -> bool:
        """Exact match: same nullified flag and the same credits, order-insensitive."""
        return self.nullified == other.nullified and self.credit_multiset() == other.credit_multiset()
