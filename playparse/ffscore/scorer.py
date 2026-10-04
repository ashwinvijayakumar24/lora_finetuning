"""Deterministic fantasy scorer: credits -> points.

The model only reads plays; all arithmetic lives here. A scoring config is a mapping
from each stat in `STAT_VOCAB` to points per unit (per yard for yardage stats, per
occurrence for count stats).

Built-in configs
----------------
* ``standard``  - the common default: 0.04/pass yd (1 per 25), 4/pass TD, -2/INT,
  0.1/rush or rec yd, 6/rush or rec TD, -2/fumble lost, 2/two-point conversion,
  0 per reception.
* ``half_ppr``  - standard + 0.5 per reception.
* ``ppr``       - standard + 1 per reception. This is exactly nflverse's
  ``fantasy_points_ppr`` restricted to v1 stats (it also adds 6 per special-teams
  touchdown, which v1 does not label).
* ``espn_14team_ppr`` - the owner's 14-team ESPN league. Assumed to use ESPN's
  full-PPR defaults, which for v1 stats coincide with ``ppr``: 0.04/pass yd,
  4/pass TD, -2/INT, 0.1/rush yd, 6/rush TD, 1/rec, 0.1/rec yd, 6/rec TD,
  -2/fumble lost, 2/2pt. ESPN's yardage bonuses and kicking/defense rules are not
  modelled. Edit `CONFIGS["espn_14team_ppr"]` if the league settings differ.

Scores are computed per play without rounding and only rounded at the end
(`round_points`), because ESPN-style 0.04/yd scoring must sum yards before rounding
to match official totals.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from playparse.ffscore.schema import STAT_VOCAB, PlayLabel

ScoringConfig = Mapping[str, float]

_STANDARD: dict[str, float] = {
    "pass_yds": 0.04,
    "pass_td": 4.0,
    "int": -2.0,
    "rush_yds": 0.1,
    "rush_td": 6.0,
    "rec": 0.0,
    "rec_yds": 0.1,
    "rec_td": 6.0,
    "fumble_lost": -2.0,
    "two_pt": 2.0,
}

CONFIGS: dict[str, dict[str, float]] = {
    "standard": dict(_STANDARD),
    "half_ppr": {**_STANDARD, "rec": 0.5},
    "ppr": {**_STANDARD, "rec": 1.0},
    "espn_14team_ppr": {**_STANDARD, "rec": 1.0},
}

for _name, _cfg in CONFIGS.items():
    assert set(_cfg) == set(STAT_VOCAB), f"config {_name} must cover exactly STAT_VOCAB"


def get_config(config: str | ScoringConfig) -> ScoringConfig:
    if isinstance(config, str):
        try:
            return CONFIGS[config]
        except KeyError:
            raise KeyError(f"unknown scoring config {config!r}; known: {sorted(CONFIGS)}") from None
    missing = set(STAT_VOCAB) - set(config)
    if missing:
        raise ValueError(f"scoring config missing stats: {sorted(missing)}")
    return config


def stat_totals(labels: Iterable[PlayLabel]) -> dict[str, dict[str, int]]:
    """Sum credits per player per stat across plays. Nullified plays contribute nothing."""
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for label in labels:
        if label.nullified:
            continue
        for c in label.credits:
            out[c.player][c.stat] += c.value
    return {p: dict(s) for p, s in out.items()}


def score_totals(totals: Mapping[str, Mapping[str, int]], config: str | ScoringConfig) -> dict[str, float]:
    cfg = get_config(config)
    return {p: sum(cfg[s] * v for s, v in stats.items()) for p, stats in totals.items()}


def score_label(label: PlayLabel, config: str | ScoringConfig) -> dict[str, float]:
    """Points per player for one play."""
    return score_totals(stat_totals([label]), config)


def score_game(labels: Iterable[PlayLabel], config: str | ScoringConfig) -> dict[str, float]:
    """Points per player for a sequence of plays (e.g. one game).

    Yards are summed before multiplying, so 0.04/yd scoring is exact on game totals.
    """
    return score_totals(stat_totals(labels), config)


def score_games(
    plays: Iterable[tuple[str, PlayLabel]], config: str | ScoringConfig
) -> dict[tuple[str, str], float]:
    """Aggregate (game_id, label) pairs into {(game_id, player): points}."""
    by_game: dict[str, list[PlayLabel]] = defaultdict(list)
    for game_id, label in plays:
        by_game[game_id].append(label)
    out: dict[tuple[str, str], float] = {}
    for game_id, labels in by_game.items():
        for player, pts in score_game(labels, config).items():
            out[(game_id, player)] = pts
    return out


def round_points(x: float, ndigits: int = 2) -> float:
    return round(x + 0.0, ndigits)
