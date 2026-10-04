"""A minimal full-PPR scoring function used as the eval's default `score_fn`.

The real scorer (`playparse/ffscore/scorer.py`) is being built separately. Until it
lands, game-level fantasy-point MAE uses this stand-in. The interface is deliberately
"per-player stat totals -> points" rather than "one credit -> points": real leagues
have non-additive rules (for example a 100-yard bonus), and those can only be
expressed on a player's totals for the game. Swap this for the real scorer by passing
`score_fn=` to the metrics and harness functions.
"""
from __future__ import annotations

from collections.abc import Mapping

# Standard full-PPR weights (ESPN / Yahoo defaults).
PPR_WEIGHTS: dict[str, float] = {
    "pass_yds": 0.04,  # 1 point per 25 yards
    "pass_td": 4.0,
    "int": -2.0,
    "rush_yds": 0.1,
    "rush_td": 6.0,
    "rec": 1.0,
    "rec_yds": 0.1,
    "rec_td": 6.0,
    "fumble_lost": -2.0,
    "two_pt": 2.0,
}


def ppr_points(totals: Mapping[str, int]) -> float:
    """Fantasy points for one player's stat totals (stat name -> summed value)."""
    return float(sum(PPR_WEIGHTS[stat] * value for stat, value in totals.items()))
