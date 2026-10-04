"""Load nflverse play-by-play and weekly player stats from local parquet files.

Only the columns the ground-truth builder, the bucketer, and the cross-check need
are read, which keeps a full season at roughly 1/10th of its 372-column size.

Scope decisions (see docs/phases/P1-data.md for the reasoning):

* **Season types.** Both regular season (`REG`) and postseason (`POST`) plays are
  kept. Playoff plays are written in exactly the same language, and the official
  weekly stats we cross-check against include both.
* **Play types.** v1 keeps `play_type in {pass, run, no_play}` plus every
  two-point attempt. In nflverse data a two-point attempt is an ordinary row with
  `two_point_attempt == 1` and `play_type` of `pass` or `run` (`play_type_nfl ==
  'PAT2'`); one followed by a penalty shows up as `play_type == 'no_play'` (whether
  the try was wiped out or the foul was enforced between downs and the try stood;
  the ground-truth builder tells them apart). The `| two_point_attempt == 1` term
  in the filter is therefore defensive: it changes nothing today but keeps the rule
  explicit if nflverse ever relabels them.
* **Aborted plays** (fumbled snaps, botched handoffs) are real plays with real
  stats. nflverse files them as `play_type == 'run'` (occasionally `pass`), so they
  pass the filter and are labeled like any other run.
* **Excluded:** kickoffs, punts, field goals, extra points, kneels, spikes, and
  rows with no play type (quarter ends). Kneels do carry official rushing yards
  (usually -1); the cross-check reports their effect separately.
* **Timeouts are excluded too**, although nflverse files them as `no_play`
  (`is_timeout_row`). They are not plays, and keeping them would have made 44% of
  the `penalty_nullified` bucket the trivial text "Timeout #1 by SF at 10:39."
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd

from playparse import paths

ALL_SEASONS: tuple[int, ...] = tuple(range(2015, 2025))

V1_PLAY_TYPES: frozenset[str] = frozenset({"pass", "run", "no_play"})
# Offensive plays outside the v1 dataset that still carry official stats. The
# cross-check can label them to measure how much the scope choice costs.
EXTRA_OFFENSE_PLAY_TYPES: frozenset[str] = frozenset({"qb_kneel", "qb_spike"})

PBP_COLUMNS: tuple[str, ...] = (
    # identity and splits
    "game_id", "play_id", "season", "week", "season_type", "game_date",
    # input
    "desc", "posteam", "defteam",
    # play shape
    "play_type", "play_type_nfl", "aborted_play", "play_deleted",
    "penalty", "penalty_type", "penalty_team",
    "replay_or_challenge", "replay_or_challenge_result",
    "pass_attempt", "rush_attempt", "complete_pass", "incomplete_pass", "sack",
    "qb_scramble", "qb_kneel", "qb_spike", "interception",
    "touchdown", "pass_touchdown", "rush_touchdown", "return_touchdown", "td_team",
    "two_point_attempt", "two_point_conv_result",
    "fumble", "fumble_lost", "lateral_reception", "lateral_rush",
    # values
    "yards_gained", "passing_yards", "receiving_yards", "rushing_yards",
    "lateral_receiving_yards", "lateral_rushing_yards",
    # players (abbreviated desc-style names plus GSIS ids for diagnostics)
    "passer_player_name", "passer_player_id",
    "receiver_player_name", "receiver_player_id",
    "rusher_player_name", "rusher_player_id",
    "lateral_receiver_player_name", "lateral_receiver_player_id",
    "lateral_rusher_player_name", "lateral_rusher_player_id",
    "td_player_name", "td_player_id",
    "interception_player_name",
    "fumbled_1_player_name", "fumbled_1_player_id", "fumbled_1_team",
    "fumbled_2_player_name", "fumbled_2_player_id", "fumbled_2_team",
    "fumble_recovery_1_team", "fumble_recovery_1_player_name",
    "fumble_recovery_2_team", "fumble_recovery_2_player_name",
)

STATS_COLUMNS: tuple[str, ...] = (
    "player_id", "player_name", "player_display_name", "position", "team",
    "season", "week", "season_type", "game_id",
    "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
    "sack_fumbles", "sack_fumbles_lost", "passing_2pt_conversions",
    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles", "rushing_fumbles_lost",
    "rushing_2pt_conversions",
    "receptions", "targets", "receiving_yards", "receiving_tds", "receiving_fumbles",
    "receiving_fumbles_lost", "receiving_2pt_conversions",
    "special_teams_tds", "fumble_recovery_tds", "fumbles_lost_total",
    "fantasy_points", "fantasy_points_ppr",
)


def pbp_path(season: int, data_dir: Path | None = None) -> Path:
    return (data_dir or paths.DATA_RAW) / f"pbp_{season}.parquet"


def stats_path(season: int, data_dir: Path | None = None) -> Path:
    return (data_dir or paths.DATA_RAW) / f"stats_player_week_{season}.parquet"


def is_timeout_row(df: pd.DataFrame) -> pd.Series:
    """`no_play` rows that are not plays at all (timeouts).

    nflverse files every timeout ("Timeout #2 by BAL at 00:31.") as
    ``play_type == 'no_play'`` with no `posteam`: 20,006 of the 45,784 `no_play`
    rows in 2015-2024. A real wiped-out play always has a penalty in its text,
    so a `no_play` row without the word "penalty" is dropped. Two timeouts mention
    a penalty in a scorer's comment ("Timeout #3 by TEN at 00:22. penalty was
    charged due to an injury..."), so a row that starts with "Timeout" and never
    says "No Play" is dropped too. See docs/issues/p1-timeouts-filed-as-no-play.md.
    """
    desc = df["desc"].fillna("")
    no_penalty_text = ~desc.str.contains("penalty", case=False, regex=False)
    bare_timeout = desc.str.match(r"\s*Timeout\b") & ~desc.str.contains("No Play", case=False, regex=False)
    return (df["play_type"] == "no_play") & (no_penalty_text | bare_timeout)


def in_v1_scope(df: pd.DataFrame, extra_play_types: Iterable[str] = ()) -> pd.Series:
    """Boolean mask of rows that belong in the v1 dataset."""
    keep = set(V1_PLAY_TYPES) | set(extra_play_types)
    in_scope = df["play_type"].isin(keep) | (df["two_point_attempt"].fillna(0) == 1)
    return in_scope & ~is_timeout_row(df)


def load_pbp_season(
    season: int,
    data_dir: Path | None = None,
    *,
    scope: bool = True,
    extra_play_types: Iterable[str] = (),
) -> pd.DataFrame:
    """Read one season with only the needed columns, optionally filtered to v1 scope."""
    path = pbp_path(season, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/fetch_data.sh or set PLAYPARSE_DATA_DIR")
    df = pd.read_parquet(path, columns=list(PBP_COLUMNS))
    if scope:
        df = df[in_v1_scope(df, extra_play_types)]
    df = df[df["desc"].notna()]
    return df.reset_index(drop=True)


def iter_pbp(
    seasons: Iterable[int] = ALL_SEASONS,
    data_dir: Path | None = None,
    **kwargs,
) -> Iterator[pd.DataFrame]:
    """Yield one filtered season at a time so a full build never holds 10 seasons."""
    for season in seasons:
        yield load_pbp_season(season, data_dir, **kwargs)


def load_weekly_stats(season: int, data_dir: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(stats_path(season, data_dir), columns=list(STATS_COLUMNS))
