"""Game-level cross-check: summed per-play ground truth vs official weekly stats.

For every season, the per-play labels are summed into (game, team, player) totals
and compared, stat by stat, with nflverse's official weekly player stats
(`stats_player_week_<season>.parquet`). Those official numbers are built from the
NFL's own per-play stat records, not from the play-by-play columns the builder
reads, so they are an independent check.

Matching method: players are joined on ``(game_id, team, abbreviated name)``, where
the label side uses the credited `desc`-style name and its team is the play's
`posteam` (every v1 stat belongs to the offense), and the official side uses
``player_name`` (with "D. Thomas" collapsed to "D.Thomas", as the builder does) and
``team``. This is the same information a label carries, so it
tests the labels as the model will see them. Official rows whose v1 stats are all
zero (defenders, kickers) are ignored. A GSIS-id join is computed alongside as a
diagnostic, to separate name-matching problems from stat problems.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from playparse.data.ground_truth import build_label, normalize_name
from playparse.data.load_pbp import EXTRA_OFFENSE_PLAY_TYPES, load_pbp_season, load_weekly_stats
from playparse.ffscore.schema import STAT_VOCAB
from playparse.ffscore.scorer import CONFIGS

# official weekly-stats expression for each v1 stat
OFFICIAL_STAT_COLUMNS: dict[str, tuple[str, ...]] = {
    "pass_yds": ("passing_yards",),
    "pass_td": ("passing_tds",),
    "int": ("passing_interceptions",),
    "rush_yds": ("rushing_yards",),
    "rush_td": ("rushing_tds",),
    "rec": ("receptions",),
    "rec_yds": ("receiving_yards",),
    "rec_td": ("receiving_tds",),
    "fumble_lost": ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost"),
    "two_pt": ("passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions"),
}

KEY = ["game_id", "team", "player"]


def credits_frame(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per credit: game_id, team (posteam), player, player_id, stat, value, play_id.

    `player` is the label name with `normalize_name` applied (the join key); the official
    side gets the same normalization, so "Dam. Williams" and "Dam.Williams" meet.
    """
    rows = []
    for rec in pbp.to_dict("records"):
        res = build_label(rec)
        for c, pid in zip(res.label.credits, res.player_ids):
            rows.append((rec["game_id"], rec["play_id"], rec["posteam"], normalize_name(c.player), pid, c.stat, c.value,
                         rec["play_type"]))
    return pd.DataFrame(rows, columns=["game_id", "play_id", "team", "player", "player_id", "stat", "value",
                                       "play_type"])


def gt_totals(credits: pd.DataFrame, by: list[str] = KEY) -> pd.DataFrame:
    wide = credits.pivot_table(index=by, columns="stat", values="value", aggfunc="sum", fill_value=0)
    for s in STAT_VOCAB:
        if s not in wide.columns:
            wide[s] = 0
    return wide[list(STAT_VOCAB)].astype(int)


def official_totals(stats: pd.DataFrame, by: str = "player_name") -> pd.DataFrame:
    s = stats.fillna({c: 0 for cols in OFFICIAL_STAT_COLUMNS.values() for c in cols})
    names = s[by].map(normalize_name, na_action="ignore") if by == "player_name" else s[by]
    out = pd.DataFrame({"game_id": s["game_id"], "team": s["team"], "player": names})
    for stat, cols in OFFICIAL_STAT_COLUMNS.items():
        out[stat] = s[list(cols)].sum(axis=1).astype(int)
    # every lost fumble, including ones nflverse's fantasy formula does not count
    # (lateral fumbles, a lineman's fumbled snap, a fumble after a recovery)
    out["fumble_lost_total"] = s["fumbles_lost_total"].fillna(0).astype(int)
    out["fantasy_points_ppr"] = s["fantasy_points_ppr"].fillna(0)
    out["special_teams_tds"] = s["special_teams_tds"].fillna(0)
    out["season_type"] = s["season_type"]
    # Rows with no v1 stats (defenders, kickers, linemen) are kept so a GT credit can
    # still be looked up, but they only count as "official" players when active.
    out = out.groupby(["game_id", "team", "player"], as_index=True).agg(
        {**{c: "sum" for c in list(STAT_VOCAB) + ["fumble_lost_total", "fantasy_points_ppr",
                                                    "special_teams_tds"]},
         "season_type": "first"})
    out["off_active"] = (out[list(STAT_VOCAB)] != 0).any(axis=1)
    return out


def ppr(df: pd.DataFrame) -> pd.Series:
    cfg = CONFIGS["ppr"]
    return sum(df[s] * cfg[s] for s in STAT_VOCAB)


@dataclass
class SeasonCheck:
    season: int
    joined: pd.DataFrame  # outer join of GT and official totals, columns <stat>_gt / <stat>_off
    summary: dict
    pbp: pd.DataFrame | None = None
    credits: pd.DataFrame | None = None
    # (game_id, team, name) of every player with a kneel in the game (kneels are outside
    # the v1 dataset but carry official rushing yards, usually -1 each)
    kneelers: frozenset = frozenset()


def compare(gt: pd.DataFrame, off: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    j = gt.join(off, how="outer", lsuffix="_gt", rsuffix="_off")
    # a player whose credits net to zero (e.g. +2 and -2 rushing) is not "active"
    j["in_gt"] = (j[[f"{s}_gt" for s in STAT_VOCAB]].fillna(0) != 0).any(axis=1)
    j["in_off"] = j["off_active"].astype("boolean").fillna(False).astype(bool)
    j = j[j.in_gt | j.in_off].copy()
    for s in STAT_VOCAB:
        j[f"{s}_gt"] = j[f"{s}_gt"].fillna(0).astype(int)
        j[f"{s}_off"] = j[f"{s}_off"].fillna(0).astype(int)
    j["fumble_lost_total"] = j["fumble_lost_total"].fillna(0).astype(int)
    j["fantasy_points_ppr"] = j["fantasy_points_ppr"].fillna(0)
    j["special_teams_tds"] = j["special_teams_tds"].fillna(0)
    gt_cols = j[[f"{s}_gt" for s in STAT_VOCAB]].set_axis(list(STAT_VOCAB), axis=1)
    off_cols = j[[f"{s}_off" for s in STAT_VOCAB]].set_axis(list(STAT_VOCAB), axis=1)
    j["ppr_gt"] = ppr(gt_cols)
    j["ppr_off_v1"] = ppr(off_cols)  # official stats, v1 stats only, scored by our scorer
    j["ppr_off_v1_check"] = j["fantasy_points_ppr"] - 6 * j["special_teams_tds"]

    n = len(j)
    summary: dict = {
        "player_games": int(n),
        "matched": int((j.in_gt & j.in_off).sum()),
        "gt_only": int((j.in_gt & ~j.in_off).sum()),
        "official_only": int((~j.in_gt & j.in_off).sum()),
    }
    summary["unmatched_rate"] = (summary["gt_only"] + summary["official_only"]) / max(n, 1)
    per_stat = {}
    for s in STAT_VOCAB:
        g, o = j[f"{s}_gt"], j[f"{s}_off"]
        active = (g != 0) | (o != 0)
        eq = (g == o) & active
        per_stat[s] = {
            "player_games_nonzero": int(active.sum()),
            "agree": int(eq.sum()),
            "agreement_rate": float(eq.sum() / max(active.sum(), 1)),
            "gt_total": int(g.sum()),
            "official_total": int(o.sum()),
            "abs_diff_total": int((g - o).abs().sum()),
        }
    # nflverse's fantasy formula only counts sack/rushing/receiving fumbles. A lost
    # fumble on a lateral, after a recovery, or by a lineman on a botched snap is
    # still in the official `fumbles_lost_total`. The builder charges every lost
    # fumble (it is what the text says), so a mismatch is "explained" when the GT
    # equals the player's official total of lost fumbles.
    g = j["fumble_lost_gt"]
    o = j["fumble_lost_off"]
    t = j["fumble_lost_total"]
    active = (g != 0) | (o != 0)
    explained = (g == o) | (g == t)
    per_stat["fumble_lost"]["agree_incl_uncategorized_fumbles"] = int((explained & active).sum())
    per_stat["fumble_lost"]["agreement_rate_incl_uncategorized_fumbles"] = float(
        (explained & active).sum() / max(active.sum(), 1))
    summary["per_stat"] = per_stat
    all_eq = np.ones(n, dtype=bool)
    for s in STAT_VOCAB:
        all_eq &= (j[f"{s}_gt"] == j[f"{s}_off"]).to_numpy()
    summary["player_games_all_stats_agree"] = float(all_eq.mean()) if n else 1.0
    summary["ppr_mae_vs_official_v1"] = float((j.ppr_gt - j.ppr_off_v1_check).abs().mean())
    summary["ppr_mae_vs_official_raw"] = float((j.ppr_gt - j.fantasy_points_ppr).abs().mean())
    summary["ppr_max_abs_err_v1"] = float((j.ppr_gt - j.ppr_off_v1_check).abs().max())
    summary["ppr_exact_rate_v1"] = float(((j.ppr_gt - j.ppr_off_v1_check).abs() < 0.011).mean())
    # sanity: our scorer applied to official v1 stats reproduces nflverse's points
    summary["scorer_reproduces_official"] = float(
        ((j.ppr_off_v1 - j.ppr_off_v1_check).abs()[j.in_off] < 0.011).mean())
    return j, summary


def check_season(season: int, data_dir=None, extra_play_types: Iterable[str] = ()) -> SeasonCheck:
    pbp = load_pbp_season(season, data_dir, extra_play_types=extra_play_types)
    credits = credits_frame(pbp)
    gt = gt_totals(credits)
    off = official_totals(load_weekly_stats(season, data_dir))
    joined, summary = compare(gt, off)
    summary = {"season": season, "plays": int(len(pbp)), **summary}
    raw = load_pbp_season(season, data_dir, scope=False)
    kn = raw[raw["play_type"] == "qb_kneel"]
    kneelers = frozenset(zip(kn["game_id"], kn["posteam"], kn["rusher_player_name"].map(normalize_name, na_action="ignore")))
    return SeasonCheck(season, joined, summary, pbp, credits, kneelers)


def check_season_by_id(season: int, data_dir=None, extra_play_types: Iterable[str] = ()) -> dict:
    """Diagnostic: the same comparison joined on GSIS id instead of name."""
    pbp = load_pbp_season(season, data_dir, extra_play_types=extra_play_types)
    credits = credits_frame(pbp)
    credits = credits.assign(player=credits["player_id"].fillna("?" + credits["player"]))
    gt = gt_totals(credits)
    off = official_totals(load_weekly_stats(season, data_dir), by="player_id")
    _, summary = compare(gt, off)
    return summary


# ---------------------------------------------------------------------------
# Explaining the disagreements
# ---------------------------------------------------------------------------

MISMATCH_CLASSES = (
    "lateral",                # player was in a lateral play that game (only the last lateral is recorded)
    "uncategorized_fumble",   # GT charges a lost fumble that nflverse's fantasy formula leaves out
    "desc_name_variant",      # desc spells the same player two ways in one game; credits split
    "kneel_excluded",         # QB rushing yards from kneels, which the v1 dataset does not include
    "out_of_scope_play",      # official yards from another play type v1 does not label (fake punt)
    "unexplained",
)


def mismatches(check: SeasonCheck) -> pd.DataFrame:
    """One row per (game, team, player, stat) where GT and official disagree, with a class."""
    j, pbp, credits = check.joined, check.pbp, check.credits

    lateral_rows = pbp[(pbp["lateral_reception"] == 1) | (pbp["lateral_rush"] == 1)
                       | pbp["desc"].str.contains(r"Lateral to|Pass back to", regex=True)]
    lateral_desc: dict[tuple[str, str], list[str]] = {}
    for g, t, d in zip(lateral_rows["game_id"], lateral_rows["posteam"], lateral_rows["desc"]):
        lateral_desc.setdefault((g, t), []).append(d)

    # names that share one GSIS id within a game -> the desc spelled a player two ways
    ids = credits.dropna(subset=["player_id"]).groupby(["game_id", "team", "player_id"])["player"].unique()
    variant_names = {(g, t, n) for (g, t, _), names in ids.items() if len(names) > 1 for n in names}
    variant_games = {(g, t) for (g, t, _), names in ids.items() if len(names) > 1}
    id_of = {(g, t, n): pid for (g, t, pid), names in ids.items() for n in names}

    rows = []
    for (game_id, team, player), r in j.iterrows():
        for s in STAT_VOCAB:
            gv, ov = int(r[f"{s}_gt"]), int(r[f"{s}_off"])
            if gv == ov:
                continue
            if s == "fumble_lost" and gv == int(r["fumble_lost_total"]):
                cls = "uncategorized_fumble"
            elif s == "rush_yds" and (game_id, team, player) in check.kneelers and gv > ov:
                cls = "kneel_excluded"
            elif any(player in d for d in lateral_desc.get((game_id, team), [])):
                cls = "lateral"
            elif (game_id, team, player) in variant_names or (not r["in_gt"] and (game_id, team) in variant_games):
                cls = "desc_name_variant"
            elif not r["in_gt"] and gv == 0:
                cls = "out_of_scope_play"
            else:
                cls = "unexplained"
            rows.append({"season": check.season, "game_id": game_id, "team": team, "player": player,
                         "stat": s, "gt": gv, "official": ov, "class": cls,
                         "player_id": id_of.get((game_id, team, player))})
    return pd.DataFrame(rows, columns=["season", "game_id", "team", "player", "stat", "gt", "official",
                                       "class", "player_id"])


def run_all(seasons: Iterable[int], data_dir=None, log=print) -> dict:
    """Cross-check every season in two scopes and explain every disagreement."""
    scopes = {"dataset": (), "dataset_plus_kneels_spikes": tuple(sorted(EXTRA_OFFENSE_PLAY_TYPES))}
    out: dict = {"method": __doc__.strip(), "scopes": {}, "mismatch_classes": {}, "mismatches": []}
    for scope, extra in scopes.items():
        per_season = {}
        all_mm = []
        for season in seasons:
            check = check_season(season, data_dir, extra_play_types=extra)
            per_season[str(season)] = check.summary
            mm = mismatches(check)
            all_mm.append(mm)
            log(f"[{scope}] {season}: all-stat agreement {check.summary['player_games_all_stats_agree']:.4f}, "
                f"PPR MAE {check.summary['ppr_mae_vs_official_v1']:.4f}, mismatches {len(mm)}")
        mm = pd.concat(all_mm, ignore_index=True)
        out["scopes"][scope] = {"extra_play_types": list(extra), "per_season": per_season,
                                "overall": _overall(per_season)}
        counts = mm.groupby(["class", "stat"]).size()
        out["mismatch_classes"][scope] = {
            cls: {stat: int(n) for (c, stat), n in counts.items() if c == cls} for cls in MISMATCH_CLASSES}
        if scope == "dataset_plus_kneels_spikes":
            out["mismatches"] = mm.sort_values(["class", "season", "game_id", "player", "stat"]).to_dict("records")
    return out


def _overall(per_season: dict) -> dict:
    stats = {}
    for s in STAT_VOCAB:
        agree = sum(v["per_stat"][s]["agree"] for v in per_season.values())
        n = sum(v["per_stat"][s]["player_games_nonzero"] for v in per_season.values())
        stats[s] = {"agree": agree, "player_games_nonzero": n, "agreement_rate": agree / max(n, 1)}
    fl_x = sum(v["per_stat"]["fumble_lost"]["agree_incl_uncategorized_fumbles"] for v in per_season.values())
    stats["fumble_lost"]["agreement_rate_incl_uncategorized_fumbles"] = (
        fl_x / max(stats["fumble_lost"]["player_games_nonzero"], 1))
    pg = sum(v["player_games"] for v in per_season.values())
    return {
        "player_games": pg,
        "unmatched": sum(v["gt_only"] + v["official_only"] for v in per_season.values()),
        "player_games_all_stats_agree": sum(v["player_games_all_stats_agree"] * v["player_games"]
                                            for v in per_season.values()) / max(pg, 1),
        "ppr_mae_vs_official_v1": sum(v["ppr_mae_vs_official_v1"] * v["player_games"]
                                      for v in per_season.values()) / max(pg, 1),
        "ppr_exact_rate_v1": sum(v["ppr_exact_rate_v1"] * v["player_games"]
                                 for v in per_season.values()) / max(pg, 1),
        "per_stat": stats,
    }


def main(argv=None) -> int:
    import argparse
    import json
    from pathlib import Path

    from playparse import paths
    from playparse.data.load_pbp import ALL_SEASONS

    ap = argparse.ArgumentParser(description="Cross-check per-play ground truth against official weekly stats")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=paths.RESULTS / "p1" / "crosscheck.json")
    ap.add_argument("--seasons", type=int, nargs="*", default=list(ALL_SEASONS))
    args = ap.parse_args(argv)
    result = run_all(args.seasons, args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(f"wrote {args.out}")
    return 0


__all__ = ["EXTRA_OFFENSE_PLAY_TYPES", "MISMATCH_CLASSES", "check_season", "check_season_by_id", "compare",
           "credits_frame", "gt_totals", "mismatches", "official_totals", "run_all"]


if __name__ == "__main__":
    raise SystemExit(main())
