"""Dev-set estimate for the R0 regex parser, on TRAIN seasons only (2015-2022).

The real ground-truth builder (`playparse/data/ground_truth.py`) is owned by another
workstream. This script carries its own *approximate* structured-column labeler and
bucketer so the regex could be developed and sanity-checked before that lands. Its
numbers are a development estimate, not a test result: the conventions may differ in
details from the frozen ground truth, and the test season (2024) is never read here.

Usage:
    python scripts/regex_dev_estimate.py [--seasons 2015-2022] [--sample N]
        [--write-jsonl data/cache/regex_dev.jsonl] [--mismatches 20] [--out results/...]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playparse.eval.baselines.regex_parser import RegexPredictor  # noqa: E402
from playparse.eval.harness import format_summary, run_eval  # noqa: E402
from playparse.ffscore.schema import Credit, PlayLabel  # noqa: E402
from playparse.paths import DATA_RAW  # noqa: E402

TRAIN_SEASONS = range(2015, 2023)
COLUMNS = [
    "game_id", "play_id", "season", "week", "season_type", "posteam", "desc", "play_type",
    "complete_pass", "interception", "penalty", "replay_or_challenge", "touchdown",
    "pass_touchdown", "rush_touchdown", "td_player_name", "fumble", "fumble_lost",
    "fumbled_1_player_name", "fumbled_1_team", "two_point_attempt", "two_point_conv_result",
    "passer_player_name", "receiver_player_name", "rusher_player_name", "passing_yards",
    "receiving_yards", "rushing_yards", "lateral_reception", "lateral_rush",
    "lateral_receiver_player_name", "lateral_receiving_yards", "lateral_rusher_player_name",
    "lateral_rushing_yards",
]


def _nn(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def approx_label(r) -> PlayLabel:
    """Structured columns -> label, following the conventions in the P1 brief."""
    if r.play_type == "no_play":
        return PlayLabel(True, ())
    c: list[Credit] = []
    if r.two_point_attempt == 1:
        if r.two_point_conv_result == "success":
            for n in (r.passer_player_name, r.receiver_player_name, r.rusher_player_name):
                if _nn(n):
                    c.append(Credit(n, "two_pt", 1))
        return PlayLabel(False, tuple(c))
    if r.complete_pass == 1 and _nn(r.passer_player_name):
        c.append(Credit(r.passer_player_name, "pass_yds", int(r.passing_yards)))
        c.append(Credit(r.receiver_player_name, "rec", 1))
        c.append(Credit(r.receiver_player_name, "rec_yds", int(r.receiving_yards)))
        if r.lateral_reception == 1 and _nn(r.lateral_receiver_player_name):
            c.append(Credit(r.lateral_receiver_player_name, "rec_yds", int(r.lateral_receiving_yards)))
    if r.interception == 1 and _nn(r.passer_player_name):
        c.append(Credit(r.passer_player_name, "int", 1))
    if r.pass_touchdown == 1 and _nn(r.td_player_name):
        c.append(Credit(r.passer_player_name, "pass_td", 1))
        c.append(Credit(r.td_player_name, "rec_td", 1))
    if _nn(r.rusher_player_name) and _nn(r.rushing_yards):
        c.append(Credit(r.rusher_player_name, "rush_yds", int(r.rushing_yards)))
        if r.lateral_rush == 1 and _nn(r.lateral_rusher_player_name):
            c.append(Credit(r.lateral_rusher_player_name, "rush_yds", int(r.lateral_rushing_yards)))
    if r.rush_touchdown == 1 and _nn(r.td_player_name):
        c.append(Credit(r.td_player_name, "rush_td", 1))
    if r.fumble_lost == 1 and _nn(r.fumbled_1_player_name) and r.fumbled_1_team == r.posteam:
        c.append(Credit(r.fumbled_1_player_name, "fumble_lost", 1))
    return PlayLabel(False, tuple(c))


def approx_bucket(r) -> str:
    """Rarest-first precedence (approximation of playparse/data/buckets.py)."""
    if r.lateral_reception == 1 or r.lateral_rush == 1:
        return "lateral"
    if r.two_point_attempt == 1:
        return "two_point"
    if r.replay_or_challenge == 1:
        return "challenge"
    if r.play_type == "no_play":
        return "penalty_nullified"
    if r.fumble == 1:
        return "fumble"
    if r.interception == 1:
        return "interception"
    if r.penalty == 1:
        return "penalty_stands"
    if r.touchdown == 1:
        return "td"
    return "normal"


def load(seasons, sample: int | None, seed: int = 0) -> pd.DataFrame:
    frames = []
    for s in seasons:
        if s not in TRAIN_SEASONS:
            raise SystemExit(f"season {s} is not a train season; dev estimates use 2015-2022 only")
        df = pd.read_parquet(DATA_RAW / f"pbp_{s}.parquet", columns=COLUMNS)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    in_scope = df.play_type.isin(["pass", "run"]) | (df.two_point_attempt == 1)
    nullified = (df.play_type == "no_play") & df.desc.str.contains("No Play", na=False)
    df = df[(in_scope | nullified) & df.desc.notna()]
    if sample:
        sizes = df.groupby("game_id").size().sample(frac=1, random_state=seed)
        keep = sizes.index[(sizes.cumsum() - sizes) < sample]  # whole games until ~sample plays
        df = df[df.game_id.isin(keep)]
    return df


def to_records(df: pd.DataFrame) -> list[dict]:
    recs = []
    for r in df.itertuples(index=False):
        recs.append(
            {
                "game_id": r.game_id,
                "play_id": int(r.play_id),
                "season": int(r.season),
                "week": int(r.week),
                "season_type": r.season_type,
                "posteam": r.posteam,
                "desc": r.desc,
                "bucket": approx_bucket(r),
                "label": approx_label(r).to_json(),
            }
        )
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2015-2022")
    ap.add_argument("--sample", type=int, default=None, help="approx. plays (whole games)")
    ap.add_argument("--write-jsonl", default=None)
    ap.add_argument("--mismatches", type=int, default=0)
    ap.add_argument("--bucket", default=None, help="only print mismatches from this bucket")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    a, _, b = args.seasons.partition("-")
    seasons = range(int(a), int(b or a) + 1)
    recs = to_records(load(seasons, args.sample))
    if args.write_jsonl:
        Path(args.write_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.write_jsonl, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    res = run_eval(
        RegexPredictor(),
        recs,
        rung="r0-dev",
        out_dir=args.out,
        n_boot=200,
        extra_meta={"split": f"DEV (train seasons {args.seasons}), approximate labels"},
    )
    print(format_summary(res))
    if args.mismatches:
        pred = RegexPredictor().predict_batch(recs)
        shown = 0
        for r, p in zip(recs, pred):
            if args.bucket and r["bucket"] != args.bucket:
                continue
            if not PlayLabel.from_json(p).matches(PlayLabel.from_json(r["label"])):
                print("\n--", r["bucket"], r["posteam"], "|", r["desc"])
                print("   gold:", r["label"])
                print("   pred:", p)
                shown += 1
                if shown >= args.mismatches:
                    break


if __name__ == "__main__":
    main()
