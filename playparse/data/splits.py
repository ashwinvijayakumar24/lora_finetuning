"""Season splits and the fixed subsets built from them.

* ``train`` 2015-2022, ``val`` 2023, ``test`` 2024. Splitting by season mimics
  deployment: the model always parses a season it has never seen.
* ``train_50k``: a seeded uniform sample of 50,000 training plays for sweeps.
* ``eval_lite``: a ~3,000-play subset of the test season for local (MPS) runs.

eval_lite is built in two parts, recorded per play in ``eval_lite_part``:

1. ``game`` - whole test games, drawn in a seeded random order until the part
   reaches `core_plays`. Keeping whole games makes game-level fantasy-point MAE
   meaningful on this part.
2. ``topup`` - for every bucket still under `min_per_bucket` plays, extra plays of
   that bucket sampled from the remaining games, so per-bucket metrics are not
   computed on a handful of plays. A bucket with fewer plays than that in the whole
   test season (``lateral``) simply contributes all of them.

Game-level MAE on eval_lite must use only the ``game`` part.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Sequence

from playparse.data.buckets import PRECEDENCE

SPLIT_SEASONS: dict[str, tuple[int, ...]] = {
    "train": tuple(range(2015, 2023)),
    "val": (2023,),
    "test": (2024,),
}

TRAIN_SUBSET_SIZE = 50_000
TRAIN_SUBSET_SEED = 20151
EVAL_LITE_SEED = 20241
EVAL_LITE_CORE_PLAYS = 2_400
EVAL_LITE_MIN_PER_BUCKET = 100


def split_of(season: int) -> str:
    for name, seasons in SPLIT_SEASONS.items():
        if season in seasons:
            return name
    raise ValueError(f"season {season} is not in any split")


def sample_indices(n_total: int, k: int = TRAIN_SUBSET_SIZE, seed: int = TRAIN_SUBSET_SEED) -> list[int]:
    """Sorted, seeded sample of row indices (rows are in a fixed canonical order)."""
    if k >= n_total:
        return list(range(n_total))
    return sorted(random.Random(seed).sample(range(n_total), k))


def eval_lite(
    records: Sequence[dict],
    core_plays: int = EVAL_LITE_CORE_PLAYS,
    min_per_bucket: int = EVAL_LITE_MIN_PER_BUCKET,
    seed: int = EVAL_LITE_SEED,
) -> list[dict]:
    """Select the eval_lite subset from test-split records (each needs game_id, bucket)."""
    rng = random.Random(seed)
    by_game: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_game[r["game_id"]].append(r)
    games = sorted(by_game)
    rng.shuffle(games)

    chosen_games: list[str] = []
    n = 0
    for g in games:
        if n >= core_plays:
            break
        chosen_games.append(g)
        n += len(by_game[g])
    core_set = set(chosen_games)
    core = [dict(r, eval_lite_part="game") for g in chosen_games for r in by_game[g]]

    counts = Counter(r["bucket"] for r in core)
    topup: list[dict] = []
    for bucket in PRECEDENCE:
        need = min_per_bucket - counts[bucket]
        if need <= 0:
            continue
        pool = [r for r in records if r["bucket"] == bucket and r["game_id"] not in core_set]
        take = pool if len(pool) <= need else rng.sample(pool, need)
        topup.extend(dict(r, eval_lite_part="topup") for r in take)

    out = core + topup
    out.sort(key=lambda r: (r["game_id"], r["play_id"]))
    return out
