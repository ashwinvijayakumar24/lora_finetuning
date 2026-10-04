from collections import Counter

import pytest

from playparse.data.buckets import PRECEDENCE
from playparse.data.splits import SPLIT_SEASONS, eval_lite, sample_indices, split_of


def test_seasons_partition():
    assert split_of(2015) == split_of(2022) == "train"
    assert split_of(2023) == "val" and split_of(2024) == "test"
    all_seasons = [s for v in SPLIT_SEASONS.values() for s in v]
    assert sorted(all_seasons) == list(range(2015, 2025))
    with pytest.raises(ValueError):
        split_of(2014)


def test_sample_indices_deterministic_and_sorted():
    a = sample_indices(1000, 50, seed=7)
    assert a == sample_indices(1000, 50, seed=7)
    assert a == sorted(a) and len(set(a)) == 50
    assert a != sample_indices(1000, 50, seed=8)
    assert sample_indices(10, 50) == list(range(10))


def _records():
    # 40 games x 50 plays; rare buckets sprinkled in
    recs = []
    for g in range(40):
        for p in range(50):
            bucket = "normal"
            if p == 0 and g % 4 == 0:
                bucket = "two_point"
            elif p == 1 and g % 10 == 0:
                bucket = "lateral"
            elif p < 6:
                bucket = "td"
            recs.append({"game_id": f"g{g:02d}", "play_id": p, "bucket": bucket})
    return recs


def test_eval_lite_whole_games_plus_topup():
    recs = _records()
    lite = eval_lite(recs, core_plays=300, min_per_bucket=8, seed=1)
    game_part = [r for r in lite if r["eval_lite_part"] == "game"]
    games = {r["game_id"] for r in game_part}
    # the game part is made of complete games
    per_game = Counter(r["game_id"] for r in recs)
    assert all(sum(1 for r in game_part if r["game_id"] == g) == per_game[g] for g in games)
    assert len(game_part) >= 300 and len(games) == 6
    counts = Counter(r["bucket"] for r in lite)
    assert counts["two_point"] >= 8  # topped up from other games
    assert counts["lateral"] == 4  # only 4 exist in total: all of them
    topup = [r for r in lite if r["eval_lite_part"] == "topup"]
    assert all(r["game_id"] not in games for r in topup)
    assert lite == eval_lite(recs, core_plays=300, min_per_bucket=8, seed=1)
    assert set(counts) <= set(PRECEDENCE)
