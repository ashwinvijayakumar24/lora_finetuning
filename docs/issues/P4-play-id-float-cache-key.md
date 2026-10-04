# P4: float play ids would miss the teacher cache and pay twice

**What happened.** The teacher cache is keyed by `(game_id, play_id, sample_idx)`.
nflverse `play_id` is an integer, but pandas stores an integer column as float64 as
soon as one value is missing, so a record can arrive with `play_id = 55.0`. Keyed
naively, `"55.0"` and `"55"` are different plays: a resumed run would find nothing in
the cache and buy every label again.

**How it was found.** Reviewing how the dataset JSONL is likely produced (pandas,
then `to_json`) while designing the cache key. Not yet observed in real data.

**Root cause.** The cache key used the raw value's string form, so the same play
could have several spellings.

**Fix.** `playparse/distill/teacher.py::play_key` normalizes `play_id`: an integral
float or a numeric string becomes the integer's string. Every cache read and write,
the deduplication step, and `subset_order` go through it.

**Guarding test.** `tests/test_distill_teacher.py::test_float_play_ids_hit_the_cache`.
