"""Teacher orchestration with a mock teacher: caching/resume, budget caps, R8/R9 files.

The mock teacher's behavior is fixed per play by ``play_id % 7``:

    0, 1  correct on every sample
    2     wrong yardage on every sample (credits the yard line)
    3     hallucinated player name on every sample
    4     sample 0 is not JSON, samples 1-2 correct
    5     sample 0 correct, sample 2 disagrees (drops a credit)
    6     consistently wrong but plausible: credits the tackler, whose name and the
          yardage both appear in desc, so every check passes. This is what keeps the
          filter's precision below 1.

So per 7 plays: R9 keeps modes {0, 1, 6} -> precision 2/3. R8 keeps every play whose
sample 0 parses (all but mode 4) -> correct are {0, 1, 5} -> precision 3/6.
"""
from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from playparse.distill import (
    Budget,
    FilterConfig,
    TeacherBatch,
    TeacherCache,
    TeacherProtocolError,
    Usage,
    build_training_sets,
    label_dataset,
    play_key,
    precision_report,
    subset_order,
)
from playparse.distill.teacher import CACHE_FILE, LEDGER_FILE, load_ledger
from playparse.ffscore.schema import Credit, PlayLabel

COST_PER_CALL = 0.01


def make_records(n: int, game: str = "2023_01_PHI_DAL") -> list[dict]:
    recs = []
    for pid in range(1, n + 1):
        y = 3 + pid % 17
        line = 50 - y
        if pid % 2:
            desc = (f"(10:00) 26-S.Barkley up the middle to PHI {line} for {y} yards (90-D.Smith).")
            label = PlayLabel(False, (Credit("S.Barkley", "rush_yds", y),))
        else:
            desc = (f"(3:12) (Shotgun) 1-J.Hurts pass short right to 11-A.Brown to DAL {line} "
                    f"for {y} yards (21-T.Diggs).")
            label = PlayLabel(False, (Credit("J.Hurts", "pass_yds", y), Credit("A.Brown", "rec", 1),
                                      Credit("A.Brown", "rec_yds", y)))
        recs.append({"game_id": game, "play_id": pid, "season": 2023, "week": 1,
                     "season_type": "REG", "posteam": "PHI", "desc": desc,
                     "bucket": "normal", "label": label.to_json()})
    return recs


class MockTeacher:
    """Knows the right answers (like a strong teacher would) but is wrong in set ways."""

    def __init__(self, records, fail_after_calls: int | None = None, cost=COST_PER_CALL):
        self.truth = {play_key(r["game_id"], r["play_id"]): PlayLabel.from_json(r["label"]) for r in records}
        self.descs = {play_key(r["game_id"], r["play_id"]): r["desc"] for r in records}
        self.requests: list[tuple[list[dict], int]] = []
        self.fail_after_calls = fail_after_calls
        self.calls = 0
        self.cost = cost

    def sample(self, key, j):
        truth = self.truth[key]
        mode = int(key[1]) % 7
        yds = next(c.value for c in truth.credits if c.stat.endswith("_yds"))
        line = 50 - yds
        if mode == 2:
            wrong = PlayLabel(False, tuple(Credit(c.player, c.stat, line) if c.stat.endswith("_yds") else c
                                           for c in truth.credits))
            return wrong.to_json()
        if mode == 3:
            return PlayLabel(False, tuple(Credit("Z.Ghost", c.stat, c.value) for c in truth.credits)).to_json()
        if mode == 4 and j == 0:
            return "I think the answer is {nullified: false"
        if mode == 5 and j == 2:
            return PlayLabel(False, truth.credits[:1] + truth.credits[2:] if len(truth.credits) > 1 else ()).to_json()
        if mode == 6:
            tackler = "D.Smith" if "D.Smith" in self.descs[key] else "T.Diggs"
            return PlayLabel(False, (Credit(tackler, "rush_yds", yds),)).to_json()
        return truth.to_json()

    def __call__(self, records, n_samples):
        for r in records:
            assert "label" not in r, "teacher was shown ground truth"
            assert set(r) <= {"game_id", "play_id", "posteam", "desc"}
        n_calls = len(records) * n_samples
        if self.fail_after_calls is not None and self.calls + n_calls > self.fail_after_calls:
            raise ConnectionError("simulated network failure")
        self.calls += n_calls
        self.requests.append((records, n_samples))
        outs = []
        for r in records:
            key = play_key(r["game_id"], r["play_id"])
            # A resumed request asks for fewer samples; it gets the *last* indices'
            # behavior so mode 4/5 semantics stay tied to their sample index.
            idxs = list(range(3 - n_samples, 3)) if n_samples < 3 else list(range(n_samples))
            outs.append([self.sample(key, j) for j in idxs])
        return TeacherBatch(outs, Usage(calls=n_calls, input_tokens=100 * n_calls,
                                        output_tokens=40 * n_calls, cost_usd=self.cost * n_calls))


def _strip(records):
    return [{k: v for k, v in r.items() if k != "label"} for r in records]


# --------------------------------------------------------------------------- end to end


def test_full_pipeline_filter_decisions_and_precision(tmp_path):
    recs = make_records(70)
    teacher = MockTeacher(recs)
    rep = label_dataset(_strip(recs), teacher, tmp_path / "cache", n_samples=3, chunk_size=16)
    assert rep.stopped_reason is None and rep.n_labeled == 70 and rep.n_remaining == 0
    assert rep.usage_total.calls == 210
    assert rep.usage_total.cost_usd == pytest.approx(2.10)

    b = build_training_sets(_strip(recs), tmp_path / "cache", tmp_path / "out", k=3, sizes=(10, 30, 50, 1000))
    assert b.n_r9 == 30           # modes 0, 1, 6
    assert b.n_r8 == 60           # all but mode 4
    s = b.r9_stats
    assert s["dropped"] == 40
    assert s["first_failure_by_check"] == {"schema": 10, "yardage": 10, "names": 10, "agreement": 10}
    assert s["failed_by_check"]["agreement"] == 20  # mode 4 also fails agreement
    assert b.r8_stats["failed_by_check"] == {"schema": 10, "yardage": 0, "names": 0, "agreement": 0}

    p9 = precision_report(b.r9_path, recs)
    p8 = precision_report(b.r8_path, recs)
    assert p9["precision"] == pytest.approx(2 / 3)
    assert p8["precision"] == pytest.approx(0.5)
    assert p9["per_bucket"]["normal"]["n"] == 30


def test_training_file_format_and_no_ground_truth(tmp_path):
    recs = make_records(14)
    label_dataset(_strip(recs), MockTeacher(recs), tmp_path / "c", n_samples=3)
    b = build_training_sets(recs, tmp_path / "c", tmp_path / "out", k=3, sizes=())
    rows = [json.loads(line) for line in b.r9_path.read_text().splitlines()]
    gt = {play_key(r["game_id"], r["play_id"]): r["label"] for r in recs}
    assert rows
    for row in rows:
        assert set(row) == {"game_id", "play_id", "season", "week", "season_type", "posteam", "desc",
                            "bucket", "label", "label_source"}
        assert row["label_source"] == "teacher_filtered"
        PlayLabel.from_json(row["label"])
    # Mode 6 plays are in R9 with the *teacher's* wrong label, proving the label was
    # not copied from ground truth.
    mode6 = [r for r in rows if r["play_id"] % 7 == 6]
    assert mode6 and all(r["label"] != gt[play_key(r["game_id"], r["play_id"])] for r in mode6)
    r8 = [json.loads(line) for line in b.r8_path.read_text().splitlines()]
    assert {r["label_source"] for r in r8} == {"teacher_raw"}


class Tripwire(Mapping):
    def __init__(self, d):
        self._d = d

    def __getitem__(self, k):
        if k == "label":
            raise AssertionError("pipeline read ground truth")
        return self._d[k]

    def __iter__(self):
        return iter(k for k in self._d if k != "label")

    def __len__(self):
        return len(self._d) - 1


def test_pipeline_never_reads_ground_truth(tmp_path):
    recs = make_records(21)
    wired = [Tripwire(r) for r in recs]
    label_dataset(wired, MockTeacher(recs), tmp_path / "c", n_samples=3)
    b = build_training_sets(wired, tmp_path / "c", tmp_path / "out", k=3, sizes=())
    assert b.n_r9 == 9


def test_subsets_are_nested_and_oversized_skipped(tmp_path):
    recs = make_records(70)
    label_dataset(recs, MockTeacher(recs), tmp_path / "c", n_samples=3)
    b = build_training_sets(recs, tmp_path / "c", tmp_path / "out", k=3, sizes=(10, 20, 40, 100))
    assert set(b.subsets["r9"]) == {10, 20} and b.skipped_sizes["r9"] == [40, 100]
    assert set(b.subsets["r8"]) == {10, 20, 40} and b.skipped_sizes["r8"] == [100]
    small = b.subsets["r9"][10].read_text().splitlines()
    big = b.subsets["r9"][20].read_text().splitlines()
    assert big[:10] == small
    report = json.loads((tmp_path / "out" / "build_report.json").read_text())
    assert report["n_r9"] == 30 and report["skipped_sizes"]["r9"] == [40, 100]


def test_subset_order_independent_of_other_plays():
    recs = make_records(50)
    full = [r["play_id"] for r in subset_order(recs, seed=3)]
    half = [r["play_id"] for r in subset_order(recs[::2], seed=3)]
    assert [p for p in full if p in set(half)] == half
    assert full != [r["play_id"] for r in subset_order(recs, seed=4)]


# --------------------------------------------------------------------------- cache and resume


def test_resume_after_crash_only_requests_missing(tmp_path):
    recs = make_records(40)
    crashing = MockTeacher(recs, fail_after_calls=60)   # 2 chunks of 10 plays x 3 samples
    with pytest.raises(ConnectionError):
        label_dataset(recs, crashing, tmp_path / "c", n_samples=3, chunk_size=10)
    assert len(TeacherCache(tmp_path / "c" / CACHE_FILE)) == 60
    assert load_ledger(tmp_path / "c").calls == 60

    fresh = MockTeacher(recs)
    rep = label_dataset(recs, fresh, tmp_path / "c", n_samples=3, chunk_size=10)
    assert rep.n_already_cached == 20 and rep.n_labeled == 20
    asked = {r["play_id"] for batch, _ in fresh.requests for r in batch}
    assert asked == set(range(21, 41))
    assert rep.usage_total.calls == 120   # every sample paid for exactly once

    # A third run is a no-op.
    again = MockTeacher(recs)
    rep3 = label_dataset(recs, again, tmp_path / "c", n_samples=3)
    assert again.requests == [] and rep3.n_already_cached == 40


def test_partial_samples_are_topped_up(tmp_path):
    """R8 run first (k=1), then R9 (k=3) reuses sample 0 and asks only for 1 and 2."""
    recs = make_records(14)
    label_dataset(recs, MockTeacher(recs), tmp_path / "c", n_samples=1)
    t = MockTeacher(recs)
    label_dataset(recs, t, tmp_path / "c", n_samples=3)
    assert {n for _, n in t.requests} == {2}
    assert t.calls == 28
    b = build_training_sets(recs, tmp_path / "c", tmp_path / "out", k=3, sizes=())
    assert b.n_missing_samples == {"r8": 0, "r9": 0}


def test_torn_last_line_is_skipped(tmp_path):
    recs = make_records(7)
    label_dataset(recs, MockTeacher(recs), tmp_path / "c", n_samples=3)
    path = tmp_path / "c" / CACHE_FILE
    with open(path, "a") as f:
        f.write('{"game_id": "2023_01_PHI_DAL", "play_id": "8", "sample_id')   # crash mid-append
    cache = TeacherCache(path)
    assert cache.skipped_malformed == 1 and len(cache) == 21


def test_float_play_ids_hit_the_cache(tmp_path):
    recs = make_records(7)
    label_dataset(recs, MockTeacher(recs), tmp_path / "c", n_samples=3)
    as_float = [{**r, "play_id": float(r["play_id"])} for r in recs]
    t = MockTeacher(recs)
    rep = label_dataset(as_float, t, tmp_path / "c", n_samples=3)
    assert t.requests == [] and rep.n_already_cached == 7
    assert play_key("g", 55.0) == play_key("g", "55") == play_key("g", 55) == ("g", "55")


def test_teacher_id_namespaces_cache(tmp_path):
    recs = make_records(7)
    label_dataset(recs, MockTeacher(recs), tmp_path / "c", n_samples=1, teacher_id="model-a/prompt-1")
    t = MockTeacher(recs)
    label_dataset(recs, t, tmp_path / "c", n_samples=1, teacher_id="model-a/prompt-2")
    assert t.calls == 7  # different prompt -> nothing reused
    assert TeacherCache(tmp_path / "c" / CACHE_FILE, "model-a/prompt-2").skipped_other_teacher == 7


def test_wrong_shape_from_client_raises_but_spend_is_recorded(tmp_path):
    recs = make_records(5)

    def bad_client(records, n):
        return [["{}"]] * (len(records) - 1), Usage(calls=len(records) * n, cost_usd=0.5)

    with pytest.raises(TeacherProtocolError):
        label_dataset(recs, bad_client, tmp_path / "c", n_samples=3)
    assert load_ledger(tmp_path / "c").cost_usd == pytest.approx(0.5)
    assert not (tmp_path / "c" / CACHE_FILE).exists()


# --------------------------------------------------------------------------- budget


def test_max_calls_stops_cleanly_and_resumes(tmp_path):
    recs = make_records(30)
    t = MockTeacher(recs)
    rep = label_dataset(recs, t, tmp_path / "c", n_samples=3, chunk_size=8, budget=Budget(max_calls=50))
    assert rep.stopped_reason == "max_calls"
    assert rep.usage_total.calls == 48 <= 50       # 16 plays; a 17th would need 51 calls
    assert rep.n_labeled == 16 and rep.n_remaining == 14
    # The cap is cumulative: rerunning with the same cap spends nothing more.
    t2 = MockTeacher(recs)
    rep2 = label_dataset(recs, t2, tmp_path / "c", n_samples=3, budget=Budget(max_calls=50))
    assert t2.requests == [] and rep2.stopped_reason == "max_calls"
    # Raising the cap finishes the job.
    rep3 = label_dataset(recs, MockTeacher(recs), tmp_path / "c", n_samples=3, budget=Budget(max_calls=90))
    assert rep3.stopped_reason is None and rep3.usage_total.calls == 90


def test_max_usd_with_accurate_estimate_never_overspends(tmp_path):
    recs = make_records(30)
    rep = label_dataset(recs, MockTeacher(recs), tmp_path / "c", n_samples=3, chunk_size=7,
                        budget=Budget(max_usd=0.50, est_usd_per_call=COST_PER_CALL))
    assert rep.stopped_reason == "max_usd" and not rep.overran_usd
    assert rep.usage_total.cost_usd <= 0.50 + 1e-9
    assert rep.n_labeled == 16


def test_max_usd_with_low_estimate_overruns_by_at_most_one_chunk(tmp_path):
    recs = make_records(30)
    # The estimate is 5x too low, so the first chunk (projected $0.06) really costs $0.30.
    rep = label_dataset(recs, MockTeacher(recs, cost=0.05), tmp_path / "c", n_samples=3, chunk_size=2,
                        budget=Budget(max_usd=0.20, est_usd_per_call=0.01))
    assert rep.stopped_reason == "max_usd" and rep.overran_usd
    one_chunk = 2 * 3 * 0.05
    assert 0.20 < rep.usage_total.cost_usd <= 0.20 + one_chunk
    assert rep.n_labeled == 2
    # Resuming sees the overspend in the ledger and refuses to spend more.
    t = MockTeacher(recs, cost=0.05)
    rep2 = label_dataset(recs, t, tmp_path / "c", n_samples=3,
                         budget=Budget(max_usd=0.20, est_usd_per_call=0.01))
    assert t.requests == [] and rep2.stopped_reason == "max_usd"


def test_max_usd_switches_to_observed_cost_after_first_chunk(tmp_path):
    recs = make_records(30)
    rep = label_dataset(recs, MockTeacher(recs, cost=0.05), tmp_path / "c", n_samples=3, chunk_size=2,
                        budget=Budget(max_usd=0.50, est_usd_per_call=0.01))
    # Chunk 1: 2 plays ($0.30). Then the observed $0.05/call allows 1 more play ($0.15).
    assert rep.n_labeled == 3 and not rep.overran_usd
    assert rep.usage_total.cost_usd == pytest.approx(0.45)


def test_max_usd_requires_estimate():
    with pytest.raises(ValueError, match="est_usd_per_call"):
        Budget(max_usd=10.0)


def test_client_without_call_counts_is_charged_worst_case(tmp_path):
    recs = make_records(4)

    def quiet(records, n):
        return TeacherBatch([[r["desc"]] * n for r in records])  # usage left at zero

    rep = label_dataset(recs, quiet, tmp_path / "c", n_samples=2, budget=Budget(max_calls=8))
    assert rep.usage_total.calls == 8
    assert json.loads((tmp_path / "c" / LEDGER_FILE).read_text())["total"]["calls"] == 8


def test_r8_config_is_default_for_r8_file(tmp_path):
    recs = make_records(7)
    label_dataset(recs, MockTeacher(recs), tmp_path / "c", n_samples=3)
    b = build_training_sets(recs, tmp_path / "c", tmp_path / "o", k=3, sizes=(),
                            r9_config=FilterConfig(agreement=False))
    # Without agreement, mode 4 (bad sample 0) is still dropped by schema, and mode 5
    # (disagreeing sample 2) is now kept.
    assert b.n_r9 == 4
