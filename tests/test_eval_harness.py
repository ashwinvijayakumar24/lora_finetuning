import json
from pathlib import Path

import pytest

from playparse.eval.baselines.regex_parser import RegexPredictor
from playparse.eval.harness import (
    Prediction,
    Predictor,
    file_sha256,
    format_summary,
    load_records,
    run_eval,
)

FIXTURE = Path(__file__).parent / "fixtures" / "eval_mini.jsonl"


class GoldEcho:
    """Returns the gold label (perfect system), optionally failing after N calls."""

    name = "gold-echo"

    def __init__(self, batch_size=4, fail_after=None, usage=None, tag="a"):
        self.batch_size = batch_size
        self.fail_after = fail_after
        self.calls = 0
        self.seen = 0
        self.usage = usage
        self.tag = tag

    def config(self):
        return {"tag": self.tag}

    def predict_batch(self, records):
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise KeyboardInterrupt("simulated crash")
        self.calls += 1
        self.seen += len(records)
        if self.usage:
            return [Prediction(r["label"], dict(self.usage)) for r in records]
        return [r["label"] for r in records]


class Garbage(GoldEcho):
    name = "garbage"

    def predict_batch(self, records):
        return ["I think the answer is {broken" for _ in records]


@pytest.fixture
def records():
    return load_records(FIXTURE)


def test_protocol_conformance():
    assert isinstance(GoldEcho(), Predictor)
    assert isinstance(RegexPredictor(), Predictor)


def test_load_records_validates(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"game_id": "g", "desc": "x"}) + "\n")
    with pytest.raises(ValueError, match="missing keys"):
        load_records(bad)


def test_perfect_predictor_scores_one(records, tmp_path):
    res = run_eval(GoldEcho(), records, rung="test", out_dir=tmp_path, eval_path=FIXTURE, n_boot=100)
    o = res["overall"]
    assert o["n"] == len(records) and o["n_games"] == 4
    for m in ("valid_rate", "strict_valid_rate", "exact_match", "credit_f1"):
        assert o[m]["point"] == 1.0 and o[m]["lo"] == 1.0
    assert o["fp_mae"]["point"] == 0.0
    assert set(res["buckets"]) == {r["bucket"] for r in records}
    meta = res["meta"]
    assert meta["eval_sha256"] == file_sha256(FIXTURE)
    assert len(meta["eval_code_sha256"]) == 64 and meta["config_hash"]
    assert meta["rung"] == "test" and "platform" in meta["hardware"] and meta["date"]
    assert "git_sha" in meta
    # artifacts
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["overall"]["exact_match"]["point"] == 1.0
    rows = [json.loads(l) for l in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == len(records) and all(r["exact"] for r in rows)
    assert {"raw", "gold", "pred", "bucket", "latency_s", "tp", "fp", "fn"} <= set(rows[0])


def test_garbage_outputs_count_invalid_not_crash(records):
    res = run_eval(Garbage(), records, rung="test", n_boot=50)
    assert res["overall"]["valid_rate"]["point"] == 0.0
    assert res["overall"]["exact_match"]["point"] == 0.0
    assert res["overall"]["credit_recall"]["point"] == 0.0


def test_resume_skips_finished_plays(records, tmp_path):
    crashing = GoldEcho(batch_size=5, fail_after=2)
    with pytest.raises(KeyboardInterrupt):
        run_eval(crashing, records, rung="test", out_dir=tmp_path, n_boot=10)
    assert crashing.seen == 10
    lines = (tmp_path / "predictions.jsonl").read_text().splitlines()
    assert len(lines) == 10  # flushed per batch
    resumed = GoldEcho(batch_size=5)
    res = run_eval(resumed, records, rung="test", out_dir=tmp_path, n_boot=10)
    assert resumed.seen == len(records) - 10
    assert res["meta"]["n_resumed"] == 10
    assert res["overall"]["exact_match"]["point"] == 1.0
    # A second run over a finished file does no work at all.
    again = GoldEcho(batch_size=5)
    run_eval(again, records, rung="test", out_dir=tmp_path, n_boot=10)
    assert again.seen == 0


def test_resume_refuses_mixed_configs(records, tmp_path):
    run_eval(GoldEcho(tag="a"), records[:5], rung="test", out_dir=tmp_path, n_boot=10)
    with pytest.raises(ValueError, match="config"):
        run_eval(GoldEcho(tag="b"), records, rung="test", out_dir=tmp_path, n_boot=10)
    # resume=False starts over
    p = GoldEcho(tag="b")
    run_eval(p, records, rung="test", out_dir=tmp_path, n_boot=10, resume=False)
    assert p.seen == len(records)


def test_cost_from_api_usage(records):
    usage = {"input_tokens": 100, "output_tokens": 50, "cost_usd": 0.002}
    res = run_eval(GoldEcho(usage=usage), records, rung="test", n_boot=10)
    c = res["cost"]
    assert c["basis"] == "api usage"
    assert c["total_usd"] == pytest.approx(0.002 * len(records))
    assert c["usd_per_1k_plays"] == pytest.approx(2.0)
    assert c["usage_totals"]["input_tokens"] == 100 * len(records)


def test_cost_from_gpu_rate_and_latency(records):
    res = run_eval(GoldEcho(batch_size=8), records, rung="test", n_boot=10, gpu_usd_per_hour=3.6)
    lt = res["latency"]
    assert lt["p50_s"] >= 0 and lt["p99_s"] >= lt["p50_s"] and lt["batch_size"] == 8
    assert res["cost"]["total_usd"] == pytest.approx(lt["predict_seconds"] * 3.6 / 3600)


def test_duplicate_keys_rejected(records):
    with pytest.raises(ValueError, match="duplicate"):
        run_eval(GoldEcho(), records + records[:1], rung="test", n_boot=10)


def test_wrong_output_count_rejected(records):
    class Short(GoldEcho):
        def predict_batch(self, recs):
            return [r["label"] for r in recs][:-1]

    with pytest.raises(RuntimeError, match="outputs"):
        run_eval(Short(), records, rung="test", n_boot=10)


def test_regex_rung_on_fixture(records):
    res = run_eval(RegexPredictor(), records, rung="r0", n_boot=50)
    assert res["overall"]["exact_match"]["point"] == 1.0  # fixture labels were hand-checked
    assert "OVERALL" in format_summary(res)
