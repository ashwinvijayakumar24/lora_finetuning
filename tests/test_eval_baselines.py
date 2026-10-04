"""Prompt building, retrieval, the R4 client (mocked), and the CLI. No model weights."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from playparse.eval.baselines import frontier
from playparse.eval.baselines.frontier import AnthropicPredictor, Price
from playparse.eval.baselines.hf_predictor import build_fewshot_messages, load_fewshot
from playparse.eval.baselines.retrieval import TfidfRetriever, normalize_desc
from playparse.eval.harness import load_records, run_eval
from playparse.ffscore.schema import PlayLabel
from playparse.prompt import SYSTEM_PROMPT, build_messages

FIXTURE = Path(__file__).parent / "fixtures" / "eval_mini.jsonl"


# ------------------------------------------------------------------ few-shot prompts


def test_fewshot_fixture_is_valid_and_diverse():
    ex = load_fewshot()
    assert len(ex) == 8
    for e in ex:
        PlayLabel.from_json(e["label"])  # schema-valid
        assert e["label"] == PlayLabel.from_json(e["label"]).to_json()  # canonical form
        assert str(e["game_id"]).startswith(tuple(str(y) for y in range(2015, 2023)))  # train seasons only
    assert len({e["bucket"] for e in ex}) >= 6


def test_fewshot_messages_wrap_shared_prompt():
    rec = {"posteam": "PHI", "desc": "1-J.Hurts up the middle for 3 yards."}
    ex = load_fewshot()[:2]
    msgs = build_fewshot_messages(rec, ex)
    base = build_messages("PHI", rec["desc"])
    assert msgs[0] == base[0] and msgs[-1] == base[1]  # same system + query turn as R1
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert msgs[2]["content"] == ex[0]["label"]
    assert build_fewshot_messages(rec, []) == base  # zero-shot is exactly build_messages


# ------------------------------------------------------------------ retrieval


def _train():
    return [
        {"game_id": "g1", "play_id": 1, "posteam": "A", "desc": "(1:00) 1-A.Aa pass short left to 2-B.Bb to X 30 for 5 yards (3-C.Cc).", "label": "L1"},
        {"game_id": "g1", "play_id": 2, "posteam": "A", "desc": "(2:00) 4-D.Dd left tackle to X 20 for 3 yards (5-E.Ee).", "label": "L2"},
        {"game_id": "g2", "play_id": 1, "posteam": "B", "desc": "(3:00) 6-F.Ff sacked at Y 10 for -7 yards (7-G.Gg). FUMBLES (7-G.Gg), RECOVERED by Z-7-G.Gg at Y 8.", "label": "L3"},
        {"game_id": "g2", "play_id": 2, "posteam": "B", "desc": "TWO-POINT CONVERSION ATTEMPT. 6-F.Ff pass to 8-H.Hh is complete. ATTEMPT SUCCEEDS.", "label": "L4"},
        {"game_id": "g3", "play_id": 1, "posteam": "C", "desc": "(4:00) 9-I.Ii right guard to Z 40 for 12 yards (10-J.Jj).", "label": "L5"},
    ]


def test_normalize_removes_names_and_numbers():
    s = normalize_desc("(3:12) 1-J.Hurts pass short right to 11-A.Brown to DAL 22 for 14 yards (21-T.Diggs).")
    assert "Hurts" not in s and "DAL" not in s and "14" not in s and "PLAYER" in s and "SPOT" in s


def test_retriever_finds_same_shape_and_excludes_own_game():
    r = TfidfRetriever(_train(), k=2)
    q = {"game_id": "g9", "play_id": 1, "desc": "(9:00) 5-Q.Qq sacked at W 30 for -9 yards (1-R.Rr). FUMBLES (1-R.Rr), RECOVERED by V-1-R.Rr at W 25."}
    ex = r.examples_for(q)
    assert len(ex) == 2 and ex[-1]["label"] == "L3"  # most similar sits last, next to the query
    own = {"game_id": "g2", "play_id": 99, "desc": q["desc"]}
    assert all(e["label"] not in ("L3", "L4") for e in r.examples_for(own))
    assert "tfidf(k=2" in r.config_id()


def test_retriever_without_game_exclusion_still_skips_self():
    tr = _train()
    r = TfidfRetriever(tr, k=1, exclude_same_game=False)
    assert r.examples_for(tr[0])[0]["label"] != "L1"


def test_retriever_from_jsonl(tmp_path):
    p = tmp_path / "train.jsonl"
    p.write_text("".join(json.dumps(x) + "\n" for x in _train()))
    r = TfidfRetriever.from_jsonl(p, max_records=3, k=2)
    assert len(r.records) == 3 and "n=3" in r.config_id()


# ------------------------------------------------------------------ R4 (mocked API)


class FakeAPIError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class FakeMessages:
    def __init__(self, fail_first=0, status=529):
        self.calls = []
        self.fail_first = fail_first
        self.status = status

    def create(self, **kw):
        self.calls.append(kw)
        if len(self.calls) <= self.fail_first:
            raise FakeAPIError(self.status)
        cached = len(self.calls) > 1
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"nullified": true, "credits": []}')],
            usage=SimpleNamespace(
                input_tokens=50,
                output_tokens=12,
                cache_creation_input_tokens=0 if cached else 1000,
                cache_read_input_tokens=1000 if cached else 0,
            ),
            stop_reason="end_turn",
        )


@pytest.fixture
def fake_errors(monkeypatch):
    monkeypatch.setattr(frontier, "_retryable_errors", lambda: (FakeAPIError,))


def _r4(messages, **kw):
    return AnthropicPredictor(
        "test-model",
        examples=load_fewshot(),
        examples_id="fixed",
        prices={"test-model": Price(input=2.0, output=10.0)},
        client=SimpleNamespace(messages=messages),
        sleep=lambda s: None,
        **kw,
    )


def test_r4_request_shape_and_cache_breakpoint(fake_errors):
    msgs = FakeMessages()
    p = _r4(msgs)
    rec = {"posteam": "PHI", "desc": "1-J.Hurts up the middle for 3 yards."}
    p.predict_batch([rec])
    kw = msgs.calls[0]
    assert kw["model"] == "test-model" and "temperature" not in kw
    assert kw["system"][0]["text"] == SYSTEM_PROMPT
    m = kw["messages"]
    assert len(m) == 17 and m[-1]["role"] == "user" and "J.Hurts" in m[-1]["content"][0]["text"]
    marked = [i for i, x in enumerate(m) if "cache_control" in x["content"][0]]
    assert marked == [15]  # last example (assistant) turn ends the shared prefix
    assert m[15]["role"] == "assistant"


def test_r4_usage_to_cost(fake_errors):
    msgs = FakeMessages()
    p = _r4(msgs)
    recs = load_records(FIXTURE)[:3]
    out = p.predict_batch(recs)
    first, second = out[0].usage, out[1].usage
    # first call: 50*2 + 12*10 + 1000 cache-write tokens at 1.25*2
    assert first["cost_usd"] == pytest.approx((50 * 2 + 12 * 10 + 1000 * 2.5) / 1e6)
    # later calls read the prefix at 0.1x input
    assert second["cost_usd"] == pytest.approx((50 * 2 + 12 * 10 + 1000 * 0.2) / 1e6)
    res = run_eval(_r4(FakeMessages()), recs, rung="r4", n_boot=10)
    assert res["cost"]["basis"] == "api usage" and res["cost"]["usd_per_1k_plays"] > 0


def test_r4_retries_transient_errors(fake_errors):
    msgs = FakeMessages(fail_first=2, status=529)
    out = _r4(msgs).predict_batch([{"posteam": "A", "desc": "x"}])
    assert len(msgs.calls) == 3 and PlayLabel.from_json(out[0].text).nullified


def test_r4_does_not_retry_bad_request(fake_errors):
    msgs = FakeMessages(fail_first=5, status=400)
    with pytest.raises(FakeAPIError):
        _r4(msgs).predict_batch([{"posteam": "A", "desc": "x"}])
    assert len(msgs.calls) == 1


def test_r4_gives_up_after_max_attempts(fake_errors):
    msgs = FakeMessages(fail_first=99, status=429)
    with pytest.raises(FakeAPIError):
        _r4(msgs, max_attempts=3).predict_batch([{"posteam": "A", "desc": "x"}])
    assert len(msgs.calls) == 3


def test_r4_optional_temperature_and_extra(fake_errors):
    msgs = FakeMessages()
    _r4(msgs, temperature=0.0, extra_create_kwargs={"metadata": {"user_id": "eval"}}).predict_batch(
        [{"posteam": "A", "desc": "x"}]
    )
    assert msgs.calls[0]["temperature"] == 0.0 and msgs.calls[0]["metadata"] == {"user_id": "eval"}


# ------------------------------------------------------------------ CLI


def test_cli_r0(tmp_path, capsys):
    from playparse.eval.run import main

    out = tmp_path / "r0"
    main(["--rung", "r0", "--data", str(FIXTURE), "--out", str(out), "--n-boot", "20"])
    res = json.loads((out / "result.json").read_text())
    assert res["meta"]["rung"] == "r0" and res["overall"]["exact_match"]["point"] == 1.0
    assert "OVERALL" in capsys.readouterr().out


def test_cli_r4_requires_model(tmp_path):
    from playparse.eval.run import main

    with pytest.raises(SystemExit):
        main(["--rung", "r4", "--data", str(FIXTURE), "--out", str(tmp_path / "x")])
