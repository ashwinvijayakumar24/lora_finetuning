from types import SimpleNamespace as NS

from playparse.eval.baselines.frontier_openai import OPENAI_PRICES, OpenAIPredictor, usage_from_response

GOOD = '{"nullified":false,"credits":[{"player":"J.Conner","stat":"rush_yds","value":3}]}'


def _resp(texts, prompt=1000, cached=800, completion=120, reasoning=64, finish="stop"):
    return NS(choices=[NS(message=NS(content=t), finish_reason=finish) for t in texts],
              usage=NS(prompt_tokens=prompt, completion_tokens=completion,
                       prompt_tokens_details=NS(cached_tokens=cached),
                       completion_tokens_details=NS(reasoning_tokens=reasoning)))


class _Client:
    def __init__(self, resp):
        self.calls = []
        self.chat = NS(completions=NS(create=self._create))
        self._resp = resp

    def _create(self, **kw):
        self.calls.append(kw)
        return self._resp


REC = {"posteam": "ARI", "desc": "(15:00) 6-J.Conner up the middle to ARI 33 for 3 yards."}


def test_usage_mapping_splits_cached_and_reasoning():
    u = usage_from_response(_resp([GOOD]))
    assert u == {"input_tokens": 200, "cache_read_input_tokens": 800, "cache_creation_input_tokens": 0,
                 "output_tokens": 120, "reasoning_tokens": 64}


def test_cost_uses_verified_prices():
    p = OpenAIPredictor("gpt-5.4-mini", client=_Client(_resp([GOOD])))
    [pred] = p.predict_batch([REC])
    expected = (200 * 0.75 + 800 * 0.075 + 120 * 4.50) / 1e6
    assert abs(pred.usage["cost_usd"] - expected) < 1e-12
    assert pred.text == GOOD


def test_request_shape_and_samples():
    c = _Client(_resp([GOOD, GOOD, "bad"]))
    p = OpenAIPredictor("gpt-5.4-nano", client=c, n=3, examples=[], reasoning_effort="minimal")
    texts, usage = p.sample(REC)
    kw = c.calls[0]
    assert kw["n"] == 3 and kw["reasoning_effort"] == "minimal" and "temperature" not in kw
    assert kw["messages"][0]["role"] == "system" and "ARI 33" in kw["messages"][-1]["content"]
    assert texts == [GOOD, GOOD, "bad"] and usage["stop_max_tokens"] == 0


def test_unknown_model_has_no_cost():
    p = OpenAIPredictor("gpt-unknown", client=_Client(_resp([GOOD])))
    assert "cost_usd" not in p.predict_batch([REC])[0].usage
    assert "gpt-5.5" in OPENAI_PRICES
