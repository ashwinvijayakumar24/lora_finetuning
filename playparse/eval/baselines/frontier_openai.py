"""R4 with an OpenAI model: few-shot, through the Chat Completions API.

The owner has OpenAI API credits, so the frontier rung and the distillation teacher can
run on an OpenAI model instead of Claude (PRD decisions log #3). The prompt is exactly
R4's: the shared system prompt, the same 8 fixed few-shot turns (`fewshot_r2.json`),
then the play. OpenAI caches long identical prompt prefixes automatically (no cache
markers), and reports the cached share in `prompt_tokens_details.cached_tokens`.

GPT-5-family models are reasoning models: they spend hidden reasoning tokens before
answering, billed as output. `reasoning_effort` controls that and is part of the config
hash; reasoning tokens are recorded separately in the usage so their cost is visible.
They also reject a custom temperature, so it is only sent when set.

Prices are USD per million tokens, copied from https://developers.openai.com/api/docs/pricing
(standard tier) on 2026-10-10. Pass `prices=` to override.
"""
from __future__ import annotations

import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from playparse.eval.baselines.frontier import Price
from playparse.eval.baselines.hf_predictor import Example, build_fewshot_messages, load_fewshot
from playparse.eval.harness import Prediction
from playparse.prompt import SYSTEM_PROMPT


def _p(inp: float, cached: float, out: float) -> Price:
    # OpenAI charges no premium to write the cache: uncached input is billed at `input`.
    return Price(inp, out, cache_write=inp, cache_read=cached)


# Verified 2026-10-10 (standard tier).
OPENAI_PRICES: dict[str, Price] = {
    "gpt-5.5": _p(5.00, 0.50, 30.00),
    "gpt-5.4": _p(2.50, 0.25, 15.00),
    "gpt-5.4-mini": _p(0.75, 0.075, 4.50),
    "gpt-5.4-nano": _p(0.20, 0.02, 1.25),
    "gpt-5.2": _p(1.75, 0.175, 14.00),
    "gpt-5-mini": _p(0.25, 0.025, 2.00),
    "gpt-5-nano": _p(0.05, 0.005, 0.40),
    "gpt-4.1": _p(2.00, 0.50, 8.00),
    "gpt-4.1-mini": _p(0.40, 0.10, 1.60),
    "gpt-4.1-nano": _p(0.10, 0.025, 0.40),
}


def usage_from_response(resp: Any) -> dict[str, int]:
    """Map Chat Completions usage onto the harness's usage keys."""
    u = resp.usage
    prompt = int(getattr(u, "prompt_tokens", 0) or 0)
    cached = int(getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0)
    reasoning = int(getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0)
    return {
        "input_tokens": prompt - cached,
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": 0,
        "output_tokens": int(getattr(u, "completion_tokens", 0) or 0),  # includes reasoning
        "reasoning_tokens": reasoning,
    }


class OpenAIPredictor:
    def __init__(
        self,
        model: str,
        *,
        examples: Sequence[Example] | None = None,
        examples_id: str | None = None,
        prices: Mapping[str, Price] | None = None,
        max_completion_tokens: int = 4096,
        reasoning_effort: str | None = "low",
        temperature: float | None = None,
        n: int = 1,
        max_attempts: int = 6,
        base_delay: float = 2.0,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        system: str = SYSTEM_PROMPT,
    ):
        """
        `max_completion_tokens` covers reasoning plus the answer, so it is set well above
        the ~90-token answer. `n` > 1 asks for several samples in one call (the input is
        billed once), which the distillation teacher uses for its agreement check.
        """
        self.model = model
        self.name = f"r4-{model}"
        self.batch_size = 1
        self._examples = list(examples or ())
        self.examples_id = examples_id
        self.prices = dict(OPENAI_PRICES if prices is None else prices)
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.n = n
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.sleep = sleep
        self.system = system
        if client is None:
            from openai import OpenAI

            from playparse.env import load_dotenv

            load_dotenv()
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set (export it or put it in the repo's .env)")
            client = OpenAI(max_retries=0)  # retries are handled here
        self.client = client

    def config(self) -> dict[str, Any]:
        return {
            "provider": "openai",
            "model": self.model,
            "examples": self.examples_id,
            "max_completion_tokens": self.max_completion_tokens,
            "reasoning_effort": self.reasoning_effort,
            "temperature": self.temperature,
            "n": self.n,
        }

    def request_kwargs(self, record: dict) -> dict[str, Any]:
        msgs = build_fewshot_messages(record, self._examples, self.system)
        kw: dict[str, Any] = {"model": self.model, "messages": msgs,
                              "max_completion_tokens": self.max_completion_tokens}
        if self.reasoning_effort is not None:
            kw["reasoning_effort"] = self.reasoning_effort
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        if self.n != 1:
            kw["n"] = self.n
        return kw

    def _call(self, kw: dict[str, Any]):
        retryable = _retryable_errors()
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.client.chat.completions.create(**kw)
            except retryable as e:
                status = getattr(e, "status_code", None)
                if status is not None and status < 500 and status not in (408, 409, 429):
                    raise
                if attempt == self.max_attempts:
                    raise
                self.sleep(min(self.base_delay * 2 ** (attempt - 1), 60.0) + random.uniform(0, 1))
        raise AssertionError("unreachable")

    def cost(self, usage: Mapping[str, int]) -> float | None:
        price = self.prices.get(self.model)
        return None if price is None else price.cost(usage)

    def sample(self, record: dict) -> tuple[list[str], dict[str, Any]]:
        """All `n` choices for one play, plus usage (with cost when the price is known)."""
        resp = self._call(self.request_kwargs(record))
        texts = [c.message.content or "" for c in resp.choices]
        usage: dict[str, Any] = usage_from_response(resp)
        usage["stop_max_tokens"] = int(any(c.finish_reason == "length" for c in resp.choices))
        cost = self.cost(usage)
        if cost is not None:
            usage["cost_usd"] = cost
        return texts, usage

    def predict_batch(self, records: Sequence[dict]) -> list[Prediction]:
        out = []
        for rec in records:
            texts, usage = self.sample(rec)
            out.append(Prediction(texts[0], usage))
        return out


def _retryable_errors() -> tuple[type[BaseException], ...]:
    try:
        import openai

        return (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError,
                openai.APIStatusError)
    except Exception:
        return (ConnectionError,)


def make_r4_openai(model: str, **kw) -> OpenAIPredictor:
    import hashlib
    import json

    examples = load_fewshot()
    eid = "fixed:fewshot_r2.json:" + hashlib.sha256(json.dumps(examples, sort_keys=True).encode()).hexdigest()[:12]
    return OpenAIPredictor(model, examples=examples, examples_id=eid, **kw)
