"""R4: a frontier Claude model, few-shot, through the Anthropic Messages API.

Which model is deferred (PRD decisions log #3), so `model` is a required parameter
and prices come from a table the caller can override. The client reads
`ANTHROPIC_API_KEY` (or any credential the SDK resolves) only when the predictor is
built without an injected client, so tests run with a mock and no key.

Prompt shape: the same system prompt and user-turn format as R1-R3 (via
`build_fewshot_messages`), with the few-shot turns before the real play. The system
prompt plus examples are identical for every play, so a `cache_control` breakpoint
on the last example turn makes that prefix cacheable: after the first call it is
billed at the cache-read rate instead of the full input rate. Note the API ignores
breakpoints on prefixes shorter than the model's minimum (512-4096 tokens depending on
the model); 8 short examples may fall under it, which `usage.cache_read_input_tokens`
will show as 0. R3-style retrieved examples change per play and are not cached.
"""
from __future__ import annotations

import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from playparse.eval.baselines.hf_predictor import Example, build_fewshot_messages, load_fewshot
from playparse.eval.harness import Prediction
from playparse.prompt import SYSTEM_PROMPT


@dataclass(frozen=True)
class Price:
    """USD per million tokens. Cache writes/reads default to 1.25x / 0.1x input."""

    input: float
    output: float
    cache_write: float | None = None
    cache_read: float | None = None

    def cost(self, usage: Mapping[str, int]) -> float:
        cw = self.cache_write if self.cache_write is not None else 1.25 * self.input
        cr = self.cache_read if self.cache_read is not None else 0.1 * self.input
        return (
            usage.get("input_tokens", 0) * self.input
            + usage.get("output_tokens", 0) * self.output
            + usage.get("cache_creation_input_tokens", 0) * cw
            + usage.get("cache_read_input_tokens", 0) * cr
        ) / 1e6


# First-party list prices (USD / MTok) as of 2026-06. Verify before quoting a cost;
# pass `prices=` to override.
DEFAULT_PRICES: dict[str, Price] = {
    "claude-opus-5": Price(5.0, 25.0),
    "claude-sonnet-5": Price(2.0, 10.0),
    "claude-haiku-4-5": Price(1.0, 5.0),
}


def _to_api_messages(msgs: list[dict[str, str]], n_examples: int, cache: bool) -> tuple[str, list[dict]]:
    """Split off the system prompt and mark the end of the shared few-shot prefix."""
    system = msgs[0]["content"]
    out: list[dict] = []
    for i, m in enumerate(msgs[1:]):
        block: dict[str, Any] = {"type": "text", "text": m["content"]}
        # Turns 0 .. 2n-1 are the examples; the last example's assistant turn ends the prefix.
        if cache and n_examples and i == 2 * n_examples - 1:
            block["cache_control"] = {"type": "ephemeral"}
        out.append({"role": m["role"], "content": [block]})
    return system, out


class AnthropicPredictor:
    def __init__(
        self,
        model: str,
        *,
        examples: Sequence[Example] | None = None,
        examples_fn: Callable[[dict], Sequence[Example]] | None = None,
        examples_id: str | None = None,
        prices: Mapping[str, Price] | None = None,
        max_tokens: int = 512,
        temperature: float | None = None,
        extra_create_kwargs: Mapping[str, Any] | None = None,
        cache_prefix: bool = True,
        max_attempts: int = 6,
        base_delay: float = 2.0,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        system: str = SYSTEM_PROMPT,
    ):
        """
        `temperature`: PRD §6 asks for temperature 0, but the newest Claude models
        reject sampling parameters (400), so it is only sent when set. Use
        `extra_create_kwargs` for model-specific options (thinking, effort).
        """
        if examples is not None and examples_fn is not None:
            raise ValueError("pass examples or examples_fn, not both")
        self.model = model
        self.name = f"r4-{model}"
        self.batch_size = 1
        self._examples = list(examples) if examples is not None else None
        self.examples_fn = examples_fn
        self.examples_id = examples_id
        self.prices = dict(DEFAULT_PRICES if prices is None else prices)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra = dict(extra_create_kwargs or {})
        self.cache_prefix = cache_prefix and examples_fn is None
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.sleep = sleep
        self.system = system
        if client is None:
            import anthropic

            if not os.environ.get("ANTHROPIC_API_KEY"):
                # The SDK can also resolve other credentials; warn rather than fail.
                import warnings

                warnings.warn("ANTHROPIC_API_KEY is not set; relying on other SDK credentials")
            client = anthropic.Anthropic(max_retries=0)  # retries are handled here, with logging
        self.client = client

    def config(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "examples": self.examples_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "extra": self.extra,
            "cache_prefix": self.cache_prefix,
        }

    def _examples_for(self, record: dict) -> Sequence[Example]:
        if self.examples_fn is not None:
            return self.examples_fn(record)
        return self._examples or ()

    def request_kwargs(self, record: dict) -> dict[str, Any]:
        examples = self._examples_for(record)
        msgs = build_fewshot_messages(record, examples, self.system)
        system, messages = _to_api_messages(msgs, len(examples), self.cache_prefix)
        kw: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": system}],
            "messages": messages,
            **self.extra,
        }
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        return kw

    def _call(self, kw: dict[str, Any]):
        retryable = _retryable_errors()
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.client.messages.create(**kw)
            except retryable as e:  # 429, 5xx, overloaded, connection errors
                status = getattr(e, "status_code", None)
                if status is not None and status < 500 and status not in (408, 409, 429):
                    raise
                if attempt == self.max_attempts:
                    raise
                delay = min(self.base_delay * 2 ** (attempt - 1), 60.0) + random.uniform(0, 1)
                self.sleep(delay)
        raise AssertionError("unreachable")

    def predict_batch(self, records: Sequence[dict]) -> list[Prediction]:
        out = []
        for rec in records:
            resp = self._call(self.request_kwargs(rec))
            text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text")
            u = resp.usage
            usage = {
                k: int(getattr(u, k, 0) or 0)
                for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
            }
            price = self.prices.get(self.model)
            if price is not None:
                usage["cost_usd"] = price.cost(usage)
            usage["stop_max_tokens"] = int(getattr(resp, "stop_reason", None) == "max_tokens")
            out.append(Prediction(text, usage))
        return out


def _retryable_errors() -> tuple[type[BaseException], ...]:
    try:
        import anthropic

        return (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError,
                anthropic.APIStatusError)
    except Exception:  # SDK missing: only possible with an injected client in tests
        return (ConnectionError,)


def make_r4(model: str, **kw) -> AnthropicPredictor:
    import hashlib
    import json

    examples = load_fewshot()
    eid = "fixed:fewshot_r2.json:" + hashlib.sha256(json.dumps(examples, sort_keys=True).encode()).hexdigest()[:12]
    return AnthropicPredictor(model, examples=examples, examples_id=eid, **kw)
