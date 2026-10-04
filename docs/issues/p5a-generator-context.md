# P5a: an adapter chosen around a generator can silently not apply

**Status:** fixed with `generate_with_adapter`; pitfall documented by a test.

## What happened

The adapter for a request is held in a `ContextVar`, so concurrent requests on
different threads or asyncio tasks cannot see each other's choice. The engine's
`generate()` is a Python generator. A generator does not capture the context it
was created in. Its body runs in the context of whoever calls `next()`.

So this works, because the tokens are pulled inside the block:

```python
with model.use_adapter("playparse-v1"):
    tokens = list(generate(model, prompt, greedy))
```

But this silently runs the **base** model, with no error:

```python
with model.use_adapter("playparse-v1"):
    stream = generate(model, prompt, greedy)   # created inside
for tok in stream:                              # consumed outside
    ...
```

A streaming HTTP handler is exactly the second shape: it builds the generator
and hands it to the response object, which iterates later.

## How it was found

Reasoning about how the engine's `server.py` streams tokens, then confirming it
with a test on the tiny model.

## Root cause

`contextvars` and generators: PEP 568 (generator-local context) was never
adopted, so a generator sees the caller's context at each resume.

## Fix

`playparse.serving.lora_engine.generate_with_adapter(model, ids, sampler, adapter)`
sets the selection around every `next()` call, so the stream can be consumed
anywhere. The P5b serving layer should pass the adapter explicitly on every
forward call (`forward_varlen(..., adapter=...)`), which the API already supports,
rather than relying on an ambient block.

## Guarding test

`tests/test_p5a_tiny_oracle.py::test_generator_consumed_outside_block_falls_back_to_base`
asserts both halves: the lazy generator yields base tokens, and
`generate_with_adapter` yields adapter tokens.
