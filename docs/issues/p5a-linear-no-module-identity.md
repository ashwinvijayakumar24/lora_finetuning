# P5a: the engine's `linear()` does not know which layer it is serving

**Status:** worked around in PlayParse; worth an engine change before P5b's fast kernels.

## What happened

The PRD planned to add the unmerged LoRA path "inside `linear()` in
`engine/components_gpu.py`, the existing quantization chokepoint". That function
has the signature `linear(x, w)`. It receives the activation and the weight
object, but not the layer index or the projection name. An adapter is defined
per (layer, projection), so `linear()` alone cannot tell which `A` and `B` to
apply.

Two smaller facts make the picture complete:

- The NumPy reference engine (`engine/components.py`, used as the fp32 oracle)
  has no chokepoint at all. It writes `x @ w.T` inline in `gqa_attention` and
  `swiglu_ffn`.
- The torch path's LM head is `last @ w["lm_head.weight"].T` in `model_gpu.py`,
  outside `linear()`. An unmerged adapter on `lm_head` could not go through the
  chokepoint even with module identity. PlayParse rejects `lm_head` targets anyway.

## How it was found

Reading `components_gpu.py` and `model_gpu.py` while designing the unmerged
mode, before writing code.

## Root cause

The chokepoint was designed for weight quantization. Quantization is a property
of the weight itself, so the weight object was enough. LoRA is a property of
(weight, request), so it needs extra context.

## Fix (in PlayParse, no engine edit)

- The adapter rides on the weight object. `LoRALinear` replaces each targeted
  entry in the model's weight dict and holds the base weight (plain fp16 or a
  `QuantWeight`) plus every registered adapter for that projection.
- `install_lora_linear()` replaces the module global `engine.components_gpu.linear`
  with `lora_linear`. `gqa_attention_gpu` and `swiglu_ffn_gpu` look `linear` up at
  call time, so they pick it up. For any weight that is not a `LoRALinear`,
  `lora_linear` does one type check and calls the original function.
- The request's adapter choice travels in a `ContextVar`, set by `LoRAModelGPU`
  around each forward call.
- On the NumPy path, `LoRALinear.T` returns an object with
  `__array_ufunc__ = None` and `__rmatmul__`. NumPy then defers `x @ w.T` to that
  object, which adds the low-rank term. No engine code is copied.

Costs of this approach, stated plainly:

- The replacement is process-wide. It is behaviour-preserving for non-LoRA
  weights, and the benchmark measures its cost (`base_patched` arm), but two
  extensions that both replace `linear` must be installed in a known order.
- If a `LoRALinear` reaches the torch path *without* the replacement installed,
  `x @ w.T` would fail confusingly. `LoRALinear.T` raises a `RuntimeError` that
  names the missing call instead.

## Suggested engine change (for the owner, not done here)

Pass an optional `name` (or `(layer_idx, proj)`) through to `linear()` from
`gqa_attention_gpu` / `swiglu_ffn_gpu`, and route the LM head through `linear()`.
That turns the global replacement into an ordinary hook and lets a P5b BGMV
kernel index a stacked adapter tensor by layer directly.

## Guarding tests

- `tests/test_p5a_lora_math.py::test_install_is_idempotent_and_reversible`
- `tests/test_p5a_lora_math.py::test_plain_weights_bit_identical_through_replacement`
- `tests/test_p5a_lora_math.py::test_torch_transpose_without_install_fails_loudly`
- `tests/test_p5a_lora_math.py::test_numpy_unmerged_matches_formula`
