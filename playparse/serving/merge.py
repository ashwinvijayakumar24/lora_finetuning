"""Merged mode: fold a LoRA adapter into the base weights once, at load time.

For every targeted projection the engine weight ``W`` (``(out, in)``, applied as
``x @ W.T``) becomes::

    W' = cast(float32(W) + scale * B @ A)

The sum is formed in fp32 and cast back to the weight's own dtype exactly once.
Forming it in fp16 would round ``scale * B @ A`` to fp16 *before* adding it to a
much larger ``W``, losing the low bits of the update twice instead of once.

After merging, the weight dict has the same keys, shapes and dtypes as before, so
the unmodified engine (``LlamaModel`` / ``LlamaModelGPU``) runs it with exactly
the same kernels and FLOPs as the base model. That is why merged mode costs
nothing at inference (claim L1): there is no adapter left at runtime, only a
different set of numbers in the same tensors.

Tied embeddings. Llama 3.2 1B ties ``lm_head.weight`` to
``model.embed_tokens.weight`` (``engine/loader.py`` stores the same object under
both names). The adapter loader already refuses ``lm_head``/``embed_tokens``
targets, and merging only replaces the seven projections, so the alias survives
untouched.

Quantized bases. Merging into an already-quantized ``QuantWeight`` is refused.
The only honest way to do it is dequantize → add → re-quantize, which rounds the
weight twice: once when the base was quantized and again after the update, and
the second rounding can erase an update smaller than one quantization step.
:func:`merge_then_quantize` instead merges into the full-precision weights and
quantizes once, which is what a deployment would do anyway (the merged model is
just a new checkpoint).
"""
from __future__ import annotations

from typing import Any, MutableMapping

import numpy as np

from playparse.serving._engine_path import ensure_engine_importable
from playparse.serving.adapter import AdapterError, LoRAAdapter


def merge_adapter(
    weights: MutableMapping[str, Any],
    adapter: LoRAAdapter,
    *,
    inplace: bool = False,
) -> MutableMapping[str, Any]:
    """Return engine weights with ``adapter`` merged into its target projections.

    ``weights`` is an engine weight dict as returned by ``engine.loader.load_weights``
    (fp32 NumPy) or ``load_weights_gpu`` (fp16 torch). With ``inplace=False`` only
    the merged entries are new objects; untouched entries (embeddings, norms, the
    tied lm_head) are shared with the input, so the copy costs only the size of
    the targeted projections.
    """
    ensure_engine_importable()
    from engine.quant import QuantWeight

    out = weights if inplace else dict(weights)
    for key, t in adapter.layers.items():
        if key not in out:
            raise AdapterError(f"adapter targets {key!r}, which is not in the engine weights")
        w = out[key]
        if isinstance(w, QuantWeight):
            raise AdapterError(
                f"{key} is quantized ({w.mode}). Merging into a quantized weight would round it "
                "twice; merge into the fp16/fp32 weights and quantize afterwards "
                "(playparse.serving.merge.merge_then_quantize)."
            )
        if type(w).__name__ == "LoRALinear":
            raise AdapterError(f"{key} already carries unmerged adapters; merge into plain weights")
        if tuple(w.shape) != t.shape:
            raise AdapterError(f"{key}: weight shape {tuple(w.shape)} != adapter shape {t.shape}")
        out[key] = _merge_one(w, t.A, t.B, t.scale)
    return out


def _merge_one(w: Any, A: np.ndarray, B: np.ndarray, scale: float) -> Any:
    if isinstance(w, np.ndarray):
        merged = w.astype(np.float32) + np.float32(scale) * (B.astype(np.float32) @ A.astype(np.float32))
        return merged.astype(w.dtype, copy=False)

    import torch

    if isinstance(w, torch.Tensor):
        A_t = torch.from_numpy(np.ascontiguousarray(A, dtype=np.float32)).to(w.device)
        B_t = torch.from_numpy(np.ascontiguousarray(B, dtype=np.float32)).to(w.device)
        merged = w.float() + float(scale) * (B_t @ A_t)
        return merged.to(w.dtype)
    raise TypeError(f"cannot merge into weight of type {type(w).__name__}")


def quantize_engine_weights(
    weights: MutableMapping[str, Any],
    mode: str = "int8",
    group_size: int = 128,
) -> MutableMapping[str, Any]:
    """Quantize the seven per-layer projections of an in-memory engine weight dict.

    Mirrors the loop inside ``engine.loader.load_weights_gpu_quant``. That function
    only accepts a path on disk, so it cannot quantize weights that were merged in
    memory; this is the same loop over the engine's own quantizers and suffix list
    (see docs/issues/p5a-quant-loader-needs-disk.md).
    """
    ensure_engine_importable()
    from engine.loader import _QUANTIZED_SUFFIXES
    from engine.quant import QuantWeight, quantize_int4_group, quantize_int8_perchannel

    out = dict(weights)
    for name, w in weights.items():
        if not name.endswith(_QUANTIZED_SUFFIXES):
            continue
        if mode == "int8":
            q, s = quantize_int8_perchannel(w)
            out[name] = QuantWeight(q, s, "int8")
        elif mode == "int4":
            q, s = quantize_int4_group(w, group_size=group_size)
            out[name] = QuantWeight(q, s, "int4", group_size)
        else:
            raise ValueError(f"unknown quant mode {mode!r}")
    return out


def merge_then_quantize(
    weights: MutableMapping[str, Any],
    adapter: LoRAAdapter,
    mode: str = "int8",
    group_size: int = 128,
) -> MutableMapping[str, Any]:
    """Merge in full precision, then quantize once. Input must be unquantized torch weights."""
    return quantize_engine_weights(merge_adapter(weights, adapter), mode=mode, group_size=group_size)
