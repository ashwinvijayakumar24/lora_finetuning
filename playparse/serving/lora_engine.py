"""Unmerged mode: run LoRA adapters beside the frozen base weights, chosen per call.

THE MATH
--------
For a projection with base weight ``W`` (``(out, in)``) and adapter ``(A, B, s)``::

    y = x @ W.T  +  s * (x @ A.T) @ B.T
        ^^^^^^^     ^^^^^^^^^^^^^^^^^^^
        unchanged   low-rank path: (tokens, in) -> (tokens, r) -> (tokens, out)

The low-rank path costs ``2 * tokens * r * (in + out)`` FLOPs against the base
matmul's ``2 * tokens * in * out``. For Llama 3.2 1B's q_proj (2048 x 2048) at
r = 16 that is 1.6% more arithmetic, but it is two extra small matmuls and an
add (three more kernel launches) per projection: 7 projections x 16 layers x 3
= 336 extra launches per token. At batch 1 on a GPU, decode is launch- and bandwidth-bound, so the
launches are what claim L3 measures. ``s`` is folded into ``B`` when the adapter
is registered so the hot path does no extra elementwise multiply.

HOW IT HOOKS INTO THE ENGINE WITHOUT EDITING IT
-----------------------------------------------
Every projection on the engine's torch path goes through one function,
``engine.components_gpu.linear(x, w)`` — the chokepoint the int8/int4 path
already uses. It receives the activation and the *weight object* but not the
layer or module name. So the adapter rides on the weight object:

1. :class:`LoRALinear` replaces a targeted entry in the model's weight dict. It
   holds the original weight (plain tensor *or* ``QuantWeight``) plus every
   registered adapter for that one projection.
2. :func:`install_lora_linear` swaps ``components_gpu.linear`` for
   :func:`lora_linear`, which handles ``LoRALinear`` and otherwise calls the
   original function. ``gqa_attention_gpu`` and ``swiglu_ffn_gpu`` look ``linear``
   up as a module global at call time, so they pick up the replacement with no
   other change. For plain tensors and ``QuantWeight`` the replacement does one
   extra type check and then calls the original, so the base path is unchanged.
3. Which adapter is applied comes from a :class:`AdapterSelection` held in a
   ``contextvars.ContextVar``. :class:`LoRAModelGPU` sets it around
   ``prefill`` / ``decode_step`` / ``forward_all`` / ``forward_varlen``. A context
   variable is per-thread and per-asyncio-task, so two requests handled
   concurrently cannot see each other's selection.

The NumPy reference engine has no chokepoint: ``engine/components.py`` writes
``x @ w.T`` inline. There, ``LoRALinear.T`` returns a small object that NumPy
hands back to via the reflected-operator protocol (``__array_ufunc__ = None``
makes ``ndarray.__matmul__`` return ``NotImplemented``, so Python calls our
``__rmatmul__``). That keeps the fp32 reference path usable as an oracle without
copying its forward pass.

THE P5b SEAM: PER-ROW ADAPTERS
------------------------------
On the serving layer's packed varlen batch (``engine/attention_backend.py``),
each token row may belong to a different request with a different adapter.
``AdapterSelection.per_row(names, row_slots)`` carries one slot index per token
row (``-1`` = base only). :meth:`LoRALinear._delta_rows` implements this as the
simple "v1" loop over the distinct adapters present; P5b replaces that one
method with gather + batched matmul (BGMV) without touching anything else. The
serving layer derives ``row_slots`` as ``seq_slots[meta.batch_indices]``.
"""
from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np

from playparse.serving._engine_path import ensure_engine_importable
from playparse.serving.adapter import AdapterError, LoRAAdapter

ensure_engine_importable()

import torch  # noqa: E402

from engine import components_gpu as _components_gpu  # noqa: E402
from engine.model import LlamaModel  # noqa: E402
from engine.model_gpu import LlamaModelGPU  # noqa: E402
from engine.quant import QuantWeight  # noqa: E402

__all__ = [
    "AdapterSelection",
    "LoRALinear",
    "LoRAModelGPU",
    "LoRALlamaModel",
    "current_selection",
    "install_lora_linear",
    "uninstall_lora_linear",
    "lora_linear",
    "use_adapter",
    "generate_with_adapter",
    "lora_param_bytes",
]


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AdapterSelection:
    """Which adapter(s) the next forward pass applies.

    Exactly one of two forms:

    * ``AdapterSelection.single("name")`` — every row uses one adapter (P5a).
    * ``AdapterSelection.per_row(names, row_slots)`` — ``row_slots[i]`` indexes
      ``names`` for token row ``i``; ``-1`` means base model only (P5b seam).
    """

    name: str | None = None
    names: tuple[str, ...] = ()
    row_slots: Any = None   # (tokens,) integer torch tensor or NumPy array

    @classmethod
    def single(cls, name: str) -> "AdapterSelection":
        return cls(name=name)

    @classmethod
    def per_row(cls, names: Sequence[str], row_slots: Any) -> "AdapterSelection":
        return cls(names=tuple(names), row_slots=row_slots)

    @property
    def is_per_row(self) -> bool:
        return self.row_slots is not None

    def adapter_names(self) -> tuple[str, ...]:
        return self.names if self.is_per_row else ((self.name,) if self.name else ())


_SELECTION: contextvars.ContextVar[AdapterSelection | None] = contextvars.ContextVar(
    "playparse_lora_selection", default=None
)


def current_selection() -> AdapterSelection | None:
    return _SELECTION.get()


@contextlib.contextmanager
def use_adapter(selection: "AdapterSelection | str | None") -> Iterator[None]:
    """Apply ``selection`` to every LoRA-aware linear inside the block.

    ``None`` selects the base model explicitly (useful inside an outer block).
    """
    if isinstance(selection, str):
        selection = AdapterSelection.single(selection)
    token = _SELECTION.set(selection)
    try:
        yield
    finally:
        _SELECTION.reset(token)


# --------------------------------------------------------------------------
# The weight wrapper
# --------------------------------------------------------------------------

class LoRALinear:
    """A weight-dict value: one base projection plus its registered adapters.

    ``adapters[name] = (A, B_scaled)`` where ``A`` is ``(r, in)`` and
    ``B_scaled = scale * B`` is ``(out, r)``. On the torch path both live on the
    base weight's device in fp16 (the engine's activation dtype); on the NumPy
    path they are fp32.
    """

    __slots__ = ("key", "base", "adapters")

    def __init__(self, key: str, base: Any):
        if isinstance(base, LoRALinear):
            raise TypeError("LoRALinear cannot wrap another LoRALinear")
        self.key = key
        self.base = base
        self.adapters: dict[str, tuple[Any, Any]] = {}

    # ---- registration -----------------------------------------------------
    @property
    def is_numpy(self) -> bool:
        return isinstance(self.base, np.ndarray)

    @property
    def device(self):
        if self.is_numpy:
            return None
        return self.base.q.device if isinstance(self.base, QuantWeight) else self.base.device

    @property
    def shape(self) -> tuple[int, int]:
        if isinstance(self.base, QuantWeight):
            out = self.base.q.shape[0]
            in_dim = self.base.q.shape[1] * (2 if self.base.mode == "int4" else 1)
            return int(out), int(in_dim)
        return tuple(int(d) for d in self.base.shape)

    def add(self, name: str, A: np.ndarray, B: np.ndarray, scale: float) -> None:
        if self.shape != (B.shape[0], A.shape[1]):
            raise AdapterError(f"{self.key}: adapter {name!r} shape does not fit base {self.shape}")
        B_scaled = (np.float32(scale) * B.astype(np.float32)).astype(np.float32)
        if self.is_numpy:
            self.adapters[name] = (A.astype(np.float32), B_scaled)
        else:
            dev = self.device
            self.adapters[name] = (
                torch.from_numpy(np.ascontiguousarray(A, dtype=np.float32)).to(dev, torch.float16),
                torch.from_numpy(np.ascontiguousarray(B_scaled)).to(dev, torch.float16),
            )

    def remove(self, name: str) -> None:
        self.adapters.pop(name, None)

    # ---- math -------------------------------------------------------------
    def delta(self, x: Any, selection: AdapterSelection | None) -> Any | None:
        """Low-rank contribution for ``x`` (``(tokens, in)``), or None if nothing applies."""
        if selection is None or not self.adapters:
            return None
        if selection.is_per_row:
            return self._delta_rows(x, selection)
        pair = self.adapters.get(selection.name)
        if pair is None:            # this adapter does not target this projection
            return None
        A, B = pair
        return (x @ A.T) @ B.T

    def _delta_rows(self, x: Any, selection: AdapterSelection) -> Any | None:
        """Per-row adapters, "v1": loop over the distinct adapters in the batch.

        P5b replaces this with gather + batched matmul. Kept deliberately simple
        here because it is the correctness reference that faster versions will be
        diffed against.
        """
        slots = selection.row_slots
        if self.is_numpy:
            slots = np.asarray(slots)
            out = None
            for slot in np.unique(slots):
                if slot < 0:
                    continue
                pair = self.adapters.get(selection.names[int(slot)])
                if pair is None:
                    continue
                A, B = pair
                rows = np.nonzero(slots == slot)[0]
                if out is None:
                    out = np.zeros((x.shape[0], B.shape[0]), dtype=np.result_type(x, B))
                out[rows] += (x[rows] @ A.T) @ B.T
            return out

        slots = torch.as_tensor(slots, device=x.device)
        out = None
        for slot in torch.unique(slots).tolist():   # host sync: fine for v1
            if slot < 0:
                continue
            pair = self.adapters.get(selection.names[slot])
            if pair is None:
                continue
            A, B = pair
            rows = torch.nonzero(slots == slot).squeeze(1)
            if out is None:
                out = torch.zeros((x.shape[0], B.shape[0]), dtype=x.dtype, device=x.device)
            out.index_add_(0, rows, (x[rows] @ A.T) @ B.T)
        return out

    # ---- NumPy reference path: `x @ w.T` inline in engine/components.py ---
    @property
    def T(self) -> "_TransposedLoRA":
        if not self.is_numpy:
            raise RuntimeError(
                f"{self.key}: LoRALinear reached `x @ w.T` on the torch path, which means "
                "components_gpu.linear has not been replaced. Call install_lora_linear() "
                "(LoRAModelGPU does this) before running a model with unmerged adapters."
            )
        return _TransposedLoRA(self)


class _TransposedLoRA:
    """What ``LoRALinear.T`` returns on the NumPy path so ``x @ w.T`` adds the adapter."""

    __slots__ = ("_w",)
    __array_ufunc__ = None   # makes ndarray.__matmul__ return NotImplemented -> __rmatmul__

    def __init__(self, w: LoRALinear):
        self._w = w

    def __rmatmul__(self, x: np.ndarray) -> np.ndarray:
        y = x @ self._w.base.T
        d = self._w.delta(x, _SELECTION.get())
        return y if d is None else y + d


# --------------------------------------------------------------------------
# The chokepoint replacement (torch path)
# --------------------------------------------------------------------------

# The engine's linear() as it was before install, i.e. the QuantWeight-aware
# matmul. None while not installed.
_BASE_LINEAR: Any = None


def lora_linear(x: torch.Tensor, w: Any) -> torch.Tensor:
    """Drop-in for ``engine.components_gpu.linear`` that understands LoRALinear."""
    if type(w) is LoRALinear:
        y = _BASE_LINEAR(x, w.base)
        d = w.delta(x, _SELECTION.get())
        return y if d is None else y + d
    return _BASE_LINEAR(x, w)


lora_linear.next_linear = lambda: _BASE_LINEAR   # type: ignore[attr-defined]


def linear_chain_contains(fn: Any) -> bool:
    """Is ``fn`` already one of the wrappers stacked on ``components_gpu.linear``?

    Wrappers (this module's ``lora_linear``, P5b's ``multi_lora_linear``) expose
    ``next_linear()``, the function they delegate to. Checking only the top of
    the stack is not enough: re-installing a wrapper that sits lower down would
    make it delegate to a wrapper that delegates back to it, an infinite
    recursion on the first forward pass (docs/issues/p5b-linear-wrapper-cycle.md).
    """
    f = _components_gpu.linear
    seen = 0
    while f is not None and seen < 16:
        if f is fn:
            return True
        nxt = getattr(f, "next_linear", None)
        f = nxt() if nxt is not None else None
        seen += 1
    return False


def install_lora_linear() -> None:
    """Replace ``engine.components_gpu.linear`` with :func:`lora_linear`.

    Idempotent, including when another wrapper has been stacked on top since
    (see :func:`linear_chain_contains`). Process-wide by nature (it replaces a
    module global), but behaviour-preserving for every weight that is not a
    LoRALinear: those go straight to the original function after one type check.
    """
    global _BASE_LINEAR
    if linear_chain_contains(lora_linear):
        return
    _BASE_LINEAR = _components_gpu.linear
    _components_gpu.linear = lora_linear


def uninstall_lora_linear() -> None:
    """Restore the engine's original ``linear``."""
    global _BASE_LINEAR
    if _components_gpu.linear is lora_linear and _BASE_LINEAR is not None:
        _components_gpu.linear = _BASE_LINEAR
        _BASE_LINEAR = None


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class _Inherit:
    def __repr__(self) -> str:
        return "INHERIT"


INHERIT = _Inherit()


class _LoRAModelMixin:
    """Adapter registry + per-call selection, shared by the NumPy and torch models."""

    weights: dict
    config: dict

    def _init_lora(self) -> None:
        # Shallow copy: load_adapter() swaps dict entries for LoRALinear wrappers,
        # and the caller's dict (often shared with a merged or base model) must
        # not change underneath them. Tensors themselves are shared, not copied.
        self.weights = dict(self.weights)
        self.adapters: dict[str, LoRAAdapter] = {}

    # ---- registry ---------------------------------------------------------
    def load_adapter(self, adapter: LoRAAdapter, name: str | None = None) -> str:
        """Register ``adapter`` under ``name`` (default ``adapter.name``). Returns the name."""
        name = name or adapter.name
        if name in self.adapters:
            raise AdapterError(f"adapter {name!r} is already loaded")
        adapter.validate_against(self.config)
        for key, t in adapter.layers.items():
            w = self.weights.get(key)
            if w is None:
                raise AdapterError(f"adapter targets {key!r}, which is not in the model weights")
            if not isinstance(w, LoRALinear):
                w = LoRALinear(key, w)
                self.weights[key] = w
            w.add(name, t.A, t.B, t.scale)
        self.adapters[name] = adapter
        return name

    def unload_adapter(self, name: str) -> None:
        """Remove an adapter; projections left with no adapters get their plain weight back."""
        adapter = self.adapters.pop(name)
        for key in adapter.layers:
            w = self.weights[key]
            w.remove(name)
            if not w.adapters:
                self.weights[key] = w.base

    @contextlib.contextmanager
    def use_adapter(self, adapter: "str | AdapterSelection | None") -> Iterator[None]:
        """``with model.use_adapter("a"): generate(model, ...)`` — selection for the block."""
        sel = self._resolve(adapter)
        with use_adapter(sel):
            yield

    def _resolve(self, adapter: Any) -> AdapterSelection | None:
        if adapter is None:
            return None
        sel = AdapterSelection.single(adapter) if isinstance(adapter, str) else adapter
        for n in sel.adapter_names():
            if n not in self.adapters:
                raise AdapterError(f"adapter {n!r} is not loaded (loaded: {sorted(self.adapters)})")
        return sel

    def _selected(self, adapter: Any):
        if adapter is INHERIT:
            return contextlib.nullcontext()
        return use_adapter(self._resolve(adapter))


class LoRAModelGPU(_LoRAModelMixin, LlamaModelGPU):
    """``LlamaModelGPU`` that can apply registered LoRA adapters per call.

    Every forward entry point takes ``adapter=``: a name, an
    :class:`AdapterSelection`, ``None`` for the base model, or (the default)
    inherit whatever an enclosing ``with model.use_adapter(...)`` selected.
    The engine's own ``generate()`` calls ``prefill``/``decode_step`` without
    keyword arguments, so wrap it in ``use_adapter`` to pick the adapter.
    """

    def __init__(self, weights: dict, config: dict, device: str = "cuda:0", **kwargs):
        install_lora_linear()
        super().__init__(weights, config, device=device, **kwargs)
        self._init_lora()

    def prefill(self, token_ids, kv_cache, adapter=INHERIT):
        with self._selected(adapter):
            return super().prefill(token_ids, kv_cache)

    def decode_step(self, token_id, kv_cache, adapter=INHERIT):
        with self._selected(adapter):
            return super().decode_step(token_id, kv_cache)

    def forward_all(self, token_ids, adapter=INHERIT):
        with self._selected(adapter):
            return super().forward_all(token_ids)

    def forward_varlen(self, token_ids, meta, backend, adapter=INHERIT):
        """Batched path. For per-request adapters pass
        ``AdapterSelection.per_row(names, seq_slots[meta.batch_indices])``."""
        with self._selected(adapter):
            return super().forward_varlen(token_ids, meta, backend)


class LoRALlamaModel(_LoRAModelMixin, LlamaModel):
    """NumPy fp32 reference ``LlamaModel`` with unmerged adapters (the oracle path)."""

    def __init__(self, weights: dict, config: dict):
        super().__init__(weights, config)
        self._init_lora()

    def forward(self, token_ids, adapter=INHERIT):
        with self._selected(adapter):
            return super().forward(token_ids)

    def prefill(self, token_ids, kv_cache, adapter=INHERIT):
        with self._selected(adapter):
            return super().prefill(token_ids, kv_cache)

    def decode_step(self, token_id, kv_cache, adapter=INHERIT):
        with self._selected(adapter):
            return super().decode_step(token_id, kv_cache)


def lora_param_bytes(model: _LoRAModelMixin, name: str) -> int:
    """Device bytes held by one registered adapter (for memory reporting)."""
    total = 0
    for key in model.adapters[name].layers:
        for t in model.weights[key].adapters[name]:
            total += t.nbytes if isinstance(t, np.ndarray) else t.element_size() * t.nelement()
    return total


def generate_with_adapter(model, token_ids, sampler_fn, adapter, **kwargs):
    """``engine.scheduler.generate`` with ``adapter`` applied to every step.

    ``generate`` is a generator, and a generator runs in the context of whoever
    calls ``next()`` on it. ``with model.use_adapter(a): for t in generate(...)``
    therefore works, but a generator created inside the block and consumed after
    it silently falls back to the base model. This wrapper sets the selection
    around each step, so it is safe to hand to a streaming consumer.
    """
    from engine.scheduler import generate

    selection = model._resolve(adapter)
    gen = generate(model, token_ids, sampler_fn, **kwargs)
    while True:
        with use_adapter(selection):
            try:
                tok = next(gen)
            except StopIteration:
                return
        yield tok
