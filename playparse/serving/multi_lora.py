"""Many adapters in one batch: the engine model wired to an :class:`AdapterPool`.

For one linear layer and a mixed batch the computation is (PRD §9.2)::

    y = x @ W.T                      shared base matmul, unchanged
      + (x @ A_i.T) @ B_i.T          per-row low-rank path, i = that row's adapter

HOW IT HOOKS IN (same seam as P5a, one level up)
------------------------------------------------
P5a established that every projection on the engine's torch path goes through
``engine.components_gpu.linear(x, w)``, which receives the weight object but not
the layer name, so the adapter has to ride on the weight object
(docs/issues/p5a-linear-no-module-identity.md). P5b does the same thing with a
different object:

1. :class:`PooledLoRALinear` replaces each adapted entry of the model's weight
   dict. It holds the base weight and a reference to the pool plus its own key,
   which is all the pool needs to find the stacked ``A``/``B`` for that
   projection.
2. :func:`install_multi_lora_linear` wraps whatever ``components_gpu.linear``
   currently is. For a ``PooledLoRALinear`` it computes the base matmul through
   the wrapped function (so int8/int4 bases still work) and adds the pool's
   delta; anything else is passed straight through. It composes with P5a's
   ``install_lora_linear`` in either order.
3. The per-row selection is P5a's :class:`AdapterSelection` in its per-row form:
   ``row_slots[t]`` is the pool slot of token row ``t`` (``-1`` = base model).
   :meth:`MultiLoRAModelGPU.forward_varlen` REQUIRES it as an explicit keyword
   on every call (there is no "inherit the ambient selection" default here, unlike
   P5a). Inside the call it travels to ``linear()`` in P5a's context variable,
   because ``linear()`` has no parameter to carry it; the context is set and reset
   around exactly one forward pass, so it cannot leak across requests or steps
   (docs/issues/p5a-generator-context.md).

Deriving ``row_slots`` from a batch is one line, and :func:`row_slots_for`
does it: ``seq_slots[meta.batch_indices]``, where ``batch_indices[t]`` is the
sequence that owns packed token ``t`` (built by the serving layer's
``build_batch_meta``).
"""
from __future__ import annotations

from typing import Any, Sequence

from playparse.serving._engine_path import ensure_engine_importable
from playparse.serving.adapter_pool import AdapterPool
from playparse.serving.lora_engine import (
    AdapterSelection,
    current_selection,
    linear_chain_contains,
    use_adapter,
)

ensure_engine_importable()

import torch  # noqa: E402

from engine import components_gpu as _components_gpu  # noqa: E402
from engine.model_gpu import LlamaModelGPU  # noqa: E402

__all__ = [
    "MultiLoRAModelGPU",
    "PooledLoRALinear",
    "install_multi_lora_linear",
    "multi_lora_linear",
    "row_slots_for",
    "uninstall_multi_lora_linear",
]


class PooledLoRALinear:
    """A weight-dict value: one base projection whose adapters live in an AdapterPool."""

    __slots__ = ("key", "base", "pool")

    def __init__(self, key: str, base: Any, pool: AdapterPool):
        if isinstance(base, PooledLoRALinear):
            raise TypeError("PooledLoRALinear cannot wrap another PooledLoRALinear")
        self.key = key
        self.base = base
        self.pool = pool

    def delta(self, x: torch.Tensor, selection: AdapterSelection | None) -> torch.Tensor | None:
        if selection is None:
            return None
        if not selection.is_per_row:
            raise TypeError(
                f"{self.key}: a pooled model takes AdapterSelection.per_row(names, row_slots); "
                f"got a single-adapter selection {selection.name!r}"
            )
        row_slots = selection.row_slots
        if row_slots.shape[0] != x.shape[0]:
            # A misaligned selection would apply some request's adapter to another
            # request's rows. Cheap to check, silent if not checked.
            raise ValueError(
                f"{self.key}: row_slots has {row_slots.shape[0]} entries for {x.shape[0]} token rows"
            )
        return self.pool.delta(self.key, x, row_slots)

    @property
    def T(self):
        raise RuntimeError(
            f"{self.key}: PooledLoRALinear reached `x @ w.T`, so components_gpu.linear has not "
            "been wrapped. Call install_multi_lora_linear() (MultiLoRAModelGPU does this)."
        )


# The linear() that was in place before install (the engine's, or P5a's lora_linear).
_PREV_LINEAR: Any = None


def multi_lora_linear(x: torch.Tensor, w: Any) -> torch.Tensor:
    """Drop-in for ``components_gpu.linear`` that understands PooledLoRALinear."""
    if type(w) is PooledLoRALinear:
        y = _PREV_LINEAR(x, w.base)
        d = w.delta(x, current_selection())
        return y if d is None else y + d
    return _PREV_LINEAR(x, w)


multi_lora_linear.next_linear = lambda: _PREV_LINEAR   # type: ignore[attr-defined]


def install_multi_lora_linear() -> None:
    """Wrap ``engine.components_gpu.linear``. Idempotent; behaviour-preserving for other weights.

    "Idempotent" checks the whole wrapper stack, not just its top: if P5a's
    ``lora_linear`` was installed on top of this one, re-installing must not
    make this wrapper delegate to it, or each would call the other forever
    (docs/issues/p5b-linear-wrapper-cycle.md).
    """
    global _PREV_LINEAR
    if linear_chain_contains(multi_lora_linear):
        return
    _PREV_LINEAR = _components_gpu.linear
    _components_gpu.linear = multi_lora_linear


def uninstall_multi_lora_linear() -> None:
    global _PREV_LINEAR
    if _components_gpu.linear is multi_lora_linear and _PREV_LINEAR is not None:
        _components_gpu.linear = _PREV_LINEAR
        _PREV_LINEAR = None


def row_slots_for(seq_slots: Sequence[int], meta, device) -> torch.Tensor:
    """Per-token-row slots from per-sequence slots: ``seq_slots[meta.batch_indices]``."""
    if len(seq_slots) != meta.n_seqs:
        raise ValueError(f"{len(seq_slots)} sequence slots for a batch of {meta.n_seqs} sequences")
    seq_t = torch.tensor(list(seq_slots), dtype=torch.long, device=device)
    return seq_t[meta.batch_indices.to(device=device, dtype=torch.long)]


class MultiLoRAModelGPU(LlamaModelGPU):
    """``LlamaModelGPU`` whose adapted projections draw per-row adapters from a pool.

    The serving path is :meth:`forward_varlen`; ``adapter=`` is mandatory.
    """

    def __init__(self, weights: dict, config: dict, pool: AdapterPool, device: str = "cuda:0", **kwargs):
        install_multi_lora_linear()
        # Shallow copy: the caller's dict (often shared with a reference model)
        # keeps its plain tensors. The tensors themselves are shared, not copied.
        weights = dict(weights)
        for key in pool.shapes:
            w = weights.get(key)
            if w is None:
                raise KeyError(f"pool adapts {key!r}, which is not in the model weights")
            weights[key] = PooledLoRALinear(key, w, pool)
        super().__init__(weights, config, device=device, **kwargs)
        self.pool = pool

    def selection_for(self, seq_slots: Sequence[int], meta) -> AdapterSelection | None:
        """The explicit selection for one batch, or None if every sequence is base-only."""
        if all(s < 0 for s in seq_slots):
            return None
        return AdapterSelection.per_row(self.pool.slot_names(), row_slots_for(seq_slots, meta, self.device))

    def forward_varlen(self, token_ids, meta, backend, *, adapter: AdapterSelection | None):
        """Batched forward. ``adapter`` is required: a per-row selection, or None for base."""
        if adapter is not None and not adapter.is_per_row:
            raise TypeError("MultiLoRAModelGPU.forward_varlen needs AdapterSelection.per_row(...) or None")
        with use_adapter(adapter):
            return super().forward_varlen(token_ids, meta, backend)

    def forward_seq_slots(self, token_ids, meta, backend, seq_slots: Sequence[int]):
        """Convenience: ``forward_varlen`` with the selection derived from per-sequence slots."""
        return self.forward_varlen(token_ids, meta, backend, adapter=self.selection_for(seq_slots, meta))
