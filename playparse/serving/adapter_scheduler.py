"""Request-level adapter routing on the serving layer's continuous-batching scheduler.

WHAT CHANGES, AND WHAT DOES NOT
-------------------------------
The upstream ``Scheduler`` stays adapter-agnostic: it decides admission,
chunked prefill, preemption and batch composition exactly as before, and
batches freely mix adapters. :class:`AdapterScheduler` subclasses it and adds
four things, each at an existing method boundary:

1. **Requests carry an adapter.** :class:`AdapterRequest` adds ``adapter_id``
   (None = base model). An adapter that was never registered is a client error
   and is rejected at ``add_request``; one that is registered but not resident
   is not an error, the request waits.
2. **Admission waits for an adapter slot.** ``_can_admit`` first asks the pool
   whether the request's adapter is resident or loadable (a free slot, or an
   evictable one: refcount 0 and not pinned). If not, admission returns False
   and the upstream head-of-line rule makes the request wait, exactly as it
   waits for KV blocks. The slot is acquired at admission (loading the adapter
   on a miss) and released at retirement or recompute-preemption. A swapped
   request keeps its slot, because it resumes straight into the running batch
   without passing through admission again.
3. **Every forward pass gets an explicit per-row selection.** The upstream
   ``step()`` calls ``self.model.forward_varlen(tokens, meta, backend)`` with no
   adapter argument, and builds the batch internally. ``_tokens_for(req, q)`` is
   called exactly once per scheduled sequence, in batch order, just before that
   forward pass, so this class records each sequence's slot there. ``self.model``
   is a thin :class:`_RoutedModel` that turns the recorded per-sequence slots
   into ``row_slots = seq_slots[meta.batch_indices]`` and calls
   ``MultiLoRAModelGPU.forward_varlen(..., adapter=selection)``. It cross-checks
   the recording against ``meta`` (sequence count and each query length) and
   raises on any mismatch: a misaligned recording would apply one request's
   adapter to another's tokens.
4. **The prefix cache is adapter-keyed.** The three upstream methods that call
   the radix cache with token ids (``_can_admit``, ``_reuse_cached_prefix``,
   ``_cache_insert``) run inside ``cache.scope(req.adapter_id)``. Passing a plain
   ``RadixCache`` is refused, because it would share KV across adapters.

The hooks in (3) depend on upstream's internal call order, which is why the
routed model verifies it on every step. The clean upstream change (a
``_forward(batch, tokens, meta)`` method the subclass can override) is written
up in docs/issues/p5b-scheduler-forward-hook.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from playparse.serving._serving_path import ensure_serving_importable

ensure_serving_importable()

import contextlib  # noqa: E402

from serving.cache.radix import RadixCache  # noqa: E402
from serving.scheduler.scheduler import Request, RequestState, Scheduler, SchedulerConfig  # noqa: E402

from playparse.serving.adapter_pool import AdapterPool  # noqa: E402
from playparse.serving.adapter_radix import AdapterRadixCache  # noqa: E402
from playparse.serving.multi_lora import MultiLoRAModelGPU  # noqa: E402

__all__ = ["AdapterRequest", "AdapterScheduler", "AdapterRoutingError"]


class AdapterRoutingError(RuntimeError):
    """The per-sequence adapter record disagrees with the batch about to run."""


@dataclass
class AdapterRequest(Request):
    """A scheduler request that names its LoRA adapter (None = base model)."""

    adapter_id: str | None = None
    adapter_slot: int | None = field(default=None, repr=False)
    """Pool slot held while admitted; None when no reference is held."""


class _RoutedModel:
    """What the upstream scheduler sees as ``self.model``: adds the explicit selection."""

    def __init__(self, model: MultiLoRAModelGPU, scheduler: "AdapterScheduler"):
        self._model = model
        self._sched = scheduler
        self.device = model.device

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def forward_varlen(self, token_ids, meta, backend):
        record = self._sched._step_record
        self._sched._step_record = []
        if len(record) != meta.n_seqs:
            raise AdapterRoutingError(
                f"recorded {len(record)} sequences' adapters for a batch of {meta.n_seqs}; "
                "the upstream step() no longer calls _tokens_for once per scheduled sequence"
            )
        q_lens = meta.query_lens.tolist()
        for i, ((slot, q), ql) in enumerate(zip(record, q_lens)):
            if q != ql:
                raise AdapterRoutingError(
                    f"sequence {i}: recorded {q} tokens but the batch gives it {ql}; "
                    "adapter record and batch order disagree"
                )
        seq_slots = [slot for slot, _ in record]
        self._sched.last_seq_slots = seq_slots
        selection = self._model.selection_for(seq_slots, meta, [q for _, q in record])
        return self._model.forward_varlen(token_ids, meta, backend, adapter=selection)


class AdapterScheduler(Scheduler):
    """Continuous batching over mixed-adapter requests, backed by an :class:`AdapterPool`."""

    def __init__(
        self,
        model: MultiLoRAModelGPU,
        backend,
        allocator,
        config: SchedulerConfig | None = None,
        prefix_cache: RadixCache | None = None,
    ):
        if not isinstance(model, MultiLoRAModelGPU):
            raise TypeError("AdapterScheduler needs a MultiLoRAModelGPU (a model wired to an AdapterPool)")
        if prefix_cache is not None and not isinstance(prefix_cache, AdapterRadixCache):
            raise TypeError(
                "a plain RadixCache keys KV by tokens alone and would reuse one adapter's "
                "prefix KV for another adapter; pass an AdapterRadixCache"
            )
        super().__init__(_RoutedModel(model, self), backend, allocator, config, prefix_cache)
        self.lora_model = model
        self.pool: AdapterPool = model.pool
        self._step_record: list[tuple[int, int]] = []
        self.last_seq_slots: list[int] = []

    # -- admission ------------------------------------------------------------

    def add_request(self, req: Request) -> bool:
        aid = getattr(req, "adapter_id", None)
        if not self.pool.is_registered(aid):
            req.state = RequestState.FAILED
            req.error = f"unknown adapter {aid!r} (registered: {sorted(self.pool.registered)})"
            return False
        return super().add_request(req)

    def _cache_scope(self, req: Request):
        if self.prefix_cache is None:
            return contextlib.nullcontext()
        return self.prefix_cache.scope(getattr(req, "adapter_id", None))

    def _can_admit(self, req: Request) -> bool:
        # The adapter check runs FIRST: the KV check below may evict cached
        # blocks to make room, which should not happen for a request that cannot
        # be admitted anyway.
        if not self.pool.can_acquire(getattr(req, "adapter_id", None)):
            self.pool.stats.admission_waits += 1
            return False
        with self._cache_scope(req):
            return super()._can_admit(req)

    def _reuse_cached_prefix(self, req: Request) -> None:
        # Called exactly once per admission, right after _can_admit said yes.
        # Base-model requests (adapter None, or a plain upstream Request) hold no slot.
        aid = getattr(req, "adapter_id", None)
        if aid is not None:
            if req.adapter_slot is not None:
                raise AdapterRoutingError(
                    f"request {req.request_id} admitted while still holding slot {req.adapter_slot}"
                )
            req.adapter_slot = self.pool.acquire(aid)
        with self._cache_scope(req):
            super()._reuse_cached_prefix(req)

    def _cache_insert(self, req: Request) -> None:
        with self._cache_scope(req):
            super()._cache_insert(req)

    # -- the step ---------------------------------------------------------------

    def step(self):
        self._step_record = []
        return super().step()

    def _tokens_for(self, req: Request, q: int) -> list[int]:
        aid = getattr(req, "adapter_id", None)
        if aid is None:
            slot = -1
        else:
            slot = req.adapter_slot
            if slot is None or slot < 0 or self.pool.slot_adapter[slot] != aid:
                raise AdapterRoutingError(
                    f"request {req.request_id} wants adapter {aid!r} but holds slot {slot} "
                    f"({self.pool.slot_adapter[slot] if slot is not None and slot >= 0 else None!r})"
                )
        self._step_record.append((slot, q))
        return super()._tokens_for(req, q)

    # -- releasing the slot --------------------------------------------------------

    def _release_slot(self, req: Request) -> None:
        slot = getattr(req, "adapter_slot", None)
        if slot is not None:
            self.pool.release(slot)
            req.adapter_slot = None

    def _retire(self, req: Request) -> None:
        super()._retire(req)
        self._release_slot(req)

    def _preempt_recompute(self, req: Request) -> None:
        super()._preempt_recompute(req)
        # Back to the waiting queue: it re-acquires (possibly another slot) on readmission.
        self._release_slot(req)

    def snapshot(self) -> dict[str, Any]:
        snap = super().snapshot()
        snap.update({f"pool_{k}": v for k, v in self.pool.snapshot().items()})
        return snap
