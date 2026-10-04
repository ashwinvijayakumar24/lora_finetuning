"""The adapter pool: a fixed number of device-resident LoRA slots, LRU-managed.

WHY A POOL
----------
A multi-tenant server may know about hundreds of adapters but can only afford to
keep some of them in GPU memory. The pool is the paged-KV idea applied to
adapters:

    paged KV cache                        adapter pool
    -------------------------------       -----------------------------------
    fixed number of physical blocks       fixed number of device slots
    a sequence holds refs on blocks       a running request holds a ref on a slot
    a block with refcount 0 is reusable   a slot with refcount 0 is evictable
    no free block -> request waits        adapter not loadable -> request waits
    swap a block to host memory           evict an adapter (its host copy stays)

Every registered adapter always has a host copy (fp16, already scaled, already
in the device layout). "Evicting" therefore only forgets which slot held it; a
later request copies it back in (a *miss*, timed and counted).

THE DEVICE LAYOUT (what the BGMV kernel needs)
----------------------------------------------
For every adapted projection key (``model.layers.{i}.self_attn.q_proj.weight``
and so on) the pool owns two stacked tensors::

    A[key]: (n_slots + 1, max_rank, in)      same (r, in) layout as PEFT's lora_A
    B[key]: (n_slots + 1, out, max_rank)     same (out, r) layout as PEFT's lora_B,
                                             with the scale folded in

Slot ``s`` holds one adapter. Three details make one tensor serve every case:

* **Mixed ranks are zero-padded to max_rank.** An r=8 adapter in a pool built for
  r=64 fills ``A[s, :8]`` and ``B[s, :, :8]``; the rest stays zero, and zeros
  contribute exactly nothing to ``(x @ A.T) @ B.T``. The price is compute: the
  batched kernel always multiplies at ``max_rank``.
* **Projections an adapter does not target stay zero** for that slot, so the
  kernel needs no per-projection mask. This is also why a slot is zero-filled
  before every load: a stale matrix left over from the previous tenant on a
  projection the new adapter does not target would silently leak into its output.
* **Index ``n_slots`` is a permanently zero slot.** Rows that belong to base-model
  requests (``row_slot == -1``) are pointed at it, so they contribute zero
  through the same gather + matmul as every other row, with no branch.

The scale ``s = alpha / r`` (or ``alpha / sqrt(r)`` for rsLoRA) is folded into
``B`` in fp32 before the cast to fp16, exactly as P5a's ``LoRALinear.add`` does,
so the loop kernel here computes the same numbers as P5a's single-adapter path.

THE TWO KERNELS
---------------
**v1** loops over the distinct adapters present in the batch and runs two
small matmuls for each one's rows. It is P5a's reference algorithm on the pool
layout and the correctness reference for everything faster. Its cost grows with
the number of distinct adapters in the batch.

**v2** is Punica's design written in PyTorch. Decode rows (one per decoding
sequence, each possibly a different adapter) go through BGMV
(``lora_delta_bgmv``): gather each row's ``A`` and ``B`` from the stacks, then
two batched matmuls, at a cost that does not depend on how many distinct
adapters there are. BGMV materialises the gathered weights
(``rows x max_rank x (in + out)``), which is fine for a decode batch and
wasteful for a prefill chunk, whose rows all share one adapter. So prefill
chunks are handled segment by segment with one ordinary matmul pair each (the
SGMV idea). Rows are processed in chunks that keep the gathered buffer under
``chunk_bytes``. A fused CUDA/Triton kernel avoids the materialisation entirely;
that is the stretch "v3".

Both kernels take a :class:`BatchPlan`, built once per forward pass on the host
from the scheduler's per-sequence slots and query lengths, so neither needs a
device-to-host sync inside the 7-per-layer projection loop. Without a plan they
fall back to deriving everything from ``row_slots`` (``lora_delta_loop`` syncs
to find the distinct slots; ``lora_delta_bgmv`` treats every row as a decode row).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from playparse.serving._engine_path import ensure_engine_importable
from playparse.serving.adapter import (
    SUPPORTED_MODULES,
    AdapterError,
    LoRAAdapter,
    adapter_from_tensors,
    engine_to_peft_key,
    engine_weight_name,
    expected_linear_shapes,
)

ensure_engine_importable()

import torch  # noqa: E402

__all__ = [
    "ALL_TARGETS",
    "AdapterPool",
    "BatchPlan",
    "build_plan",
    "PoolFull",
    "PoolStats",
    "lora_delta_bgmv",
    "lora_delta_loop",
    "synthetic_adapter",
]

ALL_TARGETS = tuple(SUPPORTED_MODULES)
KERNELS = ("v1", "v2")


class PoolFull(RuntimeError):
    """No slot can take the adapter right now: every slot is pinned or in use."""


# --------------------------------------------------------------------------
# Kernels: pure functions over the stacked layout
# --------------------------------------------------------------------------

def lora_delta_loop(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    ranks: Sequence[int],
    row_slots: torch.Tensor,
) -> torch.Tensor | None:
    """v1: loop over the distinct slots in ``row_slots``; the correctness reference.

    ``x``: (tokens, in). ``A``: (S+1, R, in). ``B``: (S+1, out, R).
    ``ranks[s]`` is slot ``s``'s true rank; only ``A[s, :r]`` / ``B[s, :, :r]`` are
    used, so this does exactly P5a's arithmetic, ``(x @ A.T) @ B.T``, per adapter.
    Returns None when no row has an adapter (pure base batch).
    """
    out = None
    # torch.unique syncs with the host. Acceptable for the reference kernel; the
    # number of distinct slots is what this kernel's cost is proportional to.
    for slot in torch.unique(row_slots).tolist():
        if slot < 0:
            continue
        r = ranks[slot]
        if r == 0:
            continue
        rows = torch.nonzero(row_slots == slot).squeeze(1)
        a = A[slot, :r]                 # (r, in), contiguous
        b = B[slot, :, :r]              # (out, r), strided when r < max_rank
        if r < B.shape[2]:
            # Same contiguous (out, r) layout P5a's LoRALinear holds. A strided
            # operand also sends MPS matmul to a slow fallback path.
            b = b.contiguous()
        if out is None:
            out = torch.zeros((x.shape[0], B.shape[1]), dtype=x.dtype, device=x.device)
        out.index_add_(0, rows, (x[rows] @ a.T) @ b.T)
    return out


def lora_delta_bgmv(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    row_slots: torch.Tensor,
    chunk_rows: int,
) -> torch.Tensor:
    """v2: gather + batched matmul (BGMV). Cost independent of the number of adapters.

    For each row ``t`` with slot ``s = row_slots[t]``::

        delta[t] = (x[t] @ A[s].T) @ B[s].T

    computed for all rows at once as two ``torch.bmm`` calls over gathered
    weights. Base rows (``-1``) are redirected to the all-zero slot at index
    ``S = A.shape[0] - 1`` and come out as exact zeros. Rows are processed
    ``chunk_rows`` at a time to bound the gathered-weight buffer.
    """
    zero_slot = A.shape[0] - 1
    idx = torch.where(row_slots < 0, torch.full_like(row_slots, zero_slot), row_slots).long()
    n = x.shape[0]
    if n <= chunk_rows:
        return _bgmv_chunk(x, A, B, idx)
    return torch.cat(
        [_bgmv_chunk(x[i:i + chunk_rows], A, B, idx[i:i + chunk_rows]) for i in range(0, n, chunk_rows)],
        dim=0,
    )


def _bgmv_chunk(x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    a = A[idx]                                            # (t, R, in)   gather
    b = B[idx]                                            # (t, out, R)  gather
    xa = torch.bmm(x.unsqueeze(1), a.transpose(1, 2))     # (t, 1, R)    shrink
    return torch.bmm(xa, b.transpose(1, 2)).squeeze(1)    # (t, out)     expand


# --------------------------------------------------------------------------
# The batch plan: host-side structure, built once per forward pass
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BatchPlan:
    """Which rows use which adapter, computed ONCE per forward pass on the host.

    The scheduler already knows, in Python, every sequence's slot and query
    length. Turning that into index tensors once per step (instead of once per
    projection, 7 x layers times) removes all device->host syncs from the
    kernels, and it exposes the one structural fact that matters most for cost:

    * a **decode** sequence contributes one row;
    * a **prefill chunk** contributes a contiguous run of rows that all use the
      SAME adapter.

    ``groups``: per distinct adapter slot, the row indices that use it (v1).
    ``decode_rows``/``decode_slots``: adapter rows from decode sequences (v2 BGMV).
    ``segments``: ``(start, end, slot)`` for prefill chunks under an adapter (v2 SGMV).
    Base-model rows appear nowhere: they contribute exactly zero.
    """

    groups: tuple[tuple[int, torch.Tensor], ...]
    decode_rows: torch.Tensor | None
    decode_slots: torch.Tensor | None
    segments: tuple[tuple[int, int, int], ...]
    n_rows: int


def build_plan(seq_slots: Sequence[int], query_lens: Sequence[int], device) -> BatchPlan:
    rows_by_slot: dict[int, list[int]] = {}
    dec_rows: list[int] = []
    dec_slots: list[int] = []
    segments: list[tuple[int, int, int]] = []
    start = 0
    for slot, q in zip(seq_slots, query_lens):
        end = start + q
        if slot >= 0:
            rows_by_slot.setdefault(slot, []).extend(range(start, end))
            if q == 1:
                dec_rows.append(start)
                dec_slots.append(slot)
            else:
                segments.append((start, end, slot))
        start = end

    def t(v):
        return torch.tensor(v, dtype=torch.long, device=device)

    return BatchPlan(
        groups=tuple((s, t(r)) for s, r in sorted(rows_by_slot.items())),
        decode_rows=t(dec_rows) if dec_rows else None,
        decode_slots=t(dec_slots) if dec_slots else None,
        segments=tuple(segments),
        n_rows=start,
    )


def _loop_planned(x, A, B, ranks, plan: BatchPlan) -> torch.Tensor:
    """v1 with the plan: one pair of matmuls per distinct adapter, no host syncs."""
    out = torch.zeros((x.shape[0], B.shape[1]), dtype=x.dtype, device=x.device)
    for slot, rows in plan.groups:
        r = ranks[slot]
        if r == 0:
            continue
        b = B[slot, :, :r]
        if r < B.shape[2]:
            b = b.contiguous()
        out.index_add_(0, rows, (x[rows] @ A[slot, :r].T) @ b.T)
    return out


def _bgmv_sgmv_planned(x, A, B, ranks, plan: BatchPlan, chunk_rows: int) -> torch.Tensor:
    """v2 with the plan: BGMV over decode rows, one matmul pair per prefill segment.

    This is Punica's split. Decode rows each need a different adapter, so they
    are gathered and multiplied in one batched matmul (BGMV) whatever the number
    of distinct adapters. A prefill chunk's rows all share one adapter, so
    gathering a private copy of the weights for every one of its rows (what pure
    BGMV does) would move ``rows x rank x (in + out)`` bytes to do the work of a
    single matmul; instead each chunk is one ordinary matmul pair (the
    segmented-GMV idea, SGMV, written as a loop over the step's few prefill chunks).
    """
    out = torch.zeros((x.shape[0], B.shape[1]), dtype=x.dtype, device=x.device)
    if plan.decode_rows is not None:
        d = lora_delta_bgmv(x[plan.decode_rows], A, B, plan.decode_slots, chunk_rows)
        out.index_copy_(0, plan.decode_rows, d)
    for start, end, slot in plan.segments:
        r = ranks[slot]
        if r == 0:
            continue
        b = B[slot, :, :r]
        if r < B.shape[2]:
            b = b.contiguous()
        out[start:end] = (x[start:end] @ A[slot, :r].T) @ b.T
    return out


# --------------------------------------------------------------------------
# Synthetic adapters (scale tests, benchmarks)
# --------------------------------------------------------------------------

def synthetic_adapter(
    config: Mapping[str, Any],
    r: int,
    seed: int,
    *,
    name: str | None = None,
    b_std: float = 0.02,
    alpha: float | None = None,
    target_modules: Sequence[str] = ALL_TARGETS,
) -> LoRAAdapter:
    """A seeded random LoRA adapter with the model's shapes (B != 0, so it changes outputs).

    Synthetic adapters are how the S-LoRA paper load-tests too: latency and
    memory do not depend on the values, and N = 256 real adapters do not exist.
    """
    rng = np.random.default_rng(seed)
    shapes = expected_linear_shapes(config)
    tensors = {}
    for layer in range(config["num_hidden_layers"]):
        for m in target_modules:
            out_dim, in_dim = shapes[m]
            key = engine_weight_name(layer, m)
            tensors[engine_to_peft_key(key, "A")] = (
                rng.uniform(-1, 1, (r, in_dim)) / np.sqrt(in_dim)).astype(np.float32)
            tensors[engine_to_peft_key(key, "B")] = (
                rng.standard_normal((out_dim, r)) * b_std).astype(np.float32)
    cfg = {"peft_type": "LORA", "r": r, "lora_alpha": alpha if alpha is not None else 2 * r,
           "bias": "none", "target_modules": list(target_modules)}
    return adapter_from_tensors(cfg, tensors, name=name or f"synthetic-r{r}-s{seed}", base_config=config)


# --------------------------------------------------------------------------
# The pool
# --------------------------------------------------------------------------

@dataclass
class PoolStats:
    """What the pool did. ``hits``/``misses`` count admissions, not forward passes."""

    hits: int = 0                 # adapter already resident when a request was admitted
    misses: int = 0               # adapter had to be copied in from host memory
    evictions: int = 0            # a resident adapter lost its slot to another
    loads: int = 0                # host -> device copies (misses + pins of cold adapters)
    load_seconds: float = 0.0     # wall time inside those copies (device-synchronised)
    load_bytes: int = 0
    admission_waits: int = 0      # scheduler steps in which a request waited for a slot

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["hit_rate"] = self.hit_rate
        d["mean_load_ms"] = 1e3 * self.load_seconds / self.loads if self.loads else 0.0
        return d


@dataclass
class _HostAdapter:
    """A registered adapter in host memory, already in the device layout (fp16, scaled)."""

    adapter_id: str
    rank: int
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]]   # key -> (A (r,in), B_scaled (out,r))

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes + b.nbytes for a, b in self.tensors.values())


class AdapterPool:
    """``n_slots`` device-resident adapter slots over a host registry of any size.

    Single-threaded, like the scheduler that drives it. Lifecycle of one request:

        pool.can_acquire(aid)   # admission probe, side-effect free
        slot = pool.acquire(aid)    # at admission: hit, or load into a free/LRU slot
        ...  forward passes use `slot` as the request's row slot ...
        pool.release(slot)      # at retirement or recompute-preemption

    A slot with a positive refcount, or one holding a pinned adapter, is never
    evicted. That is the whole safety argument: a running request's rows always
    index the adapter they were admitted with.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        n_slots: int,
        max_rank: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float16,
        target_modules: Sequence[str] = ALL_TARGETS,
        kernel: str = "v2",
        chunk_bytes: int = 64 << 20,
        pin_host_memory: bool | None = None,
    ):
        if n_slots < 1:
            raise ValueError(f"n_slots must be >= 1, got {n_slots}")
        if max_rank < 1:
            raise ValueError(f"max_rank must be >= 1, got {max_rank}")
        unknown = set(target_modules) - set(SUPPORTED_MODULES)
        if unknown:
            raise AdapterError(f"unsupported target modules {sorted(unknown)}")
        self.config = dict(config)
        self.n_slots = n_slots
        self.max_rank = max_rank
        self.device = torch.device(device)
        self.dtype = dtype
        self.target_modules = tuple(m for m in SUPPORTED_MODULES if m in set(target_modules))
        self.kernel = kernel
        self.chunk_bytes = chunk_bytes
        self.pin_host_memory = (self.device.type == "cuda") if pin_host_memory is None else pin_host_memory

        shapes = expected_linear_shapes(config)
        self.shapes: dict[str, tuple[int, int]] = {}
        self.A: dict[str, torch.Tensor] = {}
        self.B: dict[str, torch.Tensor] = {}
        for layer in range(config["num_hidden_layers"]):
            for m in self.target_modules:
                key = engine_weight_name(layer, m)
                out_dim, in_dim = shapes[m]
                self.shapes[key] = (out_dim, in_dim)
                # +1: the permanent zero slot that base rows gather from.
                self.A[key] = torch.zeros((n_slots + 1, max_rank, in_dim), dtype=dtype, device=self.device)
                self.B[key] = torch.zeros((n_slots + 1, out_dim, max_rank), dtype=dtype, device=self.device)

        self.slot_adapter: list[str | None] = [None] * n_slots
        self.slot_rank: list[int] = [0] * n_slots
        self.refcount: list[int] = [0] * n_slots
        self.last_used: list[int] = [0] * n_slots
        self.pinned: set[str] = set()
        self._resident: dict[str, int] = {}
        self._host: dict[str, _HostAdapter] = {}
        self._clock = 0
        self.stats = PoolStats()

    # -- registry (host memory) ---------------------------------------------

    def register(self, adapter: LoRAAdapter, adapter_id: str | None = None) -> str:
        """Add an adapter to the host registry. It is loaded onto the device on demand."""
        aid = adapter_id or adapter.name
        if aid in self._host:
            raise AdapterError(f"adapter {aid!r} is already registered")
        adapter.validate_against(self.config)
        if adapter.r > self.max_rank:
            raise AdapterError(
                f"adapter {aid!r} has rank {adapter.r} > pool max_rank {self.max_rank}; "
                "build the pool for the largest rank it must serve"
            )
        tensors = {}
        for key, t in adapter.layers.items():
            if key not in self.shapes:
                raise AdapterError(
                    f"adapter {aid!r} targets {key!r}, which this pool does not adapt "
                    f"(pool targets {self.target_modules})"
                )
            # Same conversion as P5a's LoRALinear.add: scale folded in fp32, one cast.
            b_scaled = (np.float32(t.scale) * t.B.astype(np.float32)).astype(np.float32)
            a_h = torch.from_numpy(np.ascontiguousarray(t.A, dtype=np.float32)).to(self.dtype)
            b_h = torch.from_numpy(np.ascontiguousarray(b_scaled)).to(self.dtype)
            if self.pin_host_memory:
                a_h, b_h = a_h.pin_memory(), b_h.pin_memory()
            tensors[key] = (a_h, b_h)
        self._host[aid] = _HostAdapter(aid, adapter.r, tensors)
        return aid

    def unregister(self, adapter_id: str) -> None:
        """Remove an adapter entirely. Refuses while a request holds it."""
        slot = self._resident.get(adapter_id)
        if slot is not None:
            if self.refcount[slot] > 0:
                raise RuntimeError(f"adapter {adapter_id!r} is in use by {self.refcount[slot]} request(s)")
            self._clear_slot(slot)
        self.pinned.discard(adapter_id)
        self._host.pop(adapter_id)

    def is_registered(self, adapter_id: str | None) -> bool:
        return adapter_id is None or adapter_id in self._host

    @property
    def registered(self) -> tuple[str, ...]:
        return tuple(self._host)

    def host_bytes(self, adapter_id: str) -> int:
        return self._host[adapter_id].nbytes

    # -- residency ------------------------------------------------------------

    def is_resident(self, adapter_id: str | None) -> bool:
        return adapter_id is None or adapter_id in self._resident

    def slot_of(self, adapter_id: str | None) -> int:
        """Slot index of a resident adapter; -1 for the base model."""
        if adapter_id is None:
            return -1
        return self._resident[adapter_id]

    def _victim(self) -> int | None:
        """An empty slot if there is one, else the least-recently-used evictable slot."""
        best = None
        for s in range(self.n_slots):
            aid = self.slot_adapter[s]
            if aid is None:
                return s
            if self.refcount[s] > 0 or aid in self.pinned:
                continue
            if best is None or self.last_used[s] < self.last_used[best]:
                best = s
        return best

    def can_acquire(self, adapter_id: str | None) -> bool:
        """Would :meth:`acquire` succeed now? Pure probe: no load, no counters."""
        if adapter_id is None or adapter_id in self._resident:
            return True
        if adapter_id not in self._host:
            return False
        return self._victim() is not None

    def acquire(self, adapter_id: str | None) -> int:
        """Take a reference on ``adapter_id``'s slot, loading it if needed. Returns the slot.

        Raises :class:`PoolFull` if the adapter is not resident and every slot is
        pinned or in use; callers that must not fail call :meth:`can_acquire` first
        and wait.
        """
        if adapter_id is None:
            return -1
        self._clock += 1
        slot = self._resident.get(adapter_id)
        if slot is not None:
            self.stats.hits += 1
        else:
            if adapter_id not in self._host:
                raise KeyError(f"adapter {adapter_id!r} is not registered")
            slot = self._victim()
            if slot is None:
                raise PoolFull(
                    f"cannot load {adapter_id!r}: all {self.n_slots} slots are pinned or in use"
                )
            self.stats.misses += 1
            self._load(adapter_id, slot)
        self.refcount[slot] += 1
        self.last_used[slot] = self._clock
        return slot

    def release(self, slot: int) -> None:
        """Drop one reference. The adapter stays resident (and LRU-ordered) until evicted.

        ``last_used`` is not refreshed here, for the same reason the radix cache
        does not refresh on release: retiring is the opposite of interest.
        """
        if slot < 0:
            return
        if self.refcount[slot] <= 0:
            raise RuntimeError(f"release of slot {slot} with refcount {self.refcount[slot]}")
        self.refcount[slot] -= 1

    def pin(self, adapter_id: str) -> int:
        """Make an adapter permanently resident (never evicted). Loads it if needed."""
        if adapter_id not in self._resident:
            slot = self._victim()
            if slot is None:
                raise PoolFull(f"cannot pin {adapter_id!r}: no evictable slot")
            self._load(adapter_id, slot)
        self.pinned.add(adapter_id)
        return self._resident[adapter_id]

    def unpin(self, adapter_id: str) -> None:
        self.pinned.discard(adapter_id)

    def _clear_slot(self, slot: int) -> None:
        old = self.slot_adapter[slot]
        if old is not None:
            del self._resident[old]
        for key in self.A:
            self.A[key][slot].zero_()
            self.B[key][slot].zero_()
        self.slot_adapter[slot] = None
        self.slot_rank[slot] = 0

    def _load(self, adapter_id: str, slot: int) -> None:
        """Copy an adapter from host memory into ``slot``, evicting the previous tenant."""
        assert self.refcount[slot] == 0, "never overwrite a slot a request is using"
        host = self._host[adapter_id]
        t0 = time.perf_counter()
        if self.slot_adapter[slot] is not None:
            self.stats.evictions += 1
        # Zero first: projections this adapter does not target must not keep the
        # previous tenant's matrices (see the module docstring).
        self._clear_slot(slot)
        r = host.rank
        non_blocking = self.pin_host_memory
        for key, (a_h, b_h) in host.tensors.items():
            self.A[key][slot, :r].copy_(a_h, non_blocking=non_blocking)
            self.B[key][slot, :, :r].copy_(b_h, non_blocking=non_blocking)
        _sync(self.device)
        self.stats.load_seconds += time.perf_counter() - t0
        self.stats.loads += 1
        self.stats.load_bytes += host.nbytes
        self.slot_adapter[slot] = adapter_id
        self.slot_rank[slot] = r
        self._resident[adapter_id] = slot
        self.last_used[slot] = self._clock

    # -- the math -------------------------------------------------------------

    def delta(
        self, key: str, x: torch.Tensor, row_slots: torch.Tensor, plan: "BatchPlan | None" = None,
    ) -> torch.Tensor | None:
        """Low-rank contribution for one projection over a mixed batch, via ``self.kernel``.

        With a :class:`BatchPlan` (what the serving path always passes) both
        kernels use host-side batch structure computed once per forward pass;
        without one they derive everything from ``row_slots`` on the device.
        """
        A, B = self.A[key], self.B[key]
        if self.kernel not in KERNELS:
            raise ValueError(f"unknown kernel {self.kernel!r}; expected one of {KERNELS}")
        out_dim, in_dim = self.shapes[key]
        chunk = max(1, self.chunk_bytes // (self.max_rank * (in_dim + out_dim) * A.element_size()))
        if plan is None:
            if row_slots.device != x.device:
                row_slots = row_slots.to(x.device)
            if self.kernel == "v1":
                return lora_delta_loop(x, A, B, self.slot_rank, row_slots)
            return lora_delta_bgmv(x, A, B, row_slots, chunk)
        if not plan.groups:
            return None
        if self.kernel == "v1":
            return _loop_planned(x, A, B, self.slot_rank, plan)
        return _bgmv_sgmv_planned(x, A, B, self.slot_rank, plan, chunk)

    # -- reporting -------------------------------------------------------------

    def slot_names(self) -> tuple[str, ...]:
        """Adapter id per slot (``""`` for an empty slot), for AdapterSelection.names."""
        return tuple(a or "" for a in self.slot_adapter)

    def device_bytes(self) -> int:
        """Bytes of the stacked slot tensors on the device (all slots incl. the zero slot)."""
        return sum(t.nbytes for t in self.A.values()) + sum(t.nbytes for t in self.B.values())

    def bytes_per_slot(self) -> int:
        return self.device_bytes() // (self.n_slots + 1)

    def snapshot(self) -> dict[str, Any]:
        d = self.stats.as_dict()
        d.update({
            "n_slots": self.n_slots,
            "max_rank": self.max_rank,
            "kernel": self.kernel,
            "registered": len(self._host),
            "resident": len(self._resident),
            "pinned": len(self.pinned),
            "slots_in_use": sum(1 for c in self.refcount if c > 0),
            "device_bytes": self.device_bytes(),
        })
        return d

    def check_invariants(self) -> None:
        """Structural consistency of slots, residency map and refcounts. Tests and debug."""
        for aid, slot in self._resident.items():
            if self.slot_adapter[slot] != aid:
                raise AssertionError(f"resident map says {aid!r} in slot {slot}, slot holds {self.slot_adapter[slot]!r}")
        for s, aid in enumerate(self.slot_adapter):
            if aid is None:
                if self.refcount[s]:
                    raise AssertionError(f"empty slot {s} has refcount {self.refcount[s]}")
            elif self._resident.get(aid) != s:
                raise AssertionError(f"slot {s} holds {aid!r} but the resident map disagrees")
            if self.refcount[s] < 0:
                raise AssertionError(f"slot {s} refcount {self.refcount[s]} < 0")
        for aid in self.pinned:
            if aid not in self._resident:
                raise AssertionError(f"pinned adapter {aid!r} is not resident")
        for key in self.A:
            if self.A[key][self.n_slots].count_nonzero() or self.B[key][self.n_slots].count_nonzero():
                raise AssertionError(f"zero slot of {key} is not zero")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
