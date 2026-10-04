"""An adapter-keyed radix prefix cache (claim L6).

THE BUG THIS PREVENTS
---------------------
The serving layer's ``RadixCache`` keys cached KV blocks by token ids alone.
That is correct for one model. With LoRA adapters it is not: the K and V stored
for a prompt depend on the adapter, because the adapter changes ``k_proj`` and
``v_proj`` directly and changes every hidden state from layer 1 onward even when
it only targets the MLP. Two requests with the same prompt under adapters A and
B produce different KV. A token-only key would hand B the blocks A computed:
B's output would be fluent, plausible, and wrong, and nothing would raise.

So the key must be ``(adapter_id, tokens)``. Base-model requests (adapter None)
are their own namespace too: base KV is no more valid for an adapter than
another adapter's KV is.

HOW, WITHOUT EDITING THE CACHE
------------------------------
The trie compares token ids and never uses them for anything else (they are not
embedded or bounds-checked). So each namespace gets a disjoint range of the
integer line: token ``t`` under namespace code ``c`` is keyed as
``t + c * 2**32``. Code 0 is the base model, whose keys are the raw ids, so a
base-only workload behaves exactly like the upstream cache. Vocabularies are far
below ``2**32``, so two namespaces can never produce the same key, and the
trie's own block-granularity and last-token rules apply unchanged inside each
namespace. Eviction stays global and LRU across all namespaces, which is right:
they share one block allocator.

The scheduler calls ``match``/``acquire``/``insert`` with raw token ids and does
not know about adapters. :class:`~playparse.serving.adapter_scheduler.AdapterScheduler`
wraps each of those calls in ``with cache.scope(req.adapter_id):``. A call made
outside any scope RAISES instead of guessing a namespace, so a future upstream
code path that touches the cache without going through those three hooks fails
loudly rather than silently sharing KV across adapters.

AUDIT: CROSS-ADAPTER HITS ARE COUNTED, NOT ASSUMED ZERO
-------------------------------------------------------
Independently of the key encoding, the cache records which adapter produced
each cached block (at insert) and checks every block it hands out (at acquire)
against the requesting adapter. Mismatches are counted in
``cross_adapter_hits``. With keying on it is zero by construction; with the
fault-injection switch ``adapter_keyed=False`` (the token-only key) it is the
number of times the bug fired. The L6 claim is earned on that counter staying
at zero AND on outputs matching unbatched references, not on the absence of an
error.
"""
from __future__ import annotations

import contextlib
from typing import Iterator, Sequence

from playparse.serving._serving_path import ensure_serving_importable

ensure_serving_importable()

from serving.cache.radix import MatchResult, RadixCache, RadixNode  # noqa: E402

__all__ = ["AdapterRadixCache", "NAMESPACE_STRIDE"]

NAMESPACE_STRIDE = 1 << 32
_UNSCOPED = object()


class AdapterRadixCache(RadixCache):
    """``RadixCache`` whose keys include the adapter id. Use inside ``scope(adapter_id)``."""

    def __init__(self, allocator, *, adapter_keyed: bool = True, **kwargs):
        super().__init__(allocator, **kwargs)
        self.adapter_keyed = adapter_keyed
        """False = FAULT INJECTION ONLY: token-only keys, i.e. the bug L6 is about."""
        self._codes: dict[str | None, int] = {None: 0}
        self._scope: list[str | None] = []
        self._depth = 0                       # >0 while inside one of our public calls
        self._producer: dict[int, str | None] = {}
        self.cross_adapter_hits = 0
        self.cross_adapter_blocks = 0

    # -- scoping --------------------------------------------------------------

    @contextlib.contextmanager
    def scope(self, adapter_id: str | None) -> Iterator["AdapterRadixCache"]:
        """Every match/acquire/insert inside the block is keyed under ``adapter_id``."""
        self._scope.append(adapter_id)
        try:
            yield self
        finally:
            self._scope.pop()

    def _current(self):
        if not self._scope:
            raise RuntimeError(
                "AdapterRadixCache used outside scope(adapter_id). Refusing to guess the "
                "adapter: a token-only lookup could hand one adapter's KV to another."
            )
        return self._scope[-1]

    def namespace_code(self, adapter_id: str | None) -> int:
        code = self._codes.get(adapter_id)
        if code is None:
            code = len(self._codes)
            self._codes[adapter_id] = code
        return code

    def _encode(self, token_ids: Sequence[int]) -> Sequence[int]:
        aid = self._current()
        if not self.adapter_keyed:
            return token_ids
        code = self.namespace_code(aid)
        if code == 0:
            return token_ids
        off = code * NAMESPACE_STRIDE
        return [t + off for t in token_ids]

    @contextlib.contextmanager
    def _entered(self):
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1

    # -- the three keyed entry points ---------------------------------------------
    # RadixCache.acquire calls self.match internally; _depth stops the key from
    # being encoded twice on that nested call.

    def match(self, token_ids: Sequence[int]) -> MatchResult:
        if self._depth:
            return super().match(token_ids)
        with self._entered():
            return super().match(self._encode(token_ids))

    def acquire(self, token_ids: Sequence[int]) -> MatchResult:
        if self._depth:
            return super().acquire(token_ids)
        aid = self._current()
        with self._entered():
            res = super().acquire(self._encode(token_ids))
        wrong = [b for b in res.block_ids if self._producer.get(b, aid) != aid]
        if wrong:
            self.cross_adapter_hits += 1
            self.cross_adapter_blocks += len(wrong)
        return res

    def insert(self, token_ids: Sequence[int], block_ids: Sequence[int]) -> int:
        if self._depth:
            return super().insert(token_ids, block_ids)
        aid = self._current()
        with self._entered():
            n = super().insert(self._encode(token_ids), block_ids)
        for b in block_ids:
            if b in self._by_block and b not in self._producer:
                self._producer[b] = aid
        return n

    # -- bookkeeping that the namespaces touch ------------------------------------

    def _remove(self, node: RadixNode) -> None:
        self._producer.pop(node.block_id, None)
        super()._remove(node)

    def snapshot(self):
        snap = super().snapshot()
        snap["adapter_keyed"] = self.adapter_keyed
        snap["namespaces"] = len(self._codes)
        snap["cross_adapter_hits"] = self.cross_adapter_hits
        snap["cross_adapter_blocks"] = self.cross_adapter_blocks
        return snap
