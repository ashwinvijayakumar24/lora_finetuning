# P5b: the serving layer's scheduler has no hook between "batch chosen" and "forward pass"

**Status:** worked around in PlayParse with a checked hook; minimal upstream patch proposed below.

## What happened

Per-request adapters need one thing from the scheduler: on every forward pass,
which adapter each sequence in the batch uses, in batch order. The serving
layer's `Scheduler.step()` (`serving/scheduler/scheduler.py:696-771`) chooses the
batch, grows block tables, builds `BatchMeta`, and calls

```python
logits = self.model.forward_varlen(tokens, meta, self.backend)
```

all inside one method, with the batch held in a local variable. There is no
method a subclass can override to see `batch` and `meta` together, and the
forward call has no way to carry extra per-sequence data.

## How it was found

Reading `step()` while designing request-level adapter routing, before writing
code.

## Root cause

The scheduler was built for one model with no per-request state beyond tokens.
Everything a forward pass needed was already in `BatchMeta`.

## Workaround (in PlayParse, no upstream edit)

`AdapterScheduler` (`playparse/serving/adapter_scheduler.py`):

1. overrides `_tokens_for(req, q)`, which `step()` calls exactly once per
   scheduled sequence, in batch order, just before building `BatchMeta`, and
   records `(slot, q)` there;
2. replaces `self.model` with a thin `_RoutedModel` whose `forward_varlen`
   consumes the record and calls `MultiLoRAModelGPU.forward_varlen(...,
   adapter=selection)`.

Because this depends on upstream's internal call order, the routed model
checks it on every step: the number of records must equal `meta.n_seqs` and
each recorded query length must equal `meta.query_lens[i]`. A mismatch raises
`AdapterRoutingError` rather than applying one request's adapter to another's
tokens. `_tokens_for` also checks that the request's slot still holds its
adapter.

## Proposed upstream patch (minimal)

Split the forward call out of `step()` into an overridable method that
receives the batch:

```python
# serving/scheduler/scheduler.py, inside step()
-        logits = self.model.forward_varlen(tokens, meta, self.backend)   # (n_seqs, vocab)
+        logits = self._forward(batch, tokens, meta)                      # (n_seqs, vocab)

+    def _forward(self, batch: list[Request], tokens, meta):
+        """One forward pass over `batch`. Override to pass per-request state (e.g. adapters)."""
+        return self.model.forward_varlen(tokens, meta, self.backend)
```

With that, `AdapterScheduler` overrides `_forward` and builds the selection
from `batch` directly; the `_tokens_for` record, the routed model and the
alignment checks all disappear.

## Guarding tests

- `tests/test_p5b_scheduler.py::test_missing_record_is_detected` (a subclass
  that skips the record must fail with `AdapterRoutingError`)
- `tests/test_p5b_scheduler.py::test_slot_holding_another_adapter_is_detected`
- The whole correctness gate in `tests/test_p5b_scheduler.py`; a mutation that
  reversed the recorded slots failed 9 of its tests.
