# P2: a non-blocking copy to MPS fed garbage labels to the loss

## What happened

Two smoke runs of the same 30-step configuration disagreed wildly. In one, the
validation loss at step 7 was 0.060. In the other, it was 2.627. The only
change between them was the evaluation batch size (8 versus 4), which should
not change the loss at all. Both runs also hit an out-of-memory error right
after that evaluation.

## How it was found

A probe evaluated the untrained base model on the same 16 validation examples
with different batch sizes:

| Eval batch size | Loss | Scored tokens |
|---|---|---|
| 1 | 0.6643 | 564 |
| 4 | 2.2834 | **2377** |
| 8 | 0.5340 | 564 |
| 16 | 0.5356 | 564 |

The number of scored tokens is fixed by the data: it is the number of
completion tokens, 564. Seeing 2377 meant the labels on the device were not the
labels we built.

## Root cause

`_to_device` in `loop.py` copied each batch with
`tensor.to(device, non_blocking=True)`. The batch is a temporary CPU tensor
created by `collate` a moment earlier, in ordinary (unpinned) memory. A
non-blocking copy to MPS returns before the data has actually been read. The
temporary CPU tensor was then freed, and its memory was reused for the next
batch before the copy finished. The GPU received a mix of old and new bytes.

The damage was silent. Garbage labels changed which positions were scored (the
2377), and garbage input ids changed the loss itself (the 0.66 at batch size 1).
The out-of-memory error was a side effect: the loss upcasts the logits at every
scored position to fp32, and 2377 positions × 128,256 vocabulary × 4 bytes is
about 1.2 GB that should have been about 0.3 GB.

Training was exposed to the same race, not only evaluation. The earlier smoke
numbers are therefore discarded.

## Fix

`_to_device` now does a blocking copy. The batches are a few kilobytes of token
ids, so overlapping the copy with compute would gain nothing. On CUDA, a
non-blocking copy from unpinned memory happens to be synchronous, which is why
this pattern is common in CUDA code and looks harmless.

After the fix, the probe gives 0.5355, 0.5363, 0.5356, 0.5356 for batch sizes 1,
4, 8, 16 (differences are bf16 rounding), with 564 scored tokens every time.

## Lesson

A loss that changes with the evaluation batch size is a bug, never noise. The
same holds for the number of scored tokens, which is cheap to check.

## Guarding tests

Both run by default on any machine with MPS and are skipped elsewhere:

- `tests/test_train_loop.py::test_eval_loss_independent_of_batch_size_on_mps`:
  failed before the fix (wrong token count even at batch size 1), passes after.
- `tests/test_train_loop.py::test_accumulated_token_count_and_loss_match_cpu_on_mps`:
  the accumulation path on MPS matches CPU.
