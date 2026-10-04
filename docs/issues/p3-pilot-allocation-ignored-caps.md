# P3: the pilot's hard-bucket allocation ignored small buckets' caps

## What happened

The first version of `enriched_train` in `scripts/p3_pilot.py` gave each hard
bucket `remaining // buckets_left` examples, in a fixed order. When a bucket held
fewer plays than its share, it gave all it had. The shortfall then went only to
the buckets *after* it in the order, not evenly to all of them.

## Why it matters

The pilot is meant to have equal shares per hard bucket, so that no rare bucket is
favored by accident. With uneven caps, the bucket order would have silently
decided which buckets got extra data.

## How it was found

By the unit test `test_enriched_train_takes_all_laterals_and_balances`, which uses
a synthetic pool where one middle bucket (`two_point`, 50 plays) is smaller than
its share.

## Impact on the real pilot

None. In `train.jsonl`, every non-lateral hard bucket has at least 1,039 plays,
far more than the ~148 requested, so no cap is hit. The fixed function was
checked to select exactly the same 2,400 keys as the version used for the run
(the manifest in `results/p3_local_pilot/data_manifest.json` is unchanged).

## Fix

Water-filling: buckets smaller than the fair share are given everything first,
and the remainder is split equally among the rest. Sampling order and the random
number generator calls are unchanged when no bucket is capped.

## Guarding test

`tests/test_p3_pilot.py::test_enriched_train_takes_all_laterals_and_balances`.
