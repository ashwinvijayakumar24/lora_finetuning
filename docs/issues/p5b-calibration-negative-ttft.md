# P5b: the first SLO calibration measured a negative TTFT

**Status:** fixed; the SLO constructor's sanity check caught it.

## What happened

The goodput arm of `scripts/p5b_bench.py` anchors its SLO on the unloaded
latency of the base model: run a few requests one at a time, take the median
TTFT and time per output token, multiply by the serving layer's fixed
multipliers (x10 and x3).

The calibration requests were built with the same workload builder as the
loaded run, which stamps each request with a Poisson arrival time (seconds
from the start of the window). The calibration driver then dispatched each
request immediately, but TTFT is measured from the *intended* arrival time
(the coordinated-omission rule). A request intended for t = 4.1 s that got its
first token at t = 0.05 s reported a TTFT of about -4,057 ms.

## How it was found

`SLO.from_unloaded` refuses to build an SLO from a non-positive latency and
raised `refusing to anchor an SLO on ttft=-4056.7 ...` on the first local run.
Without that check the SLO would have been "TTFT below -40 s", no request
could meet it, and every goodput cell would have read 0.0.

This is the same failure the serving layer hit in job 11599377 (a negative
TTFT propagated into a negative SLO). Its driver has the same guard for the
same reason.

## Root cause

Two meanings of "arrival" in one field: the scheduled time in an open-loop run,
and "now" in a closed-loop calibration.

## Fix

Calibration requests have `arrival_s` reset to 0 (the start of their own
one-request run) before they are dispatched. The burst used to estimate
capacity already did this.

## Guarding test

`SLO.from_unloaded` itself, tested in
`tests/test_p5b_workload.py::test_slo_refuses_impossible_anchor`, plus
`test_ttft_is_measured_from_intended_arrival`.
