# P6: an eval artifact in percent would disable the bucket tolerance

**What happened.** The gate's bucket tolerance is in percentage points (default 1.0,
meaning 0.01 absolute), and it compares fractions. If the eval harness wrote metrics
as percents (`91.0` instead of `0.91`), the tolerance would be 100 times too loose on
that scale, and no bucket regression could ever fail. The valid-rate threshold would
also always pass.

**How it was found.** Writing the gate before the eval harness exists, with only a
description of its output ("exact_match, valid_rate, credit_f1, with CIs"). That
description does not say which scale the numbers use.

**Root cause.** The interface between two components built at the same time left the
unit of a number unstated.

**Fix.** `playparse/registry/gate.py::load_eval_summary` requires every rate to be in
[0, 1] and raises `EvalArtifactError` naming the field otherwise. The expected format
is documented in the module docstring and in `docs/phases/P6-registry.md`.

**Guarding test.** `tests/test_registry.py::test_gate_rejects_percent_scale_artifacts`.
