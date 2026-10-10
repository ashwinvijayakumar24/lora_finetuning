"""Registry, eval gate, resolver, and CLI. No model needed: adapters are tiny fake dirs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from playparse.registry import (
    EvalArtifactError,
    GateConfig,
    NoCurrentVersion,
    Registry,
    RegistryError,
    evaluate_gate,
    resolve,
)
from playparse.registry import _io
from playparse.registry.__main__ import main as cli_main

EVAL_SHA = "a" * 64
BUCKETS = ("normal", "penalty_nullified", "penalty_stands", "fumble", "interception",
           "lateral", "challenge", "two_point", "td")


def make_eval(em=0.90, valid=0.995, buckets=None, sha=EVAL_SHA, n=2000, bucket_n=200):
    """An eval artifact in the shape the eval harness writes (see gate.py docstring)."""
    b = {name: {"n": bucket_n, "exact_match": em, "valid_rate": valid, "credit_f1": em}
         for name in BUCKETS}
    for name, v in (buckets or {}).items():
        if isinstance(v, dict):
            b[name].update(v)
        else:
            b[name]["exact_match"] = v
    return {
        "metadata": {"eval_file_sha256": sha, "git_sha": "deadbeef"},
        "overall": {"n": n, "exact_match": em, "valid_rate": valid, "credit_f1": em,
                    "exact_match_ci": [em - 0.01, em + 0.01]},
        "buckets": b,
    }


def make_adapter(root: Path, tag: str) -> Path:
    d = root / f"adapter_{tag}"
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text(json.dumps({"r": 16, "tag": tag}))
    (d / "adapter_model.safetensors").write_bytes(b"fake-weights-" + tag.encode())
    return d


@pytest.fixture
def reg(tmp_path):
    return Registry(tmp_path / "registry")


def register(reg, tmp_path, tag, **eval_kw):
    return reg.register("pp", make_adapter(tmp_path / "src", tag), make_eval(**eval_kw),
                        train_config={"lr": 2e-4, "r": 16},
                        train_data=hashlib.sha256(tag.encode()).hexdigest())


# --------------------------------------------------------------------------- gate


def test_gate_first_version_passes_absolute_minimums():
    d = evaluate_gate(make_eval(em=0.5), None, GateConfig(min_exact_match=0.4))
    assert d.promote, d.summary()
    d = evaluate_gate(make_eval(em=0.3), None, GateConfig(min_exact_match=0.4))
    assert not d.promote
    assert not d.check("min_exact_match").passed


def test_gate_tie_promotes_and_regression_rejects():
    assert evaluate_gate(make_eval(em=0.90), make_eval(em=0.90)).promote
    d = evaluate_gate(make_eval(em=0.89), make_eval(em=0.90))
    assert not d.promote
    assert [c.name for c in d.checks if not c.passed] == ["overall_exact_match"]


def test_gate_single_bucket_regression_rejects_despite_better_overall():
    cand = make_eval(em=0.93, buckets={"lateral": 0.50})
    cur = make_eval(em=0.90, buckets={"lateral": 0.60})
    d = evaluate_gate(cand, cur)
    assert not d.promote
    assert d.check("overall_exact_match").passed
    bucket = d.check("bucket_regression")
    assert not bucket.passed and "lateral" in bucket.reason


def test_gate_bucket_drop_within_tolerance_passes():
    cand = make_eval(em=0.91, buckets={"fumble": 0.795})
    cur = make_eval(em=0.90, buckets={"fumble": 0.80})
    assert evaluate_gate(cand, cur, GateConfig(bucket_tolerance_pts=1.0)).promote
    assert not evaluate_gate(cand, cur, GateConfig(bucket_tolerance_pts=0.25)).promote


def test_gate_small_bucket_is_skipped_or_enforced_explicitly():
    cand = make_eval(em=0.91, buckets={"challenge": {"exact_match": 0.40, "n": 12}})
    cur = make_eval(em=0.90, buckets={"challenge": {"exact_match": 0.60, "n": 12}})
    d = evaluate_gate(cand, cur, GateConfig(min_bucket_n=30, small_bucket_policy="skip"))
    assert d.promote
    assert "challenge" in d.check("bucket_regression").reason  # reported, not silently dropped
    assert d.check("bucket_regression").details["skipped"]
    d = evaluate_gate(cand, cur, GateConfig(min_bucket_n=30, small_bucket_policy="enforce"))
    assert not d.promote


def test_gate_bucket_missing_from_candidate_rejects():
    cand = make_eval(em=0.95)
    del cand["buckets"]["lateral"]
    d = evaluate_gate(cand, make_eval(em=0.90))
    assert not d.promote and "lateral" in d.check("bucket_regression").reason


def test_gate_eval_file_mismatch_rejects_even_if_better():
    d = evaluate_gate(make_eval(em=0.99, sha="b" * 64), make_eval(em=0.90))
    assert not d.promote
    assert not d.check("eval_file_match").passed
    assert "different eval files" in d.check("eval_file_match").reason


def test_gate_missing_eval_sha_rejects():
    cand = make_eval(em=0.95)
    del cand["metadata"]["eval_file_sha256"]
    assert not evaluate_gate(cand, make_eval()).check("eval_file_match").passed


def test_gate_broken_unmasked_prompt_adapter_rejected():
    """PRD §9.3 demo: an adapter trained with the prompt mask off mostly echoes prompt
    text, so its outputs rarely parse. Its exact match on the few plays it does parse
    is irrelevant; the valid-rate check stops it."""
    broken = make_eval(em=0.04, valid=0.06)
    d_first = evaluate_gate(broken, None)
    assert not d_first.promote and not d_first.check("valid_rate").passed
    d = evaluate_gate(broken, make_eval(em=0.90))
    assert not d.promote
    failed = {c.name for c in d.checks if not c.passed}
    assert {"valid_rate", "overall_exact_match", "bucket_regression"} <= failed


def test_gate_rejects_percent_scale_artifacts():
    bad = make_eval()
    bad["overall"]["exact_match"] = 90.0
    with pytest.raises(EvalArtifactError, match="fractions"):
        evaluate_gate(bad, None)


def test_gate_accepts_dict_metrics_and_alt_keys():
    art = {
        "eval_sha256": EVAL_SHA,
        "overall": {"n": 10, "exact_match": {"value": 0.9, "ci": [0.8, 0.95]}, "valid_rate": {"point": 1.0}},
        "per_bucket": {"normal": {"n": 10, "exact_match": {"mean": 0.9}}},
    }
    assert evaluate_gate(art, art).promote


# --------------------------------------------------------------------------- registry


def test_register_stores_metadata_and_copies_adapter(reg, tmp_path):
    info = register(reg, tmp_path, "one")
    assert info.version == "v0001" and info.status == "registered"
    assert (info.adapter_path / "adapter_model.safetensors").read_bytes() == b"fake-weights-one"
    assert info.eval_artifact_path.exists()
    for key in ("adapter_sha256", "train_config_sha256", "train_data_sha256",
                "eval_file_sha256", "eval_artifact_sha256", "created_at", "parent"):
        assert key in info.meta
    assert info["eval_file_sha256"] == EVAL_SHA
    assert info["adapter_sha256"] == _io.sha256_dir(info.adapter_path)
    # Registered is not served.
    assert reg.current("pp") is None
    with pytest.raises(NoCurrentVersion):
        resolve("pp", reg.root)


def test_register_rejects_non_peft_dir(reg, tmp_path):
    d = tmp_path / "notpeft"
    d.mkdir()
    (d / "weights.bin").write_bytes(b"x")
    with pytest.raises(RegistryError, match="not a PEFT adapter"):
        reg.register("pp", d, make_eval())


def test_train_config_hash_is_key_order_insensitive(tmp_path):
    from playparse.registry.registry import hash_train_config
    p = tmp_path / "cfg.json"
    p.write_text('{"r": 16,\n  "lr": 0.0002}')
    assert hash_train_config({"lr": 2e-4, "r": 16}) == hash_train_config(p)


def test_promote_then_resolve_then_rollback(reg, tmp_path):
    v1 = register(reg, tmp_path, "one", em=0.88)
    assert reg.promote("pp", v1.version).promote
    assert resolve("pp", reg.root) == v1.adapter_path

    v2 = register(reg, tmp_path, "two", em=0.91)
    assert v2["parent"] == "v0001"  # defaults to the current version
    assert reg.promote("pp", v2.version, reason="better on val").promote
    # The serving resolver sees the new pointer immediately; no cache to invalidate.
    assert resolve("pp", reg.root, verify=True) == v2.adapter_path
    assert reg.get("pp", "v0001").status == "superseded"

    back = reg.rollback("pp", reason="prod spot-check found lateral bug")
    assert back.version == "v0001"
    assert resolve("pp", reg.root) == v1.adapter_path
    assert reg.get("pp", "v0002").status == "rolled_back"
    with pytest.raises(RegistryError, match="nothing to roll back"):
        reg.rollback("pp")


def test_rejected_candidate_never_becomes_current(reg, tmp_path):
    v1 = register(reg, tmp_path, "one", em=0.90)
    reg.promote("pp", v1.version)
    v2 = register(reg, tmp_path, "two", em=0.93, buckets={"fumble": 0.70})
    d = reg.promote("pp", v2.version)
    assert not d.promote
    assert reg.current("pp").version == "v0001"
    assert reg.get("pp", "v0002").status == "rejected"
    assert reg.promoted_stack("pp") == ["v0001"]


def test_eval_file_mismatch_rejected_via_registry(reg, tmp_path):
    reg.promote("pp", register(reg, tmp_path, "one", em=0.90).version)
    v2 = register(reg, tmp_path, "two", em=0.97, sha="c" * 64)
    d = reg.promote("pp", v2.version)
    assert not d.promote and not d.check("eval_file_match").passed


def test_history_log_records_every_decision_with_reasons(reg, tmp_path):
    reg.promote("pp", register(reg, tmp_path, "one", em=0.90).version)
    reg.promote("pp", register(reg, tmp_path, "two", em=0.85).version)       # reject
    reg.promote("pp", register(reg, tmp_path, "three", em=0.92).version)     # promote
    reg.rollback("pp", reason="manual")
    h = reg.history("pp")
    assert [e["action"] for e in h] == [
        "register", "promote", "register", "reject", "register", "promote", "rollback"]
    rej = h[3]
    assert rej["version"] == "v0002" and rej["against"] == "v0001"
    assert any("overall_exact_match" in r for r in rej["reasons"])
    assert rej["decision"]["promote"] is False
    assert {c["name"] for c in rej["decision"]["checks"]} >= {"valid_rate", "bucket_regression"}
    rb = h[-1]
    assert rb["from_version"] == "v0003" and rb["version"] == "v0001" and rb["reason"] == "manual"
    assert all("ts" in e for e in h)


def test_cannot_promote_current_or_unknown(reg, tmp_path):
    v1 = register(reg, tmp_path, "one")
    reg.promote("pp", v1.version)
    with pytest.raises(RegistryError, match="already current"):
        reg.promote("pp", v1.version)
    with pytest.raises(RegistryError, match="no version"):
        reg.promote("pp", "v0099")


def test_resolve_detects_tampered_weights(reg, tmp_path):
    v1 = register(reg, tmp_path, "one")
    reg.promote("pp", v1.version)
    (v1.adapter_path / "adapter_model.safetensors").write_bytes(b"corrupted")
    assert resolve("pp", reg.root) == v1.adapter_path  # cheap path does not hash
    with pytest.raises(RegistryError, match="changed after registration"):
        resolve("pp", reg.root, verify=True)


def test_link_mode_symlinks(reg, tmp_path):
    src = make_adapter(tmp_path / "src", "linked")
    info = reg.register("pp", src, make_eval(), link=True)
    assert info.adapter_path.is_symlink()
    assert info.adapter_path.resolve() == src.resolve()


# --------------------------------------------------------------------------- atomicity


def test_crash_during_index_rename_leaves_old_index(reg, tmp_path, monkeypatch):
    v1 = register(reg, tmp_path, "one", em=0.90)
    reg.promote("pp", v1.version)
    v2 = register(reg, tmp_path, "two", em=0.95)
    before = reg.index_path.read_bytes()

    real_replace = os.replace

    def crash_on_index(src, dst):
        if Path(dst).name == "index.json":
            raise OSError("simulated power loss before rename")
        return real_replace(src, dst)

    monkeypatch.setattr(_io.os, "replace", crash_on_index)
    with pytest.raises(OSError, match="simulated"):
        reg.promote("pp", v2.version)
    monkeypatch.undo()

    assert reg.index_path.read_bytes() == before          # untouched
    assert reg.current("pp").version == "v0001"           # still serving the old one
    assert not list(reg.root.glob(".index.json.*.tmp"))   # temp file cleaned up
    assert reg.promote("pp", v2.version).promote          # and the registry still works


def test_crash_mid_write_of_temp_file_leaves_old_index(reg, tmp_path, monkeypatch):
    register(reg, tmp_path, "one")
    before = reg.index_path.read_bytes()

    real_fdopen = os.fdopen

    class HalfWriter:
        def __init__(self, f):
            self.f = f

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.f.close()
            return False

        def write(self, text):
            self.f.write(text[: len(text) // 2])
            raise OSError("simulated crash halfway through write")

    monkeypatch.setattr(_io.os, "fdopen", lambda fd, *a, **k: HalfWriter(real_fdopen(fd, *a, **k)))
    with pytest.raises(OSError, match="halfway"):
        _io.atomic_write_json(reg.index_path, {"schema_version": 1, "adapters": {}, "history": []})
    monkeypatch.undo()
    assert reg.index_path.read_bytes() == before
    json.loads(before)  # still parseable
    assert not list(reg.root.glob(".index.json.*.tmp"))


def test_failed_register_leaves_no_orphan_version(reg, tmp_path, monkeypatch):
    real_replace = os.replace

    def crash_on_index(src, dst):
        if Path(dst).name == "index.json":
            raise OSError("simulated crash")
        return real_replace(src, dst)

    monkeypatch.setattr(_io.os, "replace", crash_on_index)
    with pytest.raises(OSError):
        register(reg, tmp_path, "one")
    monkeypatch.undo()
    assert not (reg.root / "adapters" / "pp" / "v0001").exists()
    assert not (reg.root / "evals" / "pp" / "v0001.json").exists()
    # A leftover dir from a hard crash (no cleanup ran) is cleared on the next register.
    debris = reg.root / "adapters" / "pp" / "v0001"
    debris.mkdir(parents=True)
    (debris / "junk").write_text("x")
    info = register(reg, tmp_path, "two")
    assert info.version == "v0001" and not (info.adapter_path / "junk").exists()


# --------------------------------------------------------------------------- CLI


def test_cli_end_to_end(tmp_path, capsys):
    root = tmp_path / "registry"
    good = tmp_path / "good.json"
    good.write_text(json.dumps(make_eval(em=0.90)))
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(make_eval(em=0.03, valid=0.05)))
    a1 = make_adapter(tmp_path / "src", "one")
    a2 = make_adapter(tmp_path / "src", "broken")

    assert cli_main(["--root", str(root), "register", "pp", str(a1), "--eval", str(good)]) == 0
    assert cli_main(["--root", str(root), "promote", "pp", "v0001"]) == 0
    assert cli_main(["--root", str(root), "register", "pp", str(a2), "--eval", str(broken)]) == 0
    capsys.readouterr()
    assert cli_main(["--root", str(root), "promote", "pp", "v0002"]) == 2
    out = capsys.readouterr().out
    assert out.startswith("REJECT") and "valid_rate" in out
    assert cli_main(["--root", str(root), "list"]) == 0
    assert "v0001*" in capsys.readouterr().out
    assert cli_main(["--root", str(root), "show", "pp"]) == 0
    assert json.loads(capsys.readouterr().out)["version"] == "v0001"
    assert cli_main(["--root", str(root), "resolve", "pp"]) == 0
    assert capsys.readouterr().out.strip().endswith("adapters/pp/v0001")
    assert cli_main(["--root", str(root), "rollback", "pp"]) == 1  # nothing earlier
    assert "nothing to roll back" in capsys.readouterr().err
    assert cli_main(["--root", str(root), "history", "pp"]) == 0
    assert "reject" in capsys.readouterr().out


def test_gate_reads_the_eval_harness_meta_block():
    # Regression: the harness writes provenance under "meta", and the gate only read
    # "metadata", so every real artifact looked hashless and failed eval_file_match.
    from pathlib import Path

    from playparse.registry.gate import load_eval_summary

    real = Path(__file__).resolve().parent.parent / "results" / "eval_lite" / "r5_full" / "result.json"
    if not real.is_file():
        import pytest
        pytest.skip("committed R5 eval artifact not present")
    s = load_eval_summary(real)
    assert s.eval_file_sha256 and len(s.eval_file_sha256) == 64
    assert s.buckets["fumble"].n == 100
