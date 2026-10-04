"""The P4 pilot script: dry-run estimate, the minimal SDK client's accounting, and a
full mocked "real" run. No network, no API key."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from playparse.ffscore.schema import Credit, PlayLabel

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "p4_pilot.py"


@pytest.fixture(scope="module")
def pilot():
    spec = importlib.util.spec_from_file_location("p4_pilot", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_data(path: Path, n: int = 30) -> list[dict]:
    recs = []
    for pid in range(1, n + 1):
        y = 2 + pid % 9
        label = PlayLabel(False, (Credit("S.Barkley", "rush_yds", y),))
        recs.append({"game_id": "2022_05_NYG_GB", "play_id": pid, "season": 2022, "week": 5,
                     "season_type": "REG", "posteam": "NYG", "bucket": "normal",
                     "desc": f"(7:00) 26-S.Barkley left end to GB 30 for {y} yards (52-R.Gary).",
                     "label": label.to_json()})
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return recs


def test_dry_run_estimates_from_data(pilot, tmp_path, capsys):
    data = tmp_path / "train.jsonl"
    write_data(data)
    out = tmp_path / "est.json"
    assert pilot.main(["--dry-run", "--data", str(data), "--n", "20", "--json-out", str(out)]) == 0
    est = json.loads(out.read_text())
    assert est["source"].startswith("20 plays")
    assert est["k"] == 3 and est["pilot_plays"] == 20
    assert 10 < est["play_tokens_avg"] < 40
    by = {r["model"]: r for r in est["rows"]}
    assert set(by) == set(pilot.PRICES)
    opus = by["claude-opus-5"]
    assert opus["usd_per_call_cached"] < opus["usd_per_call_uncached"]
    assert opus["pilot_usd_cached"] == pytest.approx(opus["usd_per_call_cached"] * 60)
    assert by["claude-haiku-4-5"]["full_usd_cached"] < opus["full_usd_cached"]
    assert "chars/4" in capsys.readouterr().out


def test_dry_run_without_data_uses_typical_lengths(pilot, tmp_path):
    est = pilot.dry_run(pilot.build_parser().parse_args(["--dry-run", "--data", str(tmp_path / "nope.jsonl")]))
    assert "not found" in est["source"]


def test_real_run_refuses_without_credentials(pilot, tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="B2"):
        pilot.main(["--data", str(tmp_path / "x.jsonl"), "--budget-usd", "1"])
    with pytest.raises(SystemExit, match="budget-usd"):
        pilot.main(["--data", str(tmp_path / "x.jsonl")])


class FakeSDK:
    """Duck-types `anthropic.Anthropic().messages.create` for the minimal client."""

    def __init__(self, reply):
        self.calls = []
        self.messages = SimpleNamespace(create=self.create)
        self.reply = reply

    def create(self, **kw):
        self.calls.append(kw)
        desc = kw["messages"][0]["content"].split("desc: ", 1)[1]
        return SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking=""),
                     SimpleNamespace(type="text", text=self.reply(desc))],
            usage=SimpleNamespace(input_tokens=40, output_tokens=30,
                                  cache_read_input_tokens=1000, cache_creation_input_tokens=0),
        )


def _reply(desc):
    y = int(desc.split(" for ")[1].split()[0])
    return "```json\n" + PlayLabel(False, (Credit("S.Barkley", "rush_yds", y),)).to_json() + "\n```"


def test_anthropic_teacher_accounting(pilot):
    sdk = FakeSDK(_reply)
    t = pilot.AnthropicTeacher("claude-opus-5", "SYS", sdk_client=sdk)
    batch = t([{"game_id": "g", "play_id": 1, "posteam": "NYG",
                "desc": "26-S.Barkley left end for 4 yards"}], 3)
    assert len(batch.outputs) == 1 and len(batch.outputs[0]) == 3
    assert batch.usage.calls == 3
    expected = 3 * (40 * 5 + 1000 * 0.1 * 5 + 30 * 25) / 1e6
    assert batch.usage.cost_usd == pytest.approx(expected)
    kw = sdk.calls[0]
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kw["output_config"] == {"effort": "low"}
    assert "temperature" not in kw  # rejected by current Opus models
    # Haiku 4.5 rejects `effort`, so the client must not send it.
    haiku_sdk = FakeSDK(_reply)
    pilot.AnthropicTeacher("claude-haiku-4-5", "S", sdk_client=haiku_sdk)(
        [{"game_id": "g", "play_id": 1, "desc": "26-S.Barkley left end for 4 yards"}], 1)
    assert "output_config" not in haiku_sdk.calls[0]


def test_mocked_real_run_end_to_end(pilot, tmp_path, monkeypatch):
    data = tmp_path / "train.jsonl"
    write_data(data, n=12)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-used")
    monkeypatch.setattr(pilot, "make_client",
                        lambda args, system: pilot.AnthropicTeacher(args.model, system, sdk_client=FakeSDK(_reply)))
    out = tmp_path / "runs"
    assert pilot.main(["--data", str(data), "--n", "12", "--budget-usd", "5", "--out", str(out)]) == 0
    rep = json.loads((out / "claude-opus-5" / "pilot_report.json").read_text())
    assert rep["plays_labeled"] == 12 and rep["stopped_reason"] is None
    assert rep["r9"]["n"] == 12 and rep["r9"]["precision"]["precision"] == 1.0
    assert rep["usd_per_1k_plays"] > 0
