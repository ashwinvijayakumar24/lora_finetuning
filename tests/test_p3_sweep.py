"""The P3 sweep plan: every printed sbatch command resolves to a valid RunSpec,
and the data-size samples are nested and seeded."""
from __future__ import annotations

import importlib.util
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from playparse.train.build import RunSpec, nested_sample

REPO = Path(__file__).resolve().parent.parent


def _train_script():
    spec = importlib.util.spec_from_file_location("train_script", REPO / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lora_params(spec: RunSpec, layers: int = 16) -> int:
    cost = {"q_proj": 4096, "k_proj": 2560, "v_proj": 2560, "o_proj": 4096,
            "gate_proj": 10240, "up_proj": 10240, "down_proj": 10240}
    return layers * spec.lora.r * sum(cost[t] for t in spec.lora.targets)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_sweep_commands_resolve_to_valid_specs():
    out = subprocess.run(["bash", "scripts/slurm/p3_sweep.sh", "all"], cwd=REPO, capture_output=True, text=True,
                         check=True).stdout.strip().splitlines()
    train = _train_script()
    names, specs = [], {}
    for line in out:
        argv = shlex.split(line)
        i = argv.index("scripts/slurm/train_h100.sbatch")
        assert argv[0] == "sbatch"
        config, outdir, sets = argv[i + 1], argv[i + 2], argv[i + 3:]
        raw = yaml.safe_load((REPO / config).read_text())
        spec = RunSpec.from_dict(train.apply_overrides(raw, sets))
        name = Path(outdir).name
        names.append(name)
        specs[name] = spec
    assert len(names) == len(set(names)) == 26  # 1 R5 + 5 rank + 5 alpha + 4 targets + 2 dropout + 4 lr + 1 prompt control + 4 data size
    assert specs["rank_r64"].lora.r == 64 and specs["rank_r64"].lora.alpha == 128
    assert specs["targets_qv_r16"].lora.targets == ["q_proj", "v_proj"]
    assert specs["lr_1.0e-3"].train.lr == pytest.approx(1e-3)
    assert specs["datasize_1000"].data.train_sample == 1000
    assert specs["r5_full"].data.train.endswith("train.jsonl")
    assert all(s.data.train.endswith("train_50k.jsonl") for n, s in specs.items() if n.split("_")[0] in
               ("rank", "alpha", "targets", "dropout", "lr"))
    # Matched-budget target runs are within 1% of all-linear r=16 (11.27M).
    ref = _lora_params(specs["rank_r16"])
    assert ref == 11_272_192
    for n in ("targets_qv_matched_r106", "targets_qkvo_matched_r53"):
        assert abs(_lora_params(specs[n]) / ref - 1) < 0.01, n


def test_nested_sample_is_seeded_and_nested():
    recs = [{"i": i} for i in range(1000)]
    a, b = nested_sample(recs, 100, 0), nested_sample(recs, 300, 0)
    assert b[:100] == a
    assert nested_sample(recs, 100, 1) != a
    assert nested_sample(recs, 0, 0) == recs and nested_sample(recs, 5000, 0) == recs
