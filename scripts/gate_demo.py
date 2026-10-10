"""Eval-gate demo (PRD section 9.3): a good adapter is promoted, a broken one is refused.

    python scripts/gate_demo.py --current runs/sweep/r5_full/best results/eval_lite/r5_full \\
        --candidate runs/gate_demo_broken/best results/eval_lite/gate_demo_broken --root registry_demo

Registers both versions of the "playparse" adapter, promotes the current one through the
gate, asks the gate to promote the candidate (it should refuse, with per-check reasons),
then demonstrates a rollback. Prints every decision and writes them to <root>/demo.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playparse.registry.gate import GateConfig  # noqa: E402
from playparse.registry.registry import Registry  # noqa: E402

NAME = "playparse"


def run(current: tuple[str, str], candidate: tuple[str, str], root: str, min_exact_match: float) -> dict:
    """current/candidate: (adapter_dir, eval result.json path)."""
    reg = Registry(root)
    cfg = GateConfig(min_exact_match=min_exact_match)
    out: dict = {}

    v1 = reg.register(NAME, current[0], current[1], notes="R5: all 294k plays, prompt masked")
    d1 = reg.promote(NAME, v1.version, cfg, reason="first production adapter")
    out["promote_current"] = d1.to_dict()
    print("promote current:", d1.summary())

    v2 = reg.register(NAME, candidate[0], candidate[1], parent=v1.version,
                      notes="deliberately broken: prompt mask off, 20k plays")
    d2 = reg.promote(NAME, v2.version, cfg, reason="candidate")
    out["promote_candidate"] = d2.to_dict()
    print("promote candidate:", d2.summary())
    for r in d2.reasons:
        print("   -", r)
    out["serving_after_candidate"] = reg.current(NAME).version

    # Rollback needs two promoted versions: release R5 again as a new version, then roll back.
    v3 = reg.register(NAME, current[0], current[1], parent=v1.version, notes="R5 again, to show rollback")
    reg.promote(NAME, v3.version, cfg, reason="re-release")
    before = reg.current(NAME).version
    after = reg.rollback(NAME, reason="demo rollback").version
    out["rollback"] = {"from": before, "to": after}
    print(f"rollback: {before} -> {after}")

    out["history"] = reg.history(NAME)
    Path(root, "demo.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--current", nargs=2, metavar=("ADAPTER_DIR", "EVAL_DIR"), required=True)
    ap.add_argument("--candidate", nargs=2, metavar=("ADAPTER_DIR", "EVAL_DIR"), required=True)
    ap.add_argument("--root", default="registry_demo")
    ap.add_argument("--min-exact-match", type=float, default=0.95,
                    help="absolute floor on overall exact match (fraction)")
    a = ap.parse_args(argv)
    run((a.current[0], str(Path(a.current[1]) / "result.json")),
        (a.candidate[0], str(Path(a.candidate[1]) / "result.json")), a.root, a.min_exact_match)


if __name__ == "__main__":
    main()
