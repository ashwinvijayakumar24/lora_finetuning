"""CLI: python -m playparse.registry {register,list,promote,rollback,show,resolve,history}.

Examples::

    python -m playparse.registry register playparse runs/r5/adapter \\
        --eval results/r5_val.json --train-config configs/r5.json --train-data data/processed/train.jsonl
    python -m playparse.registry promote playparse v0002 --bucket-tolerance 1.0
    python -m playparse.registry rollback playparse --reason "lateral bucket regressed in prod spot-check"
    python -m playparse.registry show playparse            # current version
    python -m playparse.registry list
"""
from __future__ import annotations

import argparse
import json
import sys

from playparse.registry.gate import GateConfig
from playparse.registry.registry import Registry, RegistryError
from playparse.registry.resolver import resolve


def _fmt_pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.2f}"


def _cmd_register(reg: Registry, a: argparse.Namespace) -> int:
    info = reg.register(
        a.name, a.adapter_dir, a.eval, parent=a.parent, train_config=a.train_config,
        train_data=a.train_data, link=a.link, notes=a.notes,
    )
    print(f"registered {info.name} {info.version} (status: registered; promote it through the gate)")
    return 0


def _cmd_list(reg: Registry, a: argparse.Namespace) -> int:
    rows = reg.list(a.name)
    if not rows:
        print("registry is empty")
        return 0
    currents = {n: (c.version if (c := reg.current(n)) else None) for n in {r.name for r in rows}}
    print(f"{'name':<16} {'version':<8} {'status':<12} {'EM%':>7} {'valid%':>7} {'parent':<8} created")
    for r in rows:
        mark = "*" if currents[r.name] == r.version else " "
        m = r["metrics"]
        print(f"{r.name:<16} {r.version + mark:<8} {r.status:<12} {_fmt_pct(m['exact_match']):>7} "
              f"{_fmt_pct(m['valid_rate']):>7} {r['parent'] or '-':<8} {r['created_at']}")
    print("(* = current)")
    return 0


def _cmd_promote(reg: Registry, a: argparse.Namespace) -> int:
    cfg = GateConfig(
        min_valid_rate=a.min_valid_rate, min_exact_match=a.min_exact_match,
        bucket_tolerance_pts=a.bucket_tolerance, min_bucket_n=a.min_bucket_n,
        small_bucket_policy=a.small_buckets,
    )
    decision = reg.promote(a.name, a.version, cfg, reason=a.reason)
    print(decision.summary())
    return 0 if decision.promote else 2


def _cmd_rollback(reg: Registry, a: argparse.Namespace) -> int:
    info = reg.rollback(a.name, reason=a.reason)
    print(f"{a.name}: current is now {info.version}")
    return 0


def _cmd_show(reg: Registry, a: argparse.Namespace) -> int:
    info = reg.get(a.name, a.version) if a.version else reg.current(a.name)
    if info is None:
        print(f"{a.name} has no promoted version")
        return 1
    print(json.dumps(info.meta, indent=2, sort_keys=True))
    return 0


def _cmd_resolve(reg: Registry, a: argparse.Namespace) -> int:
    print(resolve(a.name, reg.root, verify=a.verify))
    return 0


def _cmd_history(reg: Registry, a: argparse.Namespace) -> int:
    for e in reg.history(a.name):
        extra = "; ".join(e.get("reasons", [])) or e.get("reason", "")
        ver = e.get("version")
        if e["action"] == "rollback":
            ver = f"{e['from_version']} -> {e['version']}"
        print(f"{e['ts']}  {e['action']:<8} {e['name']} {ver}  {extra}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m playparse.registry", description=__doc__.splitlines()[0])
    p.add_argument("--root", help="registry root (default: $PLAYPARSE_REGISTRY or <repo>/registry)")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("register", help="add an adapter version (not served until promoted)")
    r.add_argument("name")
    r.add_argument("adapter_dir", help="PEFT adapter dir (adapter_config.json + adapter_model.safetensors)")
    r.add_argument("--eval", required=True, help="eval harness result JSON for this adapter")
    r.add_argument("--parent", help="parent version (default: current)")
    r.add_argument("--train-config", help="training config file (.json hashed canonically) or sha256")
    r.add_argument("--train-data", help="training data file or its sha256")
    r.add_argument("--link", action="store_true", help="symlink the adapter dir instead of copying")
    r.add_argument("--notes", default="")

    ls = sub.add_parser("list", help="list versions")
    ls.add_argument("name", nargs="?")

    d = GateConfig()
    pr = sub.add_parser("promote", help="run the eval gate and promote on pass (exit 2 on reject)")
    pr.add_argument("name")
    pr.add_argument("version")
    pr.add_argument("--reason", default="")
    pr.add_argument("--min-valid-rate", type=float, default=d.min_valid_rate)
    pr.add_argument("--min-exact-match", type=float, default=d.min_exact_match)
    pr.add_argument("--bucket-tolerance", type=float, default=d.bucket_tolerance_pts, help="points")
    pr.add_argument("--min-bucket-n", type=int, default=d.min_bucket_n)
    pr.add_argument("--small-buckets", choices=("skip", "enforce"), default=d.small_bucket_policy)

    rb = sub.add_parser("rollback", help="point current back at the previous promoted version")
    rb.add_argument("name")
    rb.add_argument("--reason", default="")

    sh = sub.add_parser("show", help="show a version's metadata (default: current)")
    sh.add_argument("name")
    sh.add_argument("version", nargs="?")

    rs = sub.add_parser("resolve", help="print the adapter path serving would load")
    rs.add_argument("name")
    rs.add_argument("--verify", action="store_true")

    hi = sub.add_parser("history", help="promotions, rejections, rollbacks")
    hi.add_argument("name", nargs="?")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    reg = Registry(a.root)
    handler = {
        "register": _cmd_register, "list": _cmd_list, "promote": _cmd_promote,
        "rollback": _cmd_rollback, "show": _cmd_show, "resolve": _cmd_resolve,
        "history": _cmd_history,
    }[a.cmd]
    try:
        return handler(reg, a)
    except RegistryError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
