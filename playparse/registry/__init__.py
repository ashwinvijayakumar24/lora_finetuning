"""Adapter registry, eval gate, and serving resolver (PRD §9.3, phase P6)."""
from playparse.registry.gate import (
    CheckResult,
    EvalArtifactError,
    EvalSummary,
    GateConfig,
    GateDecision,
    evaluate_gate,
    load_eval_summary,
)
from playparse.registry.registry import Registry, RegistryError, VersionInfo, default_root
from playparse.registry.resolver import NoCurrentVersion, resolve

__all__ = [
    "CheckResult",
    "EvalArtifactError",
    "EvalSummary",
    "GateConfig",
    "GateDecision",
    "NoCurrentVersion",
    "Registry",
    "RegistryError",
    "VersionInfo",
    "default_root",
    "evaluate_gate",
    "load_eval_summary",
    "resolve",
]
