"""File-based adapter registry with gated promotion and pointer-swap rollback.

Layout under the registry root (default ``<repo>/registry``, override with the
``PLAYPARSE_REGISTRY`` environment variable or the ``root`` argument)::

    registry/
      index.json                      metadata + history     (committed to git)
      evals/<name>/<version>.json     copy of each eval artifact (committed)
      adapters/<name>/<version>/      PEFT adapter files     (gitignored: weights)
      .lock                           writer lock            (gitignored)

``index.json`` is the single source of truth. It holds every version's metadata,
which version is current, the stack of promoted versions (for rollback), and the
history log. Keeping history inside the same file means one atomic rename commits
both the state change and the record of why it happened; a separate log file could
end up disagreeing with the index after a crash.

Every mutation takes an exclusive lock, reads the index, changes it in memory, and
writes it back with write-temp-then-rename. A crash at any point leaves either the
old index or the new one on disk.

Version ids are ``v0001``, ``v0002``, ... per adapter name.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from playparse.paths import REPO_ROOT
from playparse.registry._io import (
    atomic_write_json,
    atomic_write_text,
    file_lock,
    sha256_dir,
    sha256_file,
    sha256_json,
)
from playparse.registry.gate import GateConfig, GateDecision, evaluate_gate, load_eval_summary

INDEX_SCHEMA_VERSION = 1
PEFT_FILES = ("adapter_config.json", "adapter_model.safetensors")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def default_root() -> Path:
    value = os.environ.get("PLAYPARSE_REGISTRY")
    return Path(value) if value else REPO_ROOT / "registry"


class RegistryError(RuntimeError):
    """A registry operation was refused (unknown version, nothing to roll back, ...)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hash_train_config(config: Mapping[str, Any] | str | Path | None) -> str | None:
    """sha256 of a training config.

    A dict, or a path to a ``.json`` file, is hashed in canonical JSON form so key order
    and whitespace do not matter. Any other file is hashed by its bytes. A 64-char hex
    string is taken to already be a hash.
    """
    if config is None:
        return None
    if isinstance(config, Mapping):
        return sha256_json(dict(config))
    s = str(config)
    if _HEX64_RE.match(s) and not Path(s).exists():
        return s
    p = Path(s)
    if p.suffix == ".json":
        return sha256_json(json.loads(p.read_text(encoding="utf-8")))
    return sha256_file(p)


def hash_train_data(data: str | Path | None) -> str | None:
    """sha256 of the training data file (bytes), or a precomputed 64-char hex hash."""
    if data is None:
        return None
    s = str(data)
    if _HEX64_RE.match(s) and not Path(s).exists():
        return s
    return sha256_file(Path(s))


@dataclass(frozen=True)
class VersionInfo:
    """Read-only view of one registered version's metadata."""

    meta: dict[str, Any]
    root: Path

    @property
    def version(self) -> str:
        return self.meta["version"]

    @property
    def name(self) -> str:
        return self.meta["name"]

    @property
    def status(self) -> str:
        return self.meta["status"]

    @property
    def adapter_path(self) -> Path:
        return self.root / self.meta["adapter_path"]

    @property
    def eval_artifact_path(self) -> Path:
        return self.root / self.meta["eval_artifact"]

    def __getitem__(self, key: str) -> Any:
        return self.meta[key]


class Registry:
    def __init__(self, root: str | Path | None = None, *, require_peft_files: bool = True):
        self.root = Path(root) if root is not None else default_root()
        self.require_peft_files = require_peft_files

    # ------------------------------------------------------------------ paths
    @property
    def index_path(self) -> Path:
        return self.root / "index.json"

    @property
    def lock_path(self) -> Path:
        return self.root / ".lock"

    def _adapter_rel(self, name: str, version: str) -> str:
        return f"adapters/{name}/{version}"

    def _eval_rel(self, name: str, version: str) -> str:
        return f"evals/{name}/{version}.json"

    # ------------------------------------------------------------------ index io
    def _empty_index(self) -> dict[str, Any]:
        return {"schema_version": INDEX_SCHEMA_VERSION, "adapters": {}, "history": []}

    def _load(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return self._empty_index()
        try:
            idx = json.loads(self.index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise RegistryError(f"{self.index_path} is not valid JSON ({e}); restore it from git") from e
        if idx.get("schema_version") != INDEX_SCHEMA_VERSION:
            raise RegistryError(f"unsupported index schema_version {idx.get('schema_version')!r}")
        return idx

    def _save(self, idx: dict[str, Any]) -> None:
        atomic_write_json(self.index_path, idx)

    @staticmethod
    def _entry(idx: dict[str, Any], name: str) -> dict[str, Any]:
        if name not in idx["adapters"]:
            raise RegistryError(f"no adapter named {name!r}")
        return idx["adapters"][name]

    @staticmethod
    def _version_meta(entry: dict[str, Any], name: str, version: str) -> dict[str, Any]:
        if version not in entry["versions"]:
            raise RegistryError(f"{name} has no version {version!r}")
        return entry["versions"][version]

    @staticmethod
    def _log(idx: dict[str, Any], **event: Any) -> None:
        idx["history"].append({"ts": _now(), **event})

    # ------------------------------------------------------------------ reads
    def get(self, name: str, version: str) -> VersionInfo:
        idx = self._load()
        return VersionInfo(self._version_meta(self._entry(idx, name), name, version), self.root)

    def list(self, name: str | None = None) -> list[VersionInfo]:
        idx = self._load()
        names = [name] if name else sorted(idx["adapters"])
        out = []
        for n in names:
            entry = self._entry(idx, n)
            out.extend(VersionInfo(entry["versions"][v], self.root) for v in sorted(entry["versions"]))
        return out

    def names(self) -> list[str]:
        return sorted(self._load()["adapters"])

    def current(self, name: str) -> VersionInfo | None:
        idx = self._load()
        if name not in idx["adapters"]:
            return None
        entry = idx["adapters"][name]
        cur = entry["current"]
        return None if cur is None else VersionInfo(entry["versions"][cur], self.root)

    def promoted_stack(self, name: str) -> list[str]:
        return list(self._entry(self._load(), name)["promoted_stack"])

    def history(self, name: str | None = None) -> list[dict[str, Any]]:
        h = self._load()["history"]
        return [e for e in h if name is None or e.get("name") == name]

    # ------------------------------------------------------------------ register
    def _check_adapter_dir(self, adapter_dir: Path) -> None:
        if not adapter_dir.is_dir():
            raise RegistryError(f"adapter dir {adapter_dir} does not exist")
        if self.require_peft_files:
            missing = [f for f in PEFT_FILES if not (adapter_dir / f).is_file()]
            if missing:
                raise RegistryError(f"{adapter_dir} is not a PEFT adapter dir; missing {missing}")

    def register(
        self,
        name: str,
        adapter_dir: str | Path,
        eval_artifact: str | Path | Mapping[str, Any],
        *,
        parent: str | None = None,
        train_config: Mapping[str, Any] | str | Path | None = None,
        train_data: str | Path | None = None,
        link: bool = False,
        notes: str = "",
    ) -> VersionInfo:
        """Add a new version. It is *registered*, not served, until :meth:`promote` passes.

        ``adapter_dir`` is copied into the registry (``link=True`` symlinks instead, for
        large adapters that already live somewhere permanent). ``eval_artifact`` is the
        eval harness's result JSON; a copy is kept so the gate can always re-read it.
        ``parent`` defaults to the current promoted version.
        """
        if not _NAME_RE.match(name):
            raise RegistryError(f"bad adapter name {name!r}: use letters, digits, '_', '.', '-'")
        adapter_dir = Path(adapter_dir).resolve()
        self._check_adapter_dir(adapter_dir)

        # Load and validate the eval artifact before touching the filesystem.
        if isinstance(eval_artifact, Mapping):
            eval_obj = dict(eval_artifact)
            eval_src = "<dict>"
        else:
            eval_src = str(eval_artifact)
            eval_obj = json.loads(Path(eval_artifact).read_text(encoding="utf-8"))
        summary = load_eval_summary(eval_obj)
        eval_text = json.dumps(eval_obj, indent=2, sort_keys=True) + "\n"

        adapter_sha = sha256_dir(adapter_dir)
        cfg_sha = hash_train_config(train_config)
        data_sha = hash_train_data(train_data)

        with file_lock(self.lock_path):
            idx = self._load()
            entry = idx["adapters"].setdefault(
                name, {"current": None, "promoted_stack": [], "versions": {}}
            )
            if parent is None:
                parent = entry["current"]
            elif parent not in entry["versions"]:
                raise RegistryError(f"parent {parent!r} is not a version of {name}")
            n = 1 + max((int(v[1:]) for v in entry["versions"]), default=0)
            version = f"v{n:04d}"

            target = self.root / self._adapter_rel(name, version)
            eval_path = self.root / self._eval_rel(name, version)
            # Anything already at these paths is debris from a crash between copying and
            # writing the index (the index never mentions it), so it is safe to clear.
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.exists():
                shutil.rmtree(target)

            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.parent / f".tmp-{version}-{uuid.uuid4().hex[:8]}"
            try:
                if link:
                    os.symlink(adapter_dir, tmp, target_is_directory=True)
                else:
                    shutil.copytree(adapter_dir, tmp)
                os.replace(tmp, target)
                atomic_write_text(eval_path, eval_text)

                meta = {
                    "name": name,
                    "version": version,
                    "parent": parent,
                    "created_at": _now(),
                    "status": "registered",
                    "adapter_path": self._adapter_rel(name, version),
                    "adapter_source": str(adapter_dir),
                    "adapter_linked": link,
                    "adapter_sha256": adapter_sha,
                    "train_config_sha256": cfg_sha,
                    "train_data_sha256": data_sha,
                    "eval_artifact": self._eval_rel(name, version),
                    "eval_artifact_source": eval_src,
                    "eval_artifact_sha256": sha256_json(eval_obj),
                    "eval_file_sha256": summary.eval_file_sha256,
                    "metrics": {
                        "exact_match": summary.overall.exact_match,
                        "valid_rate": summary.overall.valid_rate,
                        "n": summary.overall.n,
                    },
                    "notes": notes,
                }
                entry["versions"][version] = meta
                self._log(idx, action="register", name=name, version=version, parent=parent)
                self._save(idx)
            except BaseException:
                # Undo the filesystem side so a failed register leaves no half-version.
                for p in (tmp, target):
                    if p.is_symlink() or p.is_file():
                        p.unlink()
                    elif p.exists():
                        shutil.rmtree(p, ignore_errors=True)
                if eval_path.exists():
                    eval_path.unlink()
                raise
        return VersionInfo(meta, self.root)

    # ------------------------------------------------------------------ promote / rollback
    def promote(
        self,
        name: str,
        version: str,
        config: GateConfig | None = None,
        *,
        reason: str = "",
    ) -> GateDecision:
        """Run the eval gate; make ``version`` current only if it passes.

        Both outcomes are written to the history log with every check's result. There
        is deliberately no "force" flag: the only path to serving is through the gate.
        """
        cfg = config or GateConfig()
        with file_lock(self.lock_path):
            idx = self._load()
            entry = self._entry(idx, name)
            meta = self._version_meta(entry, name, version)
            cur = entry["current"]
            if cur == version:
                raise RegistryError(f"{name} {version} is already current")
            cand_eval = self.root / meta["eval_artifact"]
            cur_eval = None if cur is None else self.root / entry["versions"][cur]["eval_artifact"]
            decision = evaluate_gate(cand_eval, cur_eval, cfg)
            event = {
                "name": name,
                "version": version,
                "against": cur,
                "reason": reason,
                "gate_config": cfg.__dict__.copy(),
                "decision": decision.to_dict(),
            }
            if decision.promote:
                entry["promoted_stack"].append(version)
                entry["current"] = version
                meta["status"] = "promoted"
                meta["promoted_at"] = _now()
                if cur is not None:
                    entry["versions"][cur]["status"] = "superseded"
                self._log(idx, action="promote", **event)
            else:
                if meta["status"] == "registered":
                    meta["status"] = "rejected"
                meta.setdefault("rejections", []).append({"ts": _now(), "against": cur, "reasons": decision.reasons})
                self._log(idx, action="reject", reasons=decision.reasons, **event)
            self._save(idx)
        return decision

    def rollback(self, name: str, *, reason: str = "") -> VersionInfo:
        """Point ``current`` back at the previously promoted version (a pointer swap).

        The serving layer picks this up the next time it calls :func:`resolve`. The
        rolled-back version stays registered (status ``rolled_back``) so it can be
        inspected or re-promoted through the gate later.
        """
        with file_lock(self.lock_path):
            idx = self._load()
            entry = self._entry(idx, name)
            stack = entry["promoted_stack"]
            if len(stack) < 2:
                raise RegistryError(
                    f"{name}: nothing to roll back to (promoted history: {stack or 'empty'})"
                )
            bad = stack.pop()
            prev = stack[-1]
            entry["current"] = prev
            entry["versions"][bad]["status"] = "rolled_back"
            entry["versions"][prev]["status"] = "promoted"
            self._log(idx, action="rollback", name=name, version=prev, from_version=bad, reason=reason)
            self._save(idx)
            return VersionInfo(entry["versions"][prev], self.root)
