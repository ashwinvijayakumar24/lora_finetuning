"""Serving-facing lookup: which adapter directory should be loaded for a name right now.

The serving layer calls :func:`resolve` every time it (re)loads an adapter. The
function re-reads ``index.json`` on every call and caches nothing, so a promotion or
a rollback (both just rewrite the ``current`` pointer) takes effect on the next load
without restarting anything.

Usage from serving code::

    from playparse.registry import resolve
    path = resolve("playparse")          # -> Path to a PEFT adapter directory
    adapter = load_peft_adapter(path)    # whatever loader the serving layer uses
"""
from __future__ import annotations

from pathlib import Path

from playparse.registry._io import sha256_dir
from playparse.registry.registry import Registry, RegistryError


class NoCurrentVersion(RegistryError):
    """The adapter name exists (or not) but no version has passed the gate yet."""


def resolve(name: str, root: str | Path | None = None, *, verify: bool = False) -> Path:
    """Return the directory of the current promoted adapter for ``name``.

    ``verify=True`` re-hashes the directory and compares it with the sha256 recorded at
    registration, catching weights that were edited or corrupted on disk. It costs a
    full read of the adapter, so it is off by default.
    """
    reg = Registry(root)
    cur = reg.current(name)
    if cur is None:
        raise NoCurrentVersion(f"{name!r} has no promoted version in {reg.root}")
    path = cur.adapter_path
    if not path.exists():
        raise RegistryError(
            f"{name} {cur.version} points at {path}, which is missing "
            "(adapter weights are not in git; copy them back or re-register)"
        )
    if verify:
        actual = sha256_dir(path)
        if actual != cur["adapter_sha256"]:
            raise RegistryError(
                f"{name} {cur.version}: adapter sha256 {actual[:12]} != registered "
                f"{cur['adapter_sha256'][:12]}; the files changed after registration"
            )
    return path
