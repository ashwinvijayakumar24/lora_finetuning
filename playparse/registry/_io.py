"""Small filesystem helpers shared by the registry: atomic writes, hashing, locking."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator


def atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` so readers see either the old file or the new one, never half.

    The data goes to a temp file in the same directory (same filesystem, so the
    rename is atomic), is fsynced, and only then renamed over the target. If anything
    fails before the rename, the temp file is removed and the target is untouched.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    # Persist the rename itself (directory entry) where the platform allows it.
    with contextlib.suppress(OSError):
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_dir(path: Path) -> str:
    """Content hash of a directory: every file's relative path and bytes, in sorted order.

    Hidden files (names starting with '.') are skipped so OS litter such as .DS_Store
    does not change an adapter's identity.
    """
    path = Path(path)
    h = hashlib.sha256()
    files = sorted(
        p for p in path.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(path).parts)
    )
    for p in files:
        rel = p.relative_to(path).as_posix()
        h.update(rel.encode() + b"\0")
        h.update(sha256_file(p).encode() + b"\n")
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    """Hash of a JSON-serializable object in canonical form (sorted keys, no spaces)."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Exclusive advisory lock so two writers never interleave read-modify-write cycles."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
