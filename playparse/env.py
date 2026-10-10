"""Load API keys from the repo's gitignored .env file without printing them.

Only KEY=VALUE lines are read, and variables already set in the environment win, so a
key exported in the shell always overrides the file.
"""
from __future__ import annotations

import os
from pathlib import Path

from playparse.paths import REPO_ROOT


def load_dotenv(path: str | Path | None = None) -> list[str]:
    """Set unset variables from `path` (default: <repo>/.env). Returns the names it set."""
    p = Path(path) if path else REPO_ROOT / ".env"
    if not p.is_file():
        return []
    set_names = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip().removeprefix("export ").strip(), value.strip().strip("'\"")
        if name and name not in os.environ:
            os.environ[name] = value
            set_names.append(name)
    return set_names
