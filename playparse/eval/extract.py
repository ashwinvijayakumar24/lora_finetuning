"""Pull the first JSON object out of raw model text.

Base models wrap answers in prose or markdown fences (```json ... ```), so every rung's
raw output goes through the same extraction before schema validation. Using one
function for all rungs keeps "valid output" meaning the same thing across the ladder.
"""
from __future__ import annotations

import json

from playparse.ffscore.schema import PlayLabel, SchemaError


def _balanced_span(text: str, start: int) -> int | None:
    """Index one past the `}` closing the `{` at `start`, honoring JSON strings."""
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def extract_first_json_object(text: str) -> str | None:
    """Return the first substring that is a complete, parseable JSON object, else None.

    Scans each `{` in order; the first balanced span that `json.loads` accepts as a
    dict wins. Nested objects are part of their parent, so the outermost object is
    returned.
    """
    if not isinstance(text, str):
        return None
    pos = text.find("{")
    while pos != -1:
        end = _balanced_span(text, pos)
        if end is not None:
            candidate = text[pos:end]
            try:
                if isinstance(json.loads(candidate), dict):
                    return candidate
            except json.JSONDecodeError:
                pass
        pos = text.find("{", pos + 1)
    return None


def parse_prediction(text: str) -> tuple[PlayLabel | None, bool, str | None]:
    """Parse raw output into a label.

    Returns (label or None, strict, error). `strict` is True when the whole stripped
    output is the JSON object with nothing around it, which is what a fine-tuned model
    should produce; the lenient path still accepts prose-wrapped answers.
    """
    candidate = extract_first_json_object(text)
    if candidate is None:
        return None, False, "no JSON object found"
    try:
        label = PlayLabel.from_json(candidate)
    except SchemaError as e:
        return None, False, str(e)
    return label, text.strip() == candidate, None
