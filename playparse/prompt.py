"""Prompt rendering shared by training, baselines, and serving.

One function builds the chat messages for a play, so a fine-tuned adapter is always
queried with exactly the prompt it was trained on.
"""
from __future__ import annotations

from playparse.ffscore.schema import STAT_VOCAB

SYSTEM_PROMPT = (
    "You extract fantasy-football stat credits from one NFL play-by-play description. "
    "Reply with only a JSON object: "
    '{"nullified": <bool>, "credits": [{"player": <name as written in the play>, '
    '"stat": <stat>, "value": <int>}, ...]}. '
    f"Allowed stats: {', '.join(STAT_VOCAB)}. "
    "Use player names exactly as abbreviated in the play (e.g. A.Brown). "
    "Sacks give no rushing yards. Scrambles are runs. Penalty yards are ignored. "
    "Omit credits whose value would be 0. "
    "If a penalty wipes the play out (No Play), set nullified to true and credits to []. "
    "If a replay review REVERSED the ruling, credit the final ruling (the text after "
    "REVERSED) and keep nullified false."
)


def render_user(posteam: str | None, desc: str) -> str:
    return f"posteam: {posteam or 'UNK'}\ndesc: {desc}"


def build_messages(posteam: str | None, desc: str, system: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": render_user(posteam, desc)},
    ]
