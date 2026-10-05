"""Prompt rendering shared by training, baselines, and serving.

One function builds the chat messages for a play, so a fine-tuned adapter is always
queried with exactly the prompt it was trained on.
"""
from __future__ import annotations

from playparse.ffscore.schema import STAT_VOCAB

# The Llama 3.x chat template writes "Today Date: <date>" into the system header and
# uses the current day unless `date_string` is passed. Training, eval, and serving
# all pass this one constant, so the rendered prompt is a pure function of the play
# and an adapter is always evaluated on exactly the prompt it was trained on. The
# value is the template's own fallback date.
CHAT_DATE_STRING = "26 Jul 2024"

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
