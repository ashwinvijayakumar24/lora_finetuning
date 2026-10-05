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


# Prompt styles. "full" is the frozen prompt every result up to P3 used. "minimal"
# drops the system message entirely: a fine-tuned adapter learns the task and the
# JSON format from its labels, so the 186-token instruction may be dead weight that
# every training step and every served request pays for (docs/benchmarks/
# p3-prompt-ablation.md). With no system message the Llama 3.2 template still writes
# its own system header ("Cutting Knowledge Date ... Today Date ..."), so the date
# must stay pinned with CHAT_DATE_STRING in both styles.
PROMPT_STYLES = ("full", "minimal")
DEFAULT_PROMPT_STYLE = "full"


def check_prompt_style(style: str) -> str:
    if style not in PROMPT_STYLES:
        raise ValueError(f"unknown prompt style {style!r}; expected one of {PROMPT_STYLES}")
    return style


def build_messages(
    posteam: str | None,
    desc: str,
    system: str = SYSTEM_PROMPT,
    style: str = DEFAULT_PROMPT_STYLE,
) -> list[dict[str, str]]:
    """Chat messages for one play.

    style="full": [system, user] with `system` (default SYSTEM_PROMPT).
    style="minimal": [user] only. A custom `system` is an error here, since it
    would be silently dropped.
    """
    user = {"role": "user", "content": render_user(posteam, desc)}
    if check_prompt_style(style) == "minimal":
        if system != SYSTEM_PROMPT:
            raise ValueError("style='minimal' sends no system message; do not pass `system`")
        return [user]
    return [{"role": "system", "content": system}, user]
