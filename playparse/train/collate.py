"""Turn dataset records into padded training batches with completion-only labels.

One training example is a prompt followed by a completion:

    prompt     = chat template applied to build_messages(posteam, desc),
                 ending with the assistant header (add_generation_prompt=True)
    completion = PlayLabel.to_json() string + the end-of-turn token (<|eot_id|>)

The model sees both, but only the completion tokens carry a loss: prompt and
padding positions get label -100 (PyTorch's cross-entropy ignore_index).

Labels are stored *aligned* with input_ids (labels[i] is the token at position i).
The loss function shifts them by one, so that logits at position i are scored
against the token at position i + 1. See `playparse.train.loop.token_loss_sum`.

Two traps this module guards against, each with a test in tests/test_collate.py:

* Duplicated BOS. The Llama chat template already starts with <|begin_of_text|>.
  Tokenizing the rendered string with add_special_tokens=True adds a second one.
  We tokenize with add_special_tokens=False and assert exactly one BOS.
* A date inside the prompt. The Llama 3.x template writes "Today Date: <today>"
  into the system header unless `date_string` is passed. Unpinned, the prompt a
  model was trained on would differ from the prompt it is served with on any
  other day. We always pass playparse.prompt.CHAT_DATE_STRING, the same
  constant the eval harness uses. This holds for both prompt styles: with
  prompt_style="minimal" there is no system message, but the template still
  writes a system header with the date.

prompt_style ("full", "minimal" or "minimal_v2", see playparse.prompt.PROMPT_STYLES)
selects the messages. It defaults to "full", the prompt every earlier result used.
"minimal_v2" (T3b) also selects the completion: the record's schema-v2 `label_v2`
(field spots) instead of `label`, and puts the record's `los` in the prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

import torch

from playparse.prompt import CHAT_DATE_STRING, DEFAULT_PROMPT_STYLE, build_messages, schema_of_style

IGNORE_INDEX = -100

EOT_TOKEN = "<|eot_id|>"  # Llama 3.x end-of-turn; the chat template closes every turn with it
PAD_TOKEN_CANDIDATES = ("<|finetune_right_pad_id|>", "<|reserved_special_token_0|>")


class OverLengthError(ValueError):
    """An example does not fit in max_len. We never truncate silently."""


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------


def _token_id(tokenizer: Any, token: str) -> int | None:
    tid = tokenizer.convert_tokens_to_ids(token)
    if tid is None or tid == getattr(tokenizer, "unk_token_id", None):
        return None
    return int(tid)


def end_of_turn_id(tokenizer: Any) -> int:
    """The token the chat template closes the assistant turn with.

    For Llama 3.2 Instruct this is <|eot_id|> (128009), which is also
    tokenizer.eos_token. It is *not* <|end_of_text|> (128001): that token ends a
    whole document in pre-training and never appears inside a chat turn.
    """
    tid = _token_id(tokenizer, EOT_TOKEN)
    if tid is not None:
        return tid
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer has neither <|eot_id|> nor an eos_token")
    return int(tokenizer.eos_token_id)


def stop_token_ids(tokenizer: Any) -> list[int]:
    """Token ids that should end generation: end-of-turn first, then any other EOS."""
    ids = [end_of_turn_id(tokenizer)]
    for tok in ("<|end_of_text|>", "<|eom_id|>"):
        tid = _token_id(tokenizer, tok)
        if tid is not None and tid not in ids:
            ids.append(tid)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(int(tokenizer.eos_token_id))
    return ids


def pad_token_id(tokenizer: Any) -> int:
    """A pad id. Llama 3.2 ships without one; its value never matters because padded
    positions are masked out of attention and loss, but it must be a valid id."""
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    for tok in PAD_TOKEN_CANDIDATES:
        tid = _token_id(tokenizer, tok)
        if tid is not None:
            return tid
    return end_of_turn_id(tokenizer)


# ---------------------------------------------------------------------------
# Rendering and encoding
# ---------------------------------------------------------------------------


def render_prompt(
    tokenizer: Any,
    posteam: str | None,
    desc: str,
    date_string: str = CHAT_DATE_STRING,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
    los: str | None = None,
) -> str:
    """The exact prompt string (chat template + assistant header) for one play.

    `los` (line of scrimmage) is rendered only by prompt_style="minimal_v2"."""
    return tokenizer.apply_chat_template(
        build_messages(posteam, desc, style=prompt_style, los=los),
        tokenize=False,
        add_generation_prompt=True,
        date_string=date_string,
    )


def encode_prompt(
    tokenizer: Any,
    posteam: str | None,
    desc: str,
    date_string: str = CHAT_DATE_STRING,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
    los: str | None = None,
) -> list[int]:
    """Prompt token ids, with exactly one BOS. Used by training *and* generation."""
    text = render_prompt(tokenizer, posteam, desc, date_string, prompt_style, los)
    # add_special_tokens=False: the template already wrote <|begin_of_text|>.
    ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    _check_single_bos(tokenizer, ids)
    return ids


def record_los(record: Mapping[str, Any], prompt_style: str) -> str | None:
    """The record's line of scrimmage, required by a schema-v2 prompt style.

    A v1 data file has no "los" key; feeding it to a v2 run would silently render
    "los: UNK" on every play, so that is an error.
    """
    if schema_of_style(prompt_style) == "v2" and "los" not in record:
        raise KeyError(f"prompt_style={prompt_style!r} needs records with a 'los' field "
                       "(the v2 files in data/processed_v2, see playparse.data.build_dataset_v2)")
    return record.get("los")


def completion_text(record: Mapping[str, Any], prompt_style: str) -> str:
    """The training target: the v1 `label`, or `label_v2` for a schema-v2 style."""
    if schema_of_style(prompt_style) == "v2":
        text = record.get("label_v2")
        if not text:
            raise KeyError(f"prompt_style={prompt_style!r} trains on 'label_v2', which this record lacks")
        return text
    return record["label"]


def _check_single_bos(tokenizer: Any, ids: Sequence[int]) -> None:
    bos = tokenizer.bos_token_id
    if bos is None:
        return
    n = sum(1 for t in ids if t == bos)
    if n != 1 or ids[0] != bos:
        raise ValueError(f"expected exactly one BOS at position 0, found {n} (first id {ids[0]})")


@dataclass
class EncodedExample:
    """One tokenized example. labels are aligned with input_ids (unshifted)."""

    input_ids: list[int]
    labels: list[int]
    n_prompt: int
    n_completion: int
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.input_ids)


def make_example(
    prompt_ids: Sequence[int],
    completion_ids: Sequence[int],
    mask_prompt: bool = True,
    meta: dict[str, Any] | None = None,
) -> EncodedExample:
    """Concatenate prompt and completion ids and build labels.

    mask_prompt=True: labels are -100 on the prompt, the token id on the completion.
    mask_prompt=False: every token is a target (the model is also trained to
    reproduce the system prompt and the play text). This is deliberately worse; it
    exists for the masking ablation and the broken-adapter demo.
    """
    input_ids = list(prompt_ids) + list(completion_ids)
    if mask_prompt:
        labels = [IGNORE_INDEX] * len(prompt_ids) + list(completion_ids)
    else:
        labels = list(input_ids)
    return EncodedExample(input_ids, labels, len(prompt_ids), len(completion_ids), dict(meta or {}))


def encode_record(
    record: Mapping[str, Any],
    tokenizer: Any,
    max_len: int | None = None,
    mask_prompt: bool = True,
    date_string: str = CHAT_DATE_STRING,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
) -> EncodedExample:
    """Encode one dataset record ({"posteam", "desc", "label", ...}).

    The prompt and the completion are tokenized separately and concatenated as ids.
    That matches inference exactly: at serving time the model receives the prompt
    ids and generates completion ids one by one, so no BPE merge can ever cross the
    prompt/completion boundary. Raises OverLengthError if the result exceeds max_len.
    """
    p_ids = encode_prompt(tokenizer, record.get("posteam"), record["desc"], date_string, prompt_style,
                          record_los(record, prompt_style))
    c_ids = list(tokenizer(completion_text(record, prompt_style), add_special_tokens=False)["input_ids"])
    c_ids.append(end_of_turn_id(tokenizer))
    meta = {k: record[k] for k in ("game_id", "play_id", "bucket") if k in record}
    ex = make_example(p_ids, c_ids, mask_prompt=mask_prompt, meta=meta)
    if max_len is not None and len(ex) > max_len:
        raise OverLengthError(
            f"example {meta or ''} has {len(ex)} tokens "
            f"(prompt {ex.n_prompt} + completion {ex.n_completion}) > max_len {max_len}"
        )
    return ex


@dataclass
class EncodeReport:
    n_records: int
    n_kept: int
    over_length: list[dict[str, Any]]
    max_tokens: int
    mean_tokens: float


def encode_records(
    records: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    max_len: int | None = None,
    mask_prompt: bool = True,
    on_overlength: Literal["raise", "drop"] = "raise",
    date_string: str = CHAT_DATE_STRING,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
) -> tuple[list[EncodedExample], EncodeReport]:
    """Encode many records. Over-length examples raise by default.

    on_overlength="drop" skips them but lists every one in the report, so dropping
    is always visible, never silent.
    """
    if on_overlength not in ("raise", "drop"):
        raise ValueError(on_overlength)
    out: list[EncodedExample] = []
    over: list[dict[str, Any]] = []
    n = 0
    for rec in records:
        n += 1
        ex = encode_record(rec, tokenizer, None, mask_prompt, date_string, prompt_style)
        if max_len is not None and len(ex) > max_len:
            info = {**ex.meta, "n_tokens": len(ex), "n_prompt": ex.n_prompt, "n_completion": ex.n_completion}
            if on_overlength == "raise":
                raise OverLengthError(f"example {info} exceeds max_len {max_len}")
            over.append(info)
            continue
        out.append(ex)
    lens = [len(e) for e in out]
    report = EncodeReport(
        n_records=n,
        n_kept=len(out),
        over_length=over,
        max_tokens=max(lens) if lens else 0,
        mean_tokens=sum(lens) / len(lens) if lens else 0.0,
    )
    return out, report


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def collate(
    examples: Sequence[EncodedExample],
    pad_id: int,
    pad_to_multiple_of: int | None = None,
) -> dict[str, torch.Tensor]:
    """Right-pad a list of examples into tensors.

    Returns input_ids, attention_mask (1 = real token), and labels (-100 on padding
    and, if masked, on the prompt). Right padding is the right choice for training:
    with a causal mask, real tokens never attend to the pads that come after them,
    so position ids stay 0..n-1 and outputs equal the unpadded ones.
    """
    if not examples:
        raise ValueError("cannot collate an empty batch")
    width = max(len(e) for e in examples)
    if pad_to_multiple_of:
        width = -(-width // pad_to_multiple_of) * pad_to_multiple_of
    b = len(examples)
    input_ids = torch.full((b, width), pad_id, dtype=torch.long)
    labels = torch.full((b, width), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((b, width), dtype=torch.long)
    for i, e in enumerate(examples):
        n = len(e)
        input_ids[i, :n] = torch.tensor(e.input_ids, dtype=torch.long)
        labels[i, :n] = torch.tensor(e.labels, dtype=torch.long)
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def count_target_tokens(labels: torch.Tensor) -> int:
    """Number of positions that contribute to the loss after the shift by one."""
    return int((labels[:, 1:] != IGNORE_INDEX).sum())
