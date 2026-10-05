"""Batched greedy decoding for a trained model: the generation half of eval.

Why left padding here when training uses right padding: generation appends new
tokens at the right end of every row. With right padding, a short prompt's next
token would land after its pads, and its position ids would be wrong. Left padding
lines every prompt's last real token up in the final column, so one argmax over the
last position serves the whole batch. Position ids are computed from the attention
mask (cumsum - 1), so a left-padded prompt still sees positions 0..n-1.

Decoding stops per row at the first stop token (<|eot_id|> for Llama 3.2
Instruct, plus the other EOS ids), or after max_new_tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from playparse.prompt import CHAT_DATE_STRING, DEFAULT_PROMPT_STYLE
from playparse.train.collate import encode_prompt, pad_token_id, stop_token_ids
from playparse.train.loop import autocast_context


@dataclass
class Generation:
    text: str  # decoded completion, stop token excluded, special tokens kept (raw)
    token_ids: list[int]  # generated ids, stop token excluded
    stopped: bool  # True if a stop token ended it; False if max_new_tokens did


@torch.no_grad()
def greedy_generate_ids(
    model: nn.Module,
    prompts: Sequence[Sequence[int]],
    *,
    stop_ids: Sequence[int],
    pad_id: int,
    max_new_tokens: int = 128,
    batch_size: int = 16,
    autocast: str = "auto",
    sort_by_length: bool = True,
) -> list[tuple[list[int], bool]]:
    """Greedy-decode every prompt. Returns (generated ids without the stop token,
    stopped) per prompt, in input order.

    The model must be a Hugging Face style causal LM (accepting position_ids,
    past_key_values, use_cache). Prompts are grouped by length to cut padding.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    stop = torch.tensor(sorted(set(stop_ids)), device=device)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i])) if sort_by_length else list(range(len(prompts)))
    results: list[tuple[list[int], bool] | None] = [None] * len(prompts)
    try:
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            for i, res in zip(idx, _generate_batch(model, [prompts[i] for i in idx], stop, pad_id, max_new_tokens,
                                                   device, autocast)):
                results[i] = res
    finally:
        model.train(was_training)
    return results  # type: ignore[return-value]


def _generate_batch(model, prompts, stop, pad_id, max_new_tokens, device, autocast):
    b = len(prompts)
    width = max(len(p) for p in prompts)
    input_ids = torch.full((b, width), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((b, width), dtype=torch.long, device=device)
    for r, p in enumerate(prompts):
        input_ids[r, width - len(p) :] = torch.tensor(list(p), dtype=torch.long, device=device)
        attn[r, width - len(p) :] = 1
    pos = (attn.cumsum(-1) - 1).clamp(min=0)

    out_ids = torch.full((b, max_new_tokens), pad_id, dtype=torch.long, device=device)
    finished = torch.zeros(b, dtype=torch.bool, device=device)
    stopped = torch.zeros(b, dtype=torch.bool, device=device)
    lengths = torch.full((b,), max_new_tokens, dtype=torch.long, device=device)

    past = None
    cur_ids, cur_pos = input_ids, pos
    for t in range(max_new_tokens):
        with autocast_context(device, autocast):
            out = model(input_ids=cur_ids, attention_mask=attn, position_ids=cur_pos, past_key_values=past,
                        use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1)
        nxt = torch.where(finished, torch.full_like(nxt, pad_id), nxt)
        out_ids[:, t] = nxt
        hit = torch.isin(nxt, stop) & ~finished
        lengths = torch.where(hit, torch.full_like(lengths, t), lengths)
        stopped |= hit
        finished |= hit
        if bool(finished.all()):
            break
        cur_ids = nxt[:, None]
        cur_pos = cur_pos[:, -1:] + 1
        attn = torch.cat([attn, torch.ones((b, 1), dtype=attn.dtype, device=device)], dim=1)

    out_ids, lengths, stopped = out_ids.cpu(), lengths.cpu(), stopped.cpu()
    return [(out_ids[r, : int(lengths[r])].tolist(), bool(stopped[r])) for r in range(b)]


def greedy_generate(
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[Sequence[int]],
    *,
    max_new_tokens: int = 128,
    batch_size: int = 16,
    autocast: str = "auto",
) -> list[Generation]:
    """Greedy-decode tokenized prompts and decode them to raw strings."""
    raw = greedy_generate_ids(
        model,
        prompts,
        stop_ids=stop_token_ids(tokenizer),
        pad_id=pad_token_id(tokenizer),
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        autocast=autocast,
    )
    return [Generation(tokenizer.decode(ids, skip_special_tokens=False), ids, s) for ids, s in raw]


def generate_for_records(
    model: nn.Module,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    max_new_tokens: int = 128,
    batch_size: int = 16,
    autocast: str = "auto",
    date_string: str = CHAT_DATE_STRING,
    prompt_style: str = DEFAULT_PROMPT_STYLE,
) -> list[str]:
    """Raw completion strings for dataset records, using the exact training prompt.

    prompt_style must be the style the model was trained with (RunSpec.data.prompt_style).
    """
    prompts = [encode_prompt(tokenizer, r.get("posteam"), r["desc"], date_string, prompt_style) for r in records]
    gens = greedy_generate(model, tokenizer, prompts, max_new_tokens=max_new_tokens, batch_size=batch_size,
                           autocast=autocast)
    return [g.text for g in gens]
