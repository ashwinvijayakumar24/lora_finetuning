"""Shared fixtures for the P2 training tests (imported by test files, not a conftest,
so it cannot collide with other phases' test setup)."""
from __future__ import annotations

import random

import pytest
import torch

from playparse.ffscore.schema import Credit, PlayLabel
from playparse.paths import WEIGHTS
from playparse.train.collate import EncodedExample, make_example


def load_tokenizer():
    if not (WEIGHTS / "tokenizer.json").exists():
        pytest.skip(f"real tokenizer not found under {WEIGHTS}")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(WEIGHTS))


def record(desc: str, label: PlayLabel, posteam: str = "PHI", **extra) -> dict:
    return {
        "game_id": "2023_01_X_Y",
        "play_id": extra.pop("play_id", 1),
        "season": 2023,
        "week": 1,
        "season_type": "REG",
        "posteam": posteam,
        "desc": desc,
        "bucket": "pass",
        "label": label.to_json(),
        **extra,
    }


def sample_records() -> list[dict]:
    return [
        record(
            "(3:12) 1-J.Hurts pass short right to 11-A.Brown for 14 yards (24-J.Smith).",
            PlayLabel(False, (Credit("J.Hurts", "pass_yds", 14), Credit("A.Brown", "rec", 1), Credit("A.Brown", "rec_yds", 14))),
        ),
        record("(1:05) 26-S.Barkley up the middle for 3 yards.", PlayLabel(False, (Credit("S.Barkley", "rush_yds", 3),)), play_id=2),
        record(
            "(0:44) 1-J.Hurts pass incomplete deep left. PENALTY on PHI-65-L.Johnson, Offensive Holding, 10 yards, enforced at PHI 30 - No Play.",
            PlayLabel(True, ()),
            play_id=3,
        ),
    ]


def tiny_llama(vocab: int = 64, seed: int = 0, layers: int = 2, hidden: int = 32, embed_std: float | None = None):
    """A random-init Llama small enough to train in milliseconds on CPU.

    embed_std rescales the (tied) embedding matrix. With the default init (std 0.02)
    the tied output head can only produce logits within about +-0.6, so a frozen
    base plus LoRA (which cannot touch embeddings or the head) has a loss floor near
    the unigram entropy. See docs/issues/p2-tiny-lora-loss-floor.md.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=True,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(cfg)
    model.config._attn_implementation = "eager"
    if embed_std is not None:
        with torch.no_grad():
            model.get_input_embeddings().weight.normal_(0, embed_std)
    return model


def tiny_lora(vocab: int = 64, seed: int = 0, r: int = 4, dropout: float = 0.0, layers: int = 2,
              embed_std: float | None = None):
    """tiny_llama + PEFT LoRA on all linear layers: the stand-in for playparse.lora."""
    from peft import LoraConfig, get_peft_model

    base = tiny_llama(vocab=vocab, seed=seed, layers=layers, embed_std=embed_std)
    torch.manual_seed(seed + 1)
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=dropout, target_modules="all-linear", init_lora_weights=True)
    model = get_peft_model(base, cfg)
    # PEFT inits B = 0, which makes A's gradient exactly zero at step 0. Give B a
    # small random value so gradient tests exercise every adapter parameter.
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02)
    return model


def synthetic_examples(
    n: int,
    vocab: int = 64,
    seed: int = 0,
    prompt_len: tuple[int, int] = (4, 12),
    completion_len: tuple[int, int] = (1, 9),
    eot: int = 1,
    bos: int = 0,
    mask_prompt: bool = True,
) -> list[EncodedExample]:
    """Examples with variable prompt and completion lengths over a toy vocabulary.

    Ids 0 and 1 are reserved for BOS and EOT; content ids are drawn from 2..vocab-1.
    Variable completion lengths are what make per-micro-batch averaging wrong.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        p = [bos] + [rng.randrange(2, vocab) for _ in range(rng.randint(*prompt_len))]
        c = [rng.randrange(2, vocab) for _ in range(rng.randint(*completion_len))] + [eot]
        out.append(make_example(p, c, mask_prompt=mask_prompt))
    return out
