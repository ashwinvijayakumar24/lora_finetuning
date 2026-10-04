"""Greedy generation: matches Hugging Face generate, batching is invisible, stops on eot."""
from __future__ import annotations

import random

import pytest
import torch

from p2_fixtures import load_tokenizer, sample_records, synthetic_examples, tiny_llama, tiny_lora
from playparse.train.generate import generate_for_records, greedy_generate_ids
from playparse.train.loop import TrainConfig, train


def _prompts(n=6, vocab=64, seed=0):
    rng = random.Random(seed)
    return [[0] + [rng.randrange(2, vocab) for _ in range(rng.randint(3, 15))] for _ in range(n)]


@pytest.fixture(scope="module")
def memorized(tmp_path_factory):
    """A tiny LoRA model trained until it reproduces 8 completions exactly."""
    exs = synthetic_examples(8, seed=11, prompt_len=(3, 14), completion_len=(2, 7))
    model = tiny_lora(vocab=64, seed=0, r=16, embed_std=0.5)
    cfg = TrainConfig(output_dir=str(tmp_path_factory.mktemp("mem")), device="cpu", lr=1e-2, micro_batch_size=8,
                      max_steps=150, warmup_steps=10, save_final=False)
    train(model, exs, None, cfg, pad_id=0)
    return model.eval(), exs


def test_trained_model_reproduces_completions_and_stops_on_eot(memorized):
    model, exs = memorized
    prompts = [e.input_ids[: e.n_prompt] for e in exs]
    out = greedy_generate_ids(model, prompts, stop_ids=[1], pad_id=0, max_new_tokens=12, batch_size=8)
    for e, (ids, stopped) in zip(exs, out):
        assert stopped
        assert ids + [1] == e.input_ids[e.n_prompt :]  # completion, ending in the eot id 1


def test_batching_is_invisible_and_matches_hf_generate(memorized):
    model, exs = memorized
    prompts = [e.input_ids[: e.n_prompt] for e in exs] + _prompts(4)  # unseen prompts too
    assert len({len(p) for p in prompts}) > 3  # left padding really happens
    batched = greedy_generate_ids(model, prompts, stop_ids=[1], pad_id=0, max_new_tokens=10, batch_size=12)
    single = [greedy_generate_ids(model, [p], stop_ids=[1], pad_id=0, max_new_tokens=10)[0] for p in prompts]
    assert batched == single
    for p, (ids, stopped) in zip(prompts, single):
        ref = model.generate(input_ids=torch.tensor([p]), attention_mask=torch.ones(1, len(p), dtype=torch.long),
                             do_sample=False, max_new_tokens=10, eos_token_id=[1], pad_token_id=0)
        assert ref[0, len(p):].tolist() == (ids + [1] if stopped else ids)


def test_stops_on_any_stop_token(memorized):
    model, exs = memorized
    p = exs[0].input_ids[: exs[0].n_prompt]
    full = exs[0].input_ids[exs[0].n_prompt :]
    target = full[1]  # pretend the second completion token is a stop token
    (ids, stopped), = greedy_generate_ids(model, [p], stop_ids=[1, target], pad_id=0, max_new_tokens=12)
    assert stopped and ids == full[: full.index(target)]
    (ids, stopped), = greedy_generate_ids(model, [p], stop_ids=[1], pad_id=0, max_new_tokens=1)
    assert not stopped and ids == full[:1]


def test_works_with_peft_model_and_restores_train_mode():
    model = tiny_lora(seed=0)
    model.train()
    out = greedy_generate_ids(model, _prompts(3), stop_ids=[1], pad_id=0, max_new_tokens=5)
    assert len(out) == 3 and model.training


def test_generate_for_records_with_real_tokenizer():
    tok = load_tokenizer()
    model = tiny_llama(vocab=len(tok), seed=0, hidden=16, layers=1)
    texts = generate_for_records(model, tok, sample_records(), max_new_tokens=4, batch_size=2)
    assert len(texts) == 3 and all(isinstance(t, str) for t in texts)
