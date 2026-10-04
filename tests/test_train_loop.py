"""Training-loop tests on tiny random-init Llama models (CPU, seconds)."""
from __future__ import annotations

import json
import math

import pytest
import torch

from p2_fixtures import synthetic_examples, tiny_llama, tiny_lora
from playparse.train.collate import collate, count_target_tokens
from playparse.train.loop import (
    DataOrder,
    TrainConfig,
    accumulate_gradients,
    evaluate_loss,
    list_checkpoints,
    lr_lambda_factory,
    model_logits,
    token_loss_sum,
    train,
)

CPU = torch.device("cpu")


def grads(model) -> dict[str, torch.Tensor]:
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def trainable(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def max_rel_diff(a: dict, b: dict) -> float:
    num = math.sqrt(sum(float((a[k] - b[k]).pow(2).sum()) for k in a))
    den = math.sqrt(sum(float(a[k].pow(2).sum()) for k in a))
    return num / den


def cfg(tmp_path, **kw) -> TrainConfig:
    base = dict(output_dir=str(tmp_path), device="cpu", lr=1e-3, micro_batch_size=4, grad_accum_steps=1,
                max_steps=4, log_every=1, keep_last_checkpoints=None)
    base.update(kw)
    return TrainConfig(**base)


# ---------------------------------------------------------------------------
# Gradient accumulation
# ---------------------------------------------------------------------------


def _accum_setup():
    torch.manual_seed(0)
    model = tiny_lora(seed=0)
    exs = synthetic_examples(16, seed=3, completion_len=(1, 12))
    micro = [collate(exs[i : i + 4], pad_id=0) for i in range(0, 16, 4)]
    counts = [count_target_tokens(b["labels"]) for b in micro]
    assert len(set(counts)) > 1, "micro-batches must differ in target-token count for the test to bite"
    return model, exs, micro


def test_grad_accum_matches_one_big_batch():
    model, exs, micro = _accum_setup()

    # Reference: one batch of 16, mean over all its target tokens.
    model.zero_grad(set_to_none=True)
    big = collate(exs, pad_id=0)
    loss_sum, n = token_loss_sum(model_logits(model, big["input_ids"], big["attention_mask"]), big["labels"])
    (loss_sum / n).backward()
    g_ref = grads(model)
    ref_loss = float(loss_sum.detach()) / n

    # 4 micro-batches x accumulation 4, normalized by the step's total token count.
    model.zero_grad(set_to_none=True)
    loss, n_acc = accumulate_gradients(model, micro, CPU)
    g_acc = grads(model)

    assert n_acc == n
    assert loss == pytest.approx(ref_loss, rel=1e-6)
    assert all(g_ref[k].abs().sum() > 0 for k in g_ref)
    for k in g_ref:
        torch.testing.assert_close(g_acc[k], g_ref[k], rtol=1e-4, atol=1e-7)
    assert max_rel_diff(g_acc, g_ref) < 1e-5


def test_naive_per_micro_batch_mean_is_wrong():
    """The classic bug: mean loss per micro-batch, then divide by the number of
    micro-batches. Tokens in short micro-batches get extra weight."""
    model, exs, micro = _accum_setup()
    model.zero_grad(set_to_none=True)
    big = collate(exs, pad_id=0)
    loss_sum, n = token_loss_sum(model_logits(model, big["input_ids"], big["attention_mask"]), big["labels"])
    (loss_sum / n).backward()
    g_ref = grads(model)

    model.zero_grad(set_to_none=True)
    for b in micro:
        ls, nb = token_loss_sum(model_logits(model, b["input_ids"], b["attention_mask"]), b["labels"])
        (ls / nb / len(micro)).backward()
    g_naive = grads(model)
    assert max_rel_diff(g_naive, g_ref) > 1e-2  # orders of magnitude above float noise


def test_loop_step_with_accum_equals_big_batch_step(tmp_path):
    """One optimizer step: micro 4 x accum 4 lands on the same weights as micro 16."""
    exs = synthetic_examples(16, seed=5, completion_len=(1, 12))
    m1, m2 = tiny_lora(seed=0), tiny_lora(seed=0)
    train(m1, exs, None, cfg(tmp_path / "a", micro_batch_size=16, max_steps=1, save_final=False), pad_id=0)
    train(m2, exs, None, cfg(tmp_path / "b", micro_batch_size=4, grad_accum_steps=4, max_steps=1, save_final=False),
          pad_id=0)
    p1, p2 = trainable(m1), trainable(m2)
    for k in p1:
        torch.testing.assert_close(p1[k], p2[k], rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


def test_mask_prompt_false_changes_the_loss():
    model = tiny_lora(seed=0)
    masked = synthetic_examples(8, seed=1, mask_prompt=True)
    unmasked = synthetic_examples(8, seed=1, mask_prompt=False)
    assert [e.input_ids for e in masked] == [e.input_ids for e in unmasked]
    lm, nm = evaluate_loss(model, masked, CPU, 4, 0)
    lu, nu = evaluate_loss(model, unmasked, CPU, 4, 0)
    assert nu > nm
    assert abs(lm - lu) > 1e-3


def test_padding_does_not_change_the_loss():
    """Right padding plus the label mask: an example scores the same alone or batched."""
    model = tiny_lora(seed=0)
    exs = synthetic_examples(4, seed=2)
    alone = sum(evaluate_loss(model, [e], CPU, 1, 0)[0] * evaluate_loss(model, [e], CPU, 1, 0)[1] for e in exs)
    loss, n = evaluate_loss(model, exs, CPU, 4, 0)
    assert loss * n == pytest.approx(alone, rel=1e-5)


# ---------------------------------------------------------------------------
# Only the adapter trains
# ---------------------------------------------------------------------------


def test_only_requires_grad_params_change(tmp_path):
    model = tiny_lora(seed=0)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    train(model, synthetic_examples(16, seed=0), None, cfg(tmp_path, max_steps=3), pad_id=0)
    changed_trainable = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            changed_trainable += int(not torch.equal(p, before[n]))
        else:
            assert torch.equal(p, before[n]), f"frozen parameter {n} changed"
    assert changed_trainable == sum(1 for _, p in model.named_parameters() if p.requires_grad)


def test_no_trainable_params_raises(tmp_path):
    model = tiny_llama()
    for p in model.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError, match="requires_grad"):
        train(model, synthetic_examples(4), None, cfg(tmp_path), pad_id=0)


# ---------------------------------------------------------------------------
# Checkpoint and exact resume
# ---------------------------------------------------------------------------


def _train_losses(history):
    return {r["step"]: r["train_loss"] for r in history if r["event"] == "train"}


@pytest.mark.parametrize("fmt", ["safetensors", "peft"])
def test_resume_reproduces_uninterrupted_run(tmp_path, fmt):
    exs = synthetic_examples(24, seed=7)
    val = synthetic_examples(6, seed=8)
    # dropout > 0 so the RNG state matters; 8 steps over 24 examples crosses epochs.
    kw = dict(max_steps=8, micro_batch_size=3, grad_accum_steps=2, warmup_steps=2, eval_every=2, save_every=2)
    io = {}
    if fmt == "peft":
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        io = dict(
            save_fn=lambda m, p: m.save_pretrained(str(p)),
            load_fn=lambda m, p: set_peft_model_state_dict(m, load_file(str(p / "adapter_model.safetensors"))),
        )

    full = tiny_lora(seed=0, dropout=0.1)
    r_full = train(full, exs, val, cfg(tmp_path / "full", **kw), pad_id=0, **io)

    part = tiny_lora(seed=0, dropout=0.1)
    r_part = train(part, exs, val, cfg(tmp_path / "split", **kw), pad_id=0, stop_after_step=4, **io)
    assert r_part.step == 4 and list_checkpoints(tmp_path / "split")[-1].name == "step_0000004"

    # A fresh process: same frozen base, but adapter weights scrambled to prove they get loaded.
    resumed = tiny_lora(seed=0, dropout=0.1)
    with torch.no_grad():
        for n, p in resumed.named_parameters():
            if p.requires_grad:
                p.normal_()
    torch.manual_seed(12345)  # and a different RNG state, which the checkpoint must restore
    r_res = train(resumed, exs, val, cfg(tmp_path / "split", **kw), pad_id=0, resume_from="latest", **io)

    assert r_res.step == 8
    pf, pr = trainable(full), trainable(resumed)
    for k in pf:
        assert torch.equal(pf[k], pr[k]), k
    lf, lp, lr_ = _train_losses(r_full.history), _train_losses(r_part.history), _train_losses(r_res.history)
    assert {**lp, **lr_} == lf
    assert r_res.val_loss == r_full.val_loss


def test_checkpoint_contents(tmp_path):
    model = tiny_lora(seed=0)
    train(model, synthetic_examples(8), None, cfg(tmp_path, max_steps=2, save_every=1, keep_last_checkpoints=1),
          pad_id=0)
    ckpts = list_checkpoints(tmp_path)
    assert [c.name for c in ckpts] == ["step_0000002"]  # older ones pruned
    state = torch.load(ckpts[0] / "trainer_state.pt", weights_only=False)
    assert {"step", "optimizer", "scheduler", "rng", "early_stopping", "config"} <= set(state)
    assert state["step"] == 2
    from safetensors.torch import load_file

    saved = load_file(str(ckpts[0] / "adapter" / "trainable.safetensors"))
    assert set(saved) == set(trainable(model))  # adapter only: no frozen base weights


# ---------------------------------------------------------------------------
# It learns
# ---------------------------------------------------------------------------


def test_overfits_eight_examples(tmp_path):
    torch.manual_seed(0)
    # embed_std: see tiny_llama's docstring; with the default init LoRA plateaus at ~3.6.
    model = tiny_lora(vocab=64, seed=0, r=16, embed_std=0.5)
    exs = synthetic_examples(8, seed=11, completion_len=(2, 6))
    c = cfg(tmp_path, lr=1e-2, micro_batch_size=8, max_steps=150, warmup_steps=10, log_every=10,
            eval_every=50, save_final=False)
    res = train(model, exs, exs, c, pad_id=0)
    losses = _train_losses(res.history)
    assert losses[10] > 1.0
    assert res.val_loss < 0.05, res.val_loss


# ---------------------------------------------------------------------------
# Logging, schedule, early stopping, config, data order
# ---------------------------------------------------------------------------


def test_metrics_jsonl(tmp_path):
    model = tiny_lora(seed=0)
    train(model, synthetic_examples(8), synthetic_examples(4, seed=9),
          cfg(tmp_path, max_steps=4, eval_every=2), pad_id=0)
    recs = [json.loads(l) for l in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    train_recs = [r for r in recs if r["event"] == "train"]
    assert [r["step"] for r in train_recs] == [1, 2, 3, 4]
    for k in ("train_loss", "lr", "tokens_per_sec", "peak_mem_bytes", "grad_norm"):
        assert k in train_recs[0]
    assert [r["step"] for r in recs if r["event"] == "eval"] == [2, 4]


def test_lr_schedule_warmup_then_cosine():
    f = lr_lambda_factory(warmup_steps=4, total_steps=20, min_lr_ratio=0.1)
    assert [f(s) for s in range(4)] == [0.25, 0.5, 0.75, 1.0]
    assert f(4) == pytest.approx(1.0)
    assert f(12) == pytest.approx(0.1 + 0.9 * 0.5)
    assert f(20) == pytest.approx(0.1)
    vals = [f(s) for s in range(4, 21)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_lr_logged_follows_schedule(tmp_path):
    res = train(tiny_lora(seed=0), synthetic_examples(8), None,
                cfg(tmp_path, max_steps=6, warmup_steps=2, lr=1e-3, save_final=False), pad_id=0)
    lrs = [r["lr"] for r in res.history if r["event"] == "train"]
    f = lr_lambda_factory(2, 6, 0.0)
    assert lrs == pytest.approx([1e-3 * f(s) for s in range(6)])


def test_early_stopping_via_val_callback(tmp_path):
    scores = iter([0.5, 0.6, 0.55, 0.58, 0.9, 0.9])
    c = cfg(tmp_path, max_steps=6, gen_every=1, early_stop_metric="val_exact_match", early_stop_mode="max",
            early_stop_patience=2)
    calls = []

    def cb(model, step):
        assert not model.training
        calls.append(step)
        return {"val_exact_match": next(scores)}

    res = train(tiny_lora(seed=0), synthetic_examples(8), None, c, pad_id=0, val_callback=cb)
    assert res.stopped_early and res.step == 4
    assert res.best_metric == 0.6 and res.best_step == 2
    assert calls == [1, 2, 3, 4]


def test_should_stop_hook(tmp_path):
    res = train(tiny_lora(seed=0), synthetic_examples(8), None, cfg(tmp_path, max_steps=10), pad_id=0,
                should_stop=lambda step, m: step == 3)
    assert res.stopped_early and res.step == 3


def test_config_from_json_and_yaml(tmp_path):
    c = TrainConfig(lr=5e-4, max_steps=100, adam_betas=(0.9, 0.95))
    (tmp_path / "c.json").write_text(json.dumps(c.to_dict()))
    assert TrainConfig.from_file(tmp_path / "c.json") == c
    yaml = pytest.importorskip("yaml")
    (tmp_path / "c.yaml").write_text(yaml.safe_dump({"train": c.to_dict(), "lora": {"r": 16}}))
    assert TrainConfig.from_file(tmp_path / "c.yaml") == c
    with pytest.raises(ValueError, match="unknown"):
        TrainConfig.from_dict({"learning_rate": 1e-3})


def test_data_order_is_deterministic_and_covers_each_epoch():
    a, b = DataOrder(10, seed=3), DataOrder(10, seed=3)
    first = a.indices(0, 10)
    assert sorted(first) == list(range(10))
    assert a.indices(10, 10) != first and sorted(a.indices(10, 10)) == list(range(10))
    assert a.indices(7, 6) == b.indices(7, 6) == first[7:] + b.indices(10, 3)
    assert DataOrder(5, 0, shuffle=False).indices(3, 4) == [3, 4, 0, 1]


def test_same_seed_same_run(tmp_path):
    r1 = train(tiny_lora(seed=0, dropout=0.1), synthetic_examples(8), None, cfg(tmp_path / "1", save_final=False), pad_id=0)
    r2 = train(tiny_lora(seed=0, dropout=0.1), synthetic_examples(8), None, cfg(tmp_path / "2", save_final=False), pad_id=0)
    assert _train_losses(r1.history) == _train_losses(r2.history)


# ---------------------------------------------------------------------------
# Host-to-device copies (docs/issues/p2-mps-nonblocking-copy-race.md)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_eval_loss_independent_of_batch_size_on_mps():
    """A non_blocking CPU->MPS copy of a temporary batch raced with its deallocation
    and fed garbage labels to the loss. Counts and losses must not depend on batching."""
    mps = torch.device("mps")
    model = tiny_lora(vocab=64, seed=0).to(mps)
    exs = synthetic_examples(32, seed=4, prompt_len=(40, 120), completion_len=(5, 30))
    ref_loss, ref_n = evaluate_loss(model, exs, mps, 1, 0)
    assert ref_n == sum(e.n_completion for e in exs)
    for bs in (4, 8, 32):
        for _ in range(3):
            loss, n = evaluate_loss(model, exs, mps, bs, 0)
            assert n == ref_n
            assert loss == pytest.approx(ref_loss, rel=1e-3)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_accumulated_token_count_and_loss_match_cpu_on_mps():
    exs = synthetic_examples(16, seed=6, prompt_len=(40, 120), completion_len=(5, 30))
    micro = [collate(exs[i : i + 4], pad_id=0) for i in range(0, 16, 4)]
    cpu_model = tiny_lora(seed=0)
    loss_cpu, n_cpu = accumulate_gradients(cpu_model, micro, CPU)
    mps_model = tiny_lora(seed=0).to("mps")
    for _ in range(3):
        mps_model.zero_grad(set_to_none=True)
        loss_mps, n_mps = accumulate_gradients(mps_model, micro, torch.device("mps"))
        assert n_mps == n_cpu
        assert loss_mps == pytest.approx(loss_cpu, rel=1e-3)
