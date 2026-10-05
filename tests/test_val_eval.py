"""The generation-based val callback: stratified subset, scoring, and its use for
early stopping and best-checkpoint selection inside train()."""
from __future__ import annotations

import json
from collections import Counter

import pytest

from p2_fixtures import synthetic_examples, tiny_llama
from playparse.ffscore.schema import Credit, PlayLabel
from playparse.train import val_eval
from playparse.train.build import LoRASpec, apply_lora
from playparse.train.loop import TrainConfig, train
from playparse.train.val_eval import harness_val_callback, score_generations, stratified_subset

GOLD = PlayLabel(False, (Credit("A.Brown", "rec", 1), Credit("A.Brown", "rec_yds", 9))).to_json()
OTHER = PlayLabel(False, (Credit("A.Brown", "rec", 1), Credit("A.Brown", "rec_yds", 8))).to_json()


def _pop() -> list[dict]:
    sizes = {"normal": 500, "fumble": 40, "lateral": 3, "td": 60}
    recs = []
    for b, k in sizes.items():
        for i in range(k):
            recs.append({"game_id": f"G{i % 7}", "play_id": len(recs), "bucket": b, "desc": f"{b} {i}",
                         "posteam": "PHI", "label": GOLD})
    return recs


def test_balanced_subset_is_seeded_and_fills_from_small_buckets():
    pop = _pop()
    a = stratified_subset(pop, 40, seed=1)
    assert a == stratified_subset(pop, 40, seed=1)
    assert a != stratified_subset(pop, 40, seed=2)
    c = Counter(r["bucket"] for r in a)
    assert sum(c.values()) == 40
    assert c["lateral"] == 3  # all of a tiny bucket
    assert c["normal"] >= 12 and c["fumble"] >= 12 and c["td"] >= 12  # leftover spread over the rest
    assert len({r["play_id"] for r in a}) == 40


def test_proportional_subset_keeps_every_bucket():
    c = Counter(r["bucket"] for r in stratified_subset(_pop(), 60, seed=0, strategy="proportional"))
    assert sum(c.values()) == 60 and set(c) == {"normal", "fumble", "lateral", "td"}
    assert c["normal"] > c["td"] > c["lateral"]


def test_subset_larger_than_population_returns_everything():
    pop = _pop()[:10]
    assert len(stratified_subset(pop, 50)) == 10
    assert stratified_subset(pop, 0) == []


def test_score_generations_matches_hand_count():
    recs = [{"bucket": "normal", "label": GOLD}] * 3 + [{"bucket": "fumble", "label": GOLD}] * 2
    texts = [GOLD, GOLD, "not json", GOLD, OTHER]
    m = score_generations(recs, texts, population_shares={"normal": 0.9, "fumble": 0.1})
    assert m["val_gen_n"] == 5
    assert m["val_exact_match"] == pytest.approx(3 / 5)
    assert m["val_valid_rate"] == pytest.approx(4 / 5)
    assert m["val_exact_match/normal"] == pytest.approx(2 / 3)
    assert m["val_exact_match/fumble"] == pytest.approx(1 / 2)
    assert m["val_n/fumble"] == 2
    assert m["val_exact_match_macro"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert m["val_exact_match_natural"] == pytest.approx(0.9 * 2 / 3 + 0.1 * 1 / 2)
    # credits: 10 gold; preds 6 correct from 3 exact GOLD... + OTHER has 1 TP, 1 FP, 1 FN; "not json" 2 FN
    tp, fp, fn = 2 * 3 + 1, 1, 2 + 1
    assert m["val_credit_f1"] == pytest.approx(2 * tp / (2 * tp + fp + fn))


def test_callback_drives_early_stopping_and_best_adapter(tmp_path, monkeypatch):
    """Per-bucket exact match is a usable early_stop_metric; the best adapter is saved in PEFT format."""
    recs = stratified_subset(_pop(), 8, seed=0)
    # Fake generation: the share of correct fumble outputs rises, then falls.
    schedule = iter([0.0, 0.5, 1.0, 0.5, 0.5, 0.5])

    def fake_generate(model, tok, records, **kw):
        frac = next(schedule)
        fumbles = [i for i, r in enumerate(records) if r["bucket"] == "fumble"]
        good = set(fumbles[: int(round(frac * len(fumbles)))])
        return [r["label"] if (r["bucket"] != "fumble" or i in good) else "{}" for i, r in enumerate(records)]

    import playparse.train.generate as gen

    monkeypatch.setattr(gen, "generate_for_records", fake_generate)
    cb = harness_val_callback(None, recs, population=_pop(), predictions_dir=tmp_path / "preds")
    model, save_fn, load_fn, _ = apply_lora(tiny_llama(seed=0), LoRASpec(r=4, alpha=8, dropout=0.0))
    cfg = TrainConfig(output_dir=str(tmp_path), device="cpu", max_steps=6, micro_batch_size=4, gen_every=1,
                      early_stop_metric="val_exact_match/fumble", early_stop_mode="max", early_stop_patience=2,
                      save_best=True, keep_last_checkpoints=None)
    res = train(model, synthetic_examples(8), None, cfg, pad_id=0, val_callback=cb, save_fn=save_fn,
                load_fn=load_fn)
    assert res.stopped_early and res.step == 5
    assert res.best_step == 3 and res.best_metric == 1.0
    assert (tmp_path / "best" / "adapter_config.json").exists()
    logged = [json.loads(l) for l in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    cbs = [r for r in logged if r["event"] == "val_callback"]
    assert [r["val_exact_match/fumble"] for r in cbs] == [0.0, 0.5, 1.0, 0.5, 0.5]
    assert all("val_exact_match_natural" in r and "val_n/lateral" in r for r in cbs)
    assert sorted(p.name for p in (tmp_path / "preds").iterdir())[0] == "step_0000001.jsonl"


def test_module_exports():
    assert set(val_eval.__all__) == {"stratified_subset", "score_generations", "harness_val_callback"}
