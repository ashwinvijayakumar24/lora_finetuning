# T3b local pilot: field spots instead of yards, at P3 pilot scale

**Date:** 2026-10-08 · **Hardware:** Apple M4, 16 GB, MPS (no other heavy jobs) ·
**Model:** Llama 3.2 1B Instruct, bf16 base, fp32 LoRA r=16 · **Eval:** the P3
pilot's 1,014 plays from eval_lite (2024), scored only, never inspected play by
play · **CIs:** 95%, paired cluster bootstrap over games (2,000 resamples)

This is a smoke signal, not the pre-registered T3b test (PRD §17). That test is
the GPU run on all 294k train plays, scored once on the frozen test set.

## Headline

1. **Spots beat yards on the buckets they were meant to fix.** Against the v1
   minimal-prompt pilot (same 2,400 train plays, same recipe, same eval plays),
   the spot adapter gains **+16.0 points on penalty_stands** [+9.6, +22.7] and
   **+24.1 on lateral** [+5.9, +43.6]. Overall exact match is 94.8% vs 93.2%,
   **+1.6 [+0.3, +3.2]**.
2. **Fumbles did not move** (73% vs 76%, −3.0 [−12.4, +6.5]). On val, the
   fumble misses are reading errors, not arithmetic: picking the recovery spot
   when the official spot is the end of the run, crediting the defender who
   recovered, or missing the lost fumble. The spot to write is printed in the
   text in all of them.
3. **Touchdowns got worse (100% → 95%), and the cause is a format bug, now
   fixed.** The pilot used `"<defteam> 0"` for the touchdown spot. When desc
   never names the defense (a long touchdown from the offense's own half), the
   model wrote `"<posteam> 0"`, the offense's own goal line, which converts to a
   safety. On 160 val touchdown plays, all 6 misses are exactly this. The format
   now writes the opponent's goal line as the team-free `"OPP 0"`
   ([issue](../issues/t3b-goal-line-team-unseen.md)). The pilot was not rerun
   with the fix (compute budget); the GPU run uses it.
4. **At this scale, both regex arms still win.** R0 scores 97.6% and R0 + LOS
   99.1% on these plays; the spot adapter is −2.9 [−4.2, −1.7] and −4.3
   [−5.9, −3.1] behind. The pilot trains on 2,400 plays (147 fumbles); R5 had
   294k. The question is whether the gains in (1) hold at full scale and close
   the gap, which is what the GPU run decides.

## Per-bucket exact match (%)

| Bucket | n | **v2 spots** | v1 minimal | R0 | R0 + LOS | v2 − v1 [95% CI] |
|---|---|---|---|---|---|---|
| **overall** | 1014 | **94.8** | 93.2 | 97.6 | 99.1 | **+1.6 [+0.3, +3.2]** |
| fumble | 100 | 73.0 | 76.0 | 92.0 | 98.0 | −3.0 [−12.4, +6.5] |
| lateral | 29 | 65.5 | 41.4 | 62.1 | 82.8 | +24.1 [+5.9, +43.6] |
| challenge | 100 | 93.0 | 92.0 | 96.0 | 99.0 | +1.0 [−2.1, +4.3] |
| penalty_stands | 100 | 97.0 | 81.0 | 100.0 | 100.0 | +16.0 [+9.6, +22.7] |
| td | 100 | 95.0 | 100.0 | 100.0 | 100.0 | −5.0 [−9.2, −1.1] |
| penalty_nullified | 185 | 99.5 | 99.5 | 99.5 | 99.5 | 0.0 |
| normal | 200 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 |
| interception | 100 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 |
| two_point | 100 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 |

Every v2 output was valid (100%). Credit F1 is 94.7% (v1: 93.4%). Game
fantasy-point MAE is worse, 0.31 vs 0.23 points per player-game, because a
touchdown misread as a safety moves a player's yards by 70–80 at once.

Training: 300 optimizer steps (2,400 plays × 1 epoch, micro-batch 1 × 8), 40 min
wall, median 4.5 s per step (the v1 minimal pilot: 36 min). The val generation
metric ended at 92.2%, the same as the v1 minimal pilot's. Eval: 1.28 plays/s at
batch 16 (v1 minimal: 1.11), about 13 min.

## What the val misses show (2023, dev split)

A seeded set of 160 val touchdown plays and 80 val fumble plays
(`results/t3b/pilot/extra/valdiag_v2/`):

* **td 96.2%.** All 6 misses wrote `"<posteam> 0"` for a touchdown whose defense
  is never named, e.g. `los: PIT 26 ... 30-J.Warren right end for 74 yards,
  TOUCHDOWN` → `"to":"PIT 0"` (the gold is CLE's goal line).
* **fumble 66%.** The misses are about *which* spot counts, and the spot is
  printed every time: the model writes the recovery spot when the gain should
  stop at the end of the run (`to TB 34 ... RECOVERED by PHI at TB 37` → wrote
  TB 37), writes the touch spot for the passer but the end of the run for the
  receiver on the same play, credits the defender who recovered with yards,
  charges a lost fumble on a botched snap the offense recovered, or drops the
  lost fumble on an end-zone touchback. No miss is an arithmetic slip:
  the spot arithmetic is now code.

So the decomposition does what it should on penalties and laterals (spot fouls
and lateral legs are about copying the right printed spot), and exposes fumbles as
a *reading* problem: the official rules for which spot counts
([issue](../issues/t3b-official-fumble-yardage-conventions.md)) are subtle, and
147 training fumbles are not enough to learn them.

## Reproduce

```bash
python scripts/t3b_pilot.py data     # rebuilds the P3 pilot files, checks their sha256, adds v2 fields
python scripts/train.py --config configs/t3b_pilot.yaml --set 'data.val="data/processed_v2/val.jsonl"'
python -m playparse.eval.run --rung lora --adapter runs/t3b_pilot/train/checkpoints/step_0000300/adapter \
    --prompt-style minimal_v2 --schema v2 --data runs/t3b_pilot/data/eval_pilot_v2.jsonl \
    --out runs/t3b_pilot/eval/lora_v2 --dtype bfloat16 --batch-size 16
python -m playparse.eval.run --rung r0los --data runs/t3b_pilot/data/eval_pilot_v2.jsonl --out runs/t3b_pilot/eval/r0los
python scripts/t3b_pilot.py report
```

The pilot ran on the v2 files built before the `"OPP 0"` fix (pilot data hashes in
`results/t3b/pilot/data_manifest.json`); rerunning `data` now writes the fixed
format. Artifacts: `results/t3b/pilot/` (summary.json with every paired CI, the
eval predictions with each raw v2 output in `usage.raw_v2`, the training
metrics, and the val diagnostic).
