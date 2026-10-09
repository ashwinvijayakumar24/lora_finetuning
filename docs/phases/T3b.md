# T3b: let the model read the field spots and let code do the arithmetic

**Date:** 2026-10-08 · **Status:** built and tested; local pilot run; the
pre-registered GPU run and its single frozen-test eval are ready for PACE ·
**Pre-registration:** PRD §17 (written before any run; not changed here)

This page explains the T3b experiment for someone meeting it for the first time:
why splitting "reading" from "arithmetic" should help, what the model now outputs,
how we know the new labels mean exactly the same thing as the old ones, how the
regex baseline was given a fair chance, and what the GPU run will decide.

## 1. The question

R5 (the full LoRA run: r=16, minimal prompt, all 294k train plays) scores 98.5%
play exact match on eval_lite. The regex baseline R0 scores 99.1%. R5 ties the
regex on six of nine buckets and loses on three:

| Bucket | R5 | R0 |
|---|---|---|
| fumble | 85 | 92 |
| penalty_stands | 94 | 100 |
| lateral | 45 | 62 |

These are exactly the plays where the official yards are not the number printed
in the text. The official stat rules measure the gain to a *field spot* (the spot
of an offensive foul, the spot where a fumble was recovered), and the text names
that spot but never subtracts it ([issue](../issues/p1-official-yards-vs-text.md)):

```
20-T.Pollard right tackle to NYG 23 for 5 yards. PENALTY on DAL-70-Z.Martin,
Offensive Holding, 10 yards, enforced at NYG 25.
```

The line of scrimmage was NYG 28 (not printed). The official rush is NYG 28 → NYG
25 = **3 yards**, not the 5 the text says. To get this right, v1 asks a 1B model to
(a) notice that the foul spot is the one that counts, (b) work out where the play
started from "to NYG 23 for 5 yards", and (c) subtract across team-relative
coordinates. Only (a) is reading. (b) and (c) are arithmetic.

**Hypothesis (PRD §17).** The adapter's remaining errors are arithmetic, not
reading. If the model outputs the spots that bound each yardage credit and plain
code computes the yards, an adapter trained exactly like R5 beats the regex.

### Why decomposition should help

A language model produces numbers one token at a time, conditioned on what it has
seen. Copying a span it can see ("NYG 25") is the thing such models do best:
attention can point straight at it. Subtracting two field positions that live in
different coordinate frames ("NYG 28" is 72 yards from DAL's goal line; "NYG 25"
is 75) is a small algorithm the model has to run inside its activations, and a 1B
model trained on 294k examples has seen comparatively few of the hard cases (fumble
is 1.4% of train, lateral 0.06%). Moving the subtraction into code removes the part
the model is bad at and keeps the part it is good at. The project has applied the
same principle from the start: the model never computes fantasy points, a scorer
does (PRD §4).

How much reading is left? The v2 builder measures, for each ground-truth spot,
whether it is printed in the play text (§3.3). On fumble and penalty_stands plays
the spot is printed 89–100% of the time (the rest are touchdowns, whose spot is a
goal line). So on the buckets R5 loses, v2 turns arithmetic into copying.

## 2. What the model outputs now (schema v2)

### 2.1 The spot format

A *spot* is a field position written the way nflverse's `desc` writes it:

| Spot | Meaning |
|---|---|
| `"DAL 22"` | 22 yards from DAL's own goal line, on DAL's half |
| `"50"` | midfield (desc writes "to 50"; nflverse's `yrdln` writes "MID 50"; a few 2015–16 descs write "PIT 50"; all parse to `"50"`) |
| `"PHI 0"` | PHI's goal line, desc's own form ("INTERCEPTED ... at PHI 0"). A safety ends at `"<posteam> 0"` |
| `"OPP 0"` | the opponent's goal line, where a touchdown ends. No team name, because desc often never names the defense (§3.3; this was a bug in the first version, found by the pilot) |

Team abbreviations in desc and in nflverse's `posteam`/`defteam` columns are the
same strings in every season (checked on 2015, 2016, 2019, 2022 and 2023: the only
desc spot tokens that are neither team are "QB" in phrases like "in at QB 4"). The
arithmetic only needs to know whether a spot is on the offense's half, so every
team other than `posteam` counts as the opponent.

`playparse/ffscore/spots.py` holds the arithmetic:

* `spot_to_yardline100(spot, posteam)`: yards from the spot to the goal line the
  offense attacks (nflverse's `yardline_100` convention: 0 = touchdown, 100 = own
  goal line). `"PHI 30"` for PHI is 70; `"DAL 22"` for PHI is 22.
* `yardline100_to_spot(y, posteam, defteam)`: the inverse.
* `yards_between(from, to, posteam)` = `yardline100(from) − yardline100(to)`.
  `"PHI 30" → "DAL 22"` is 70 − 22 = **48**; `"DAL 22" → "DAL 30"` is −8.

### 2.2 The label

The input gains one field, the line of scrimmage (`los`, nflverse `yrdln` in desc
spelling). The output keeps v1's shape. Yardage credits (`pass_yds`, `rush_yds`,
`rec_yds`) carry `"to"` (where the credited run of the ball ended) and, only when
it did not start at the line of scrimmage, `"from"`. Every other credit keeps v1's
`"value"`.

```
posteam: NYJ
los: NYJ 25
desc: 14-S.Darnold pass short left to 11-Ro.Anderson to NYJ 33 for 8 yards. Lateral
      to 81-Q.Enunwa to NYJ 35 for 2 yards. FUMBLES, RECOVERED by BUF-93-T.Murphy at NYJ 21. ...
```

```json
{"nullified":false,"credits":[
  {"player":"S.Darnold","stat":"pass_yds","to":"NYJ 21"},
  {"player":"Ro.Anderson","stat":"rec","value":1},
  {"player":"Q.Enunwa","stat":"rec_yds","from":"NYJ 33","to":"NYJ 21"},
  {"player":"Ro.Anderson","stat":"rec_yds","to":"NYJ 33"},
  {"player":"Q.Enunwa","stat":"fumble_lost","value":1}]}
```

`to_v1` turns that into the v1 label: Darnold 25 → 21 = −4, Anderson 25 → 33 = 8,
Enunwa 33 → 21 = −12. Two v1 conventions are kept: credits for the same player and
yardage stat are summed, and a yardage credit worth 0 is dropped (so "back to the
line of scrimmage" means "no credit").

The contract lives in `playparse/ffscore/schema_v2.py`: a strict parser (as strict
as v1's), canonical serialization for training targets (sorted like v1, `from`
before `to`), `to_v1`, `from_v1`, and `v2_text_to_v1_text` for raw model output.

### 2.3 Where `from` is needed: laterals

Only lateral legs start somewhere other than the line of scrimmage. nflverse
credits only the first carrier and the last lateral taker. The v2 ground truth
starts the last taker where the first carrier's credited yards ended (line of
scrimmage + the first carrier's official yards), which desc prints on a
one-lateral play ("to NYJ 33 for 8 yards. Lateral to ..."), and puts `to` at that
start plus his official yards. Train has 66 plays with a `from` spot; val 8.

On a play with several laterals, the middle legs are credited to nobody, so this
`to` is generally not a printed spot: it is one of the 4–6% "computed" lateral
spots in §3.3. Choosing `from` as the end of the leg right before the last lateral
would make both spots printed on most such plays, but it needs the middle legs,
which nflverse does not record (the builder would have to parse them from the
text). It touches about a dozen train plays, so it was left as is; the pilot and
the GPU run use this definition.

## 3. Ground truth and the round-trip invariant

### 3.1 Derived, not re-labeled

The v1 labels are frozen (test sha256 `13d0d714…`) and stay the only source of
truth. The v2 label is *derived* from them: for each yardage credit, `to` = the
start spot advanced by the credited official yards. Nothing about yards is
re-decided. The builder needs four extra nflverse columns, joined on
(game_id, play_id): `yrdln`, `yardline_100`, `defteam` and the lateral columns
(`playparse/data/ground_truth_v2.py`, `playparse/data/build_dataset_v2.py`).

### 3.2 The round-trip invariant

If v2 is only a change of format, converting the v2 ground truth back must give
the v1 ground truth exactly:

> `to_v1(label_v2, los, posteam) == label_v1` for every play.

This is what makes T3b a fair comparison: an adapter scored through `to_v1` is
scored by the same metric, on the same frozen file, against the same ground truth
as every other rung. If the invariant failed on some play, a perfect v2 model
would be marked wrong there, and the comparison would be biased against T3b.

**Result: 100.0000% on every split** (train 294,016/294,016, val 38,102/38,102,
test 37,859/37,859, and the train_50k and eval_lite subsets). No play needed a
spot off the field, and every play has a line of scrimmage (even nullified ones).
The slow test `test_round_trip_every_play` re-checks it from the files.

### 3.3 Is the model reading or still computing?

For every ground-truth `to` spot (train and val only; the test split is not
analysed), the builder asks whether desc prints it:

* **literal**: the spot is printed in desc (`"NYG 25"` in "enforced at NYG 25");
* **goal line**: an unprinted goal line, which desc signals in words (TOUCHDOWN,
  SAFETY);
* **computed**: neither; the model would have to compute it.

| Bucket | Yardage credits (train) | Literal | Goal line | Computed | Val literal |
|---|---|---|---|---|---|
| normal | 261,941 | 100.0% | 0.0% | 0.0% | 100.0% |
| td | 15,939 | 0.0% | 100.0% | 0.0% | 0.0% (100% goal line) |
| penalty_stands | 8,193 | 90.3% | 9.7% | 0.0% | 88.6% |
| challenge | 2,921 | 80.2% | 19.7% | 0.1% | 87.0% |
| fumble | 2,629 | 98.8% | 1.2% | 0.0% | 100.0% |
| lateral | 311 | 94.2% | 1.6% | 4.2% | 93.8% |

So on every bucket R5 loses, the spot the model must write is printed in the
text, or is a goal line, on at least 95% of credits. Only multi-lateral plays
(4–6% of lateral credits) still need arithmetic. If T3b fails on fumbles and
penalties, the cause is reading (which spot counts), not arithmetic.

**A format bug the pilot found.** The first version wrote a touchdown's spot as
`"<defteam> 0"`, desc's spelling of a goal line. But on 6.6% of touchdown credits
the defense's abbreviation appears nowhere in the input (a long touchdown from the
offense's own half: `los: PIT 26 ... for 74 yards, TOUCHDOWN`). There the pilot
model wrote the one team it could see, `"PIT 0"`: the offense's *own* goal line,
which converts to a safety. All 6 touchdown misses on 160 val plays were this.
The opponent's goal line is now the team-free `"OPP 0"` (the arithmetic already
treats any non-offense team as the opponent, so the converter is unchanged); the
dataset was rebuilt with the round trip still 100%
([issue](../issues/t3b-goal-line-team-unseen.md)).

### 3.4 Files

`python -m playparse.data.build_dataset_v2` reads the v1 files (never writes them)
and writes `data/processed_v2/{train,val,test,train_50k,eval_lite}.jsonl`
(git-ignored): every v1 record key, then `los` and `label_v2`. The committed
manifest `results/t3b/dataset_v2_manifest.json` holds the sha256 of every v1
input and v2 output, the round-trip counts, and the literal-spot table. v2 eval_lite
holds the same 2,795 plays with the same v1 labels as v1 eval_lite.

v2 examples are 8.7 tokens longer on average (121.5 vs 112.8 with the minimal
prompt; longest 432, under `max_len` 640): `results/t3b/token_lengths.json`.

## 4. Training and eval plumbing

Every rule below exists so that a v2 adapter is trained, validated and scored
exactly like R5 except for the output format, and can never be paired with the
wrong format by accident.

* **Prompt style `minimal_v2`** (`playparse/prompt.py`): R5's minimal prompt (no
  system message) with one more line in the user turn, `los: NYJ 25`. The prompt
  style fixes the schema: `full` and `minimal` are v1, `minimal_v2` is v2. The two
  existing styles render byte-identically, and ignore `los` even on v2 records.
* **Collate** trains a `minimal_v2` run on `label_v2` and refuses records with no
  `los` or no `label_v2` (a v1 file fed to a v2 run).
* **`RunSpec.data.schema`** (`v1` | `v2`) must agree with the style; an old
  run_spec.json without the field gets its style's schema.
* **Validation during training** (`val_eval.as_v1_texts`) converts each v2
  generation to v1 text before the unchanged v1 metrics score it.
* **Eval** (`--prompt-style minimal_v2 --schema v2`, needs a v2 data file):
  `HFPredictor` converts each output with `v2_text_to_v1_text` before the
  unchanged harness sees it, so every metric keeps its v1 definition. The raw v2
  text is kept in `predictions.jsonl` (`usage.raw_v2`). An output that is not
  valid v2 becomes text with no JSON object, so it counts as invalid; it is never
  rescued by happening to be valid v1 (for example a yardage credit with a
  `"value"`). Text around the JSON object is kept, so `strict_valid` means the same
  thing.
* **Guards**: the eval CLI refuses a style/schema mismatch and a v2 run on a file
  without `los`; `LoRAPredictor` refuses an adapter whose run_spec.json says a
  different style or schema.
* **Parity**: training prompt ids equal eval prompt ids for `minimal_v2`,
  including a left-padded batch (tests/test_t3b_plumbing.py).
* **Config**: `configs/t3b_spots.yaml` is `configs/train_default.yaml` (R5's
  recipe) with the v2 data paths, `prompt_style: minimal_v2`, `schema: v2` and an
  output dir. A test pins that nothing else differs.
* **PACE post-eval**: `train_h100.sbatch` reads `data.schema` from the run's
  run_spec.json and scores a v2 run on `data/processed_v2/eval_lite.jsonl` with
  `--schema v2`.

## 5. The fairness arm: R0 + LOS

T3b's adapter sees the line of scrimmage; R0 never did. A win could then come from
the extra field rather than from the decomposition. So the pre-registration gives
the regex the same field with a fixed 2-hour engineering budget, developed on
train and val only (`--rung r0los`, `playparse/eval/baselines/regex_los.py`).

**Time log:** started 21:34, rules finished 21:50, tests and dev artifacts by
about 21:55 on 2026-10-08: roughly 20 minutes of the 2 hours. The rules stopped
paying off before the budget ran out: the remaining val misses are multi-lateral
plays and odd sequences (a backward pass after a deflection, a fumble recovered and
re-fumbled twice).

R0 + LOS runs the unchanged R0 parser, then recomputes the ball carrier's yards as
"spot minus line of scrimmage" for the spot the official rules pick (backward
fumbles to the first touch, forward bounces floored at the line, botched snaps,
handoffs after a recovered fumble, lateral chains, offensive spot fouls). These
rules are written up in [an issue](../issues/t3b-official-fumble-yardage-conventions.md).
It also counts end-zone touchbacks as lost fumbles, reads "Officially, a rush for
N yards", and strips two substitution notes R0 missed; those three need no line of
scrimmage, but they are part of the arm's two hours.

| Bucket | val n | R0 | R0 + LOS | train R0 | train R0 + LOS |
|---|---|---|---|---|---|
| **overall** | 38,102 | 99.78 | **99.94** | 99.84 | **99.95** |
| fumble | 523 | 91.20 | 99.24 | 94.27 | 99.03 |
| lateral | 16 | 43.75 | 68.75 | 47.59 | 75.30 |
| challenge | 330 | 98.48 | 99.70 | 99.14 | 99.90 |
| td | 1,205 | 99.67 | 99.92 | 99.93 | 99.95 |
| normal | 32,145 | 99.95 | 99.98 | 99.97 | 99.99 |
| penalty_stands | 829 | 99.88 | 99.88 | 99.90 | 99.90 |

No train play that R0 gets right is made wrong. The arm is a much stronger
baseline than R0: on the 1,014-play pilot eval set it scores 99.1% vs R0's 97.6%.
**This raises the bar T3b has to clear.**

## 6. The local pilot (a smoke signal, not the test)

Same 2,400 enriched train plays and recipe as the P3 prompt-ablation pilot
(minimal prompt), same 1,014 eval plays from eval_lite; only the v2 data, style
and schema differ. Full write-up: [benchmark](../benchmarks/t3b-pilot.md).

| Bucket | n | v2 spots | v1 minimal | R0 | R0 + LOS | v2 − v1 [95% CI] |
|---|---|---|---|---|---|---|
| **overall** | 1014 | **94.8** | 93.2 | 97.6 | 99.1 | +1.6 [+0.3, +3.2] |
| fumble | 100 | 73.0 | 76.0 | 92.0 | 98.0 | −3.0 [−12.4, +6.5] |
| lateral | 29 | 65.5 | 41.4 | 62.1 | 82.8 | +24.1 [+5.9, +43.6] |
| penalty_stands | 100 | 97.0 | 81.0 | 100.0 | 100.0 | +16.0 [+9.6, +22.7] |
| challenge | 100 | 93.0 | 92.0 | 96.0 | 99.0 | +1.0 [−2.1, +4.3] |
| td | 100 | 95.0 | 100.0 | 100.0 | 100.0 | −5.0 [−9.2, −1.1] |
| other four | 585 | 99.8 | 99.8 | 99.8 | 99.8 | 0.0 |

What it says, in order of confidence:

1. **Decomposition fixes the spot-foul and lateral arithmetic.** penalty_stands
   +16 points and lateral +24, both with CIs clear of 0. These are the plays
   where the right spot is printed and v1 had to subtract.
2. **Fumbles are a reading problem.** No change (−3, CI ±10). On val the misses
   pick the wrong printed spot (the recovery spot instead of the end of the run),
   credit the recovering defender, or get the lost-fumble flag wrong. With 147
   training fumbles the official rules are not learned; the GPU run has 4,034.
3. **The touchdown loss is the format bug above**, already fixed; the pilot was
   not rerun (compute budget).
4. **The regex arms still win at this scale** (R0 by 2.9 points, R0 + LOS by 4.3).
   The pilot trains on 1% of the data, so this says little about the GPU run.

## 7. What the GPU run will decide

The pre-registered pass condition (PRD §17): on the frozen test set, the
spot-decomposed adapter's overall exact match exceeds R0's **and** R0 + LOS's,
each with a paired, game-clustered 95% CI that excludes 0, and no bucket regresses
beyond its CI. `scripts/t3b_compare.py` applies it mechanically (a bucket
"regresses beyond its CI" when its whole paired-difference CI is below 0) and
prints the per-bucket table either way.

Possible outcomes and what each would mean:

* **Earned.** Decomposition closes the gap: the 1B adapter's remaining errors were
  arithmetic, and reading plus code beats a strong regex.
* **Beats R0 but not R0 + LOS.** The gain came from the line of scrimmage, which
  helps a regex as much as a model. The honest conclusion is that the field, not
  the learning, did the work.
* **Not better than R0.** The errors were reading after all (which spot counts),
  or the adapter simply does not see enough fumble and lateral plays.

Commands (for the orchestrator; the frozen test set is scored once):

```bash
# copy the v2 files to PACE (git-ignored, 208 MB; or rebuild there with build_dataset_v2 if the raw parquet is present)
rsync -av data/processed_v2/ <pace>:~/ps-simpliearn-0/llm_finetuning/data/processed_v2/
# train (R5's recipe, v2) and self-score on v2 eval_lite -> results/eval_lite/t3b_spots/
sbatch -J pp-t3b_spots scripts/slurm/train_h100.sbatch configs/t3b_spots.yaml runs/t3b_spots
# afterwards, once: the frozen test set (v2 copy: same plays and labels as test.jsonl)
sbatch -J pp-t3b-test scripts/slurm/eval.sbatch runs/t3b_spots/best data/processed_v2/test.jsonl \
    results/t3b_test --prompt-style minimal_v2 --schema v2
sbatch -J pp-r0los-test scripts/slurm/eval.sbatch - data/processed_v2/test.jsonl results/r0los_test --rung r0los
# verdict
python scripts/t3b_compare.py --data data/processed_v2/test.jsonl --arm t3b=results/t3b_test \
    --arm r0=results/r0_test --arm r0los=results/r0los_test --out results/t3b/test_comparison.json
```

## 8. Tests

| File | What it pins |
|---|---|
| tests/test_t3b_spots.py | spot parsing and arithmetic: midfield, both halves, goal lines, negative yards, the inverse on every yard |
| tests/test_t3b_schema_v2.py | strict parser, canonical form, `to_v1`, invalid v2 never rescued, v2 ground truth on 9 real 2019 plays (laterals, spot foul, backward fumble, touchdown, safety, midfield) |
| tests/test_t3b_dataset_v2.py | the builder on a tiny set; slow: round trip on every play, manifest hashes, v2 eval_lite = v1 eval_lite |
| tests/test_t3b_plumbing.py | styles and schemas, v1 styles unchanged, train/eval parity for minimal_v2, val callback and predictor scoring, guards, config = R5 + v2, sbatch post-eval |
| tests/test_t3b_regex_los.py | R0 + LOS on 11 real train plays, agreement with R0 on ordinary plays |
| tests/test_t3b_compare.py | the pass rule: a win, a regression, a tie |

## Issues found in T3b

* [t3b-goal-line-team-unseen](../issues/t3b-goal-line-team-unseen.md): the first
  format's touchdown spot named a team the input often omits, and the pilot model
  wrote a safety instead; fixed with `"OPP 0"` before the GPU run.
* [t3b-official-fumble-yardage-conventions](../issues/t3b-official-fumble-yardage-conventions.md):
  the official rules for fumble and lateral yardage, measured from the line of
  scrimmage, as found while building R0 + LOS.
