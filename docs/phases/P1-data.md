# P1 (data half): ground truth, buckets, splits, scorer, validation, frozen dataset

**Date:** 2026-10-04 · **Status:** built and frozen; audit pending owner
spot-check · **Frozen eval:** `test.jsonl`, sha256
`13d0d714f5abc9e3e8869e6dfab87f6becaf25ba2d15e1f281af68299553038c`

This page explains how every play in the PlayParse dataset got its label, why each
rule is the way it is, how the labels were checked, and what is still known to be
imperfect. The eval harness and baselines (R0–R4) are the other half of P1 and are
not covered here.

## 1. What was built

| Module | Job |
|---|---|
| `playparse/data/load_pbp.py` | Reads nflverse parquet (about 70 of 372 columns) and applies the v1 scope filter |
| `playparse/data/ground_truth.py` | One play-by-play row → one `PlayLabel` (the rules in section 3) |
| `playparse/data/buckets.py` | One bucket per play, rarest first |
| `playparse/data/splits.py` | Season splits, the 50k sweep subset, and `eval_lite` |
| `playparse/data/build_dataset.py` | The CLI that writes the JSONL files and the manifest |
| `playparse/data/crosscheck.py` | Sums labels into game totals and compares them with official stats |
| `playparse/data/audit.py` | Draws the stratified audit sample and summarizes verdicts |
| `playparse/ffscore/scorer.py` | Credits → fantasy points under four scoring configs |

Reproduce everything (about one minute on an M4):

```bash
export PLAYPARSE_DATA_DIR=/path/to/data/raw        # pbp_<season>.parquet, stats_player_week_<season>.parquet
python -m playparse.data.build_dataset             # data/processed/*.jsonl + manifest.json, and results/p1/dataset_manifest.json
python -m playparse.data.crosscheck                # results/p1/crosscheck.json
python -m pytest -q                                # fast tests; add -m slow for full-data tests
```

The processed files are not in git (about 150 MB). Their row counts and sha256
hashes are in the committed [`results/p1/dataset_manifest.json`](../../results/p1/dataset_manifest.json).
A rebuild from the same raw files reproduces the same hashes byte for byte. The
slow test `test_committed_manifest_matches_processed_files` checks this.

## 2. Design decisions, and why

**Scope: `pass`, `run`, `no_play`, and two-point tries.** This is the PRD's scope.
Kickoffs, punts, field goals, extra points, kneels, and spikes are filtered out,
not labeled as empty. A two-point try is an ordinary `pass`/`run` row with
`two_point_attempt == 1`.

**Timeouts are not plays.** nflverse files every timeout ("Timeout #2 by BAL at
00:31.") as `no_play`. That was 20,009 rows, 44% of all `no_play` rows. They are
dropped: a `no_play` row whose text has no penalty in it, or that starts with
"Timeout" and never says "No Play", is not a play
([issue](../issues/p1-timeouts-filed-as-no-play.md)).

**Regular season and postseason are both kept.** Playoff plays are written in
the same language, the official stats include them, and dropping them would only
shrink the data. Postseason is about 4–5% of each split.

**Aborted plays are kept.** A fumbled snap is a real play with real stats.
nflverse files it as a `run`, and it is labeled like any other run.

**Kneels are excluded, and that costs a little.** A kneel is officially a rush
for about −1 yard. Excluding kneels means a QB's labeled game rushing total can be
a few yards above his official total. This is the only reason the dataset-scope
cross-check is 94.9% instead of 99.8%; see section 7.

**Labels come from nflverse's structured columns, not from parsing the text.**
Parsing the text would make the labels depend on the very skill being measured,
and errors would be invisible. The structured columns follow the official stat
rules. The cross-check shows they match the official totals.

**Names are written the way the text writes them.** The credited name is
nflverse's abbreviated `*_player_name` (for example `A.Brown`). The text prints
the same abbreviation after the jersey number (`11-A.Brown`). The builder checks
that every credited name literally appears in the play's text. After one fix
([issue](../issues/p1-name-spacing.md)) the rate is 100% on all splits. Within a
team and game, no two players share an abbreviation (0 collisions in 50,440
player-games). The NFL's own abbreviations disambiguate ("Do.Williams" vs
"Dam. Williams").

**Penalty yards are ignored** (PRD decision 2). They are not fantasy stats.

**Zero-yard credits are dropped.** "Pass ... for no gain" is labeled
`rec = 1` with no `rec_yds` credit. A zero credit scores nothing, and emitting
it would be a formatting convention the model must guess. The credit list stays
short. If the eval later wants zero credits, it is a one-line change.

**Replay reversals follow the final ruling.** The text prints the original call,
then "the play was REVERSED", then the final call. The label follows the final
call and is *not* nullified. The shared system prompt currently says the opposite;
see [issue](../issues/p1-prompt-reversal-wording.md). The owner needs to decide on
that wording.

## 3. Labeling rules

Each rule is pinned by a real-play fixture in `tests/test_ground_truth.py` (37
real rows; the expected labels were written from the text, not from the
builder's output).

| # | Situation | Label | Example (abridged) |
|---|---|---|---|
| 1 | Penalty wipes the play out (`play_type == 'no_play'`) | `nullified: true`, no credits | "... Offensive Holding, 10 yards, enforced at CAR 30 - No Play." |
| 2 | Completion | passer `pass_yds`, receiver `rec 1` and `rec_yds` | "B.Nix pass short left to J.Williams ... for 3 yards" → Nix 3, Williams 1 rec / 3 yds |
| 3 | Incompletion, sack, spike | no credits | "S.Darnold sacked at MIN 35 for -12 yards" → [] |
| 4 | Scramble or run | rusher `rush_yds` | "K.Murray scrambles ... for 12 yards" → Murray 12 |
| 5 | Interception | passer `int 1` (a return TD is defensive, out of scope) | "D.Jones pass ... INTERCEPTED ... for 10 yards, TOUCHDOWN" → Jones int |
| 6 | Touchdown | passer `pass_td` and scorer `rec_td`, or scorer `rush_td`; a teammate's fumble-recovery TD gets nothing | "K.Murray pass ... to Mi.Wilson for 5 yards, TOUCHDOWN" |
| 7 | Lateral | the last lateral's player gets `rec_yds` (no `rec`) or `rush_yds`; the play is flagged noisy | "A.St. Brown ... for 1 yard. Lateral to J.Gibbs for 20 yards, TOUCHDOWN" → Goff 21, St. Brown 1, Gibbs 20 + rec_td |
| 8 | Fumble lost | `fumble_lost 1` to the offensive player whose fumble the defense recovered, including a sacked QB, a center on a botched snap, or a lineman who recovered and fumbled again | "J.Allen sacked ... FUMBLES ... RECOVERED by ARI" → Allen fumble_lost |
| 8b | Fumble recovered by the offense | no fumble credit; yards still count | "J.Jacobs ... for 5 yards. FUMBLES ... recovered by GB-80-B.Melton" → Jacobs 5 |
| 9 | Two-point try | success: `two_pt 1` to passer **and** receiver, or to the rusher; failure: nothing; yards never count | "C.Williams pass to D.Swift is complete. ATTEMPT SUCCEEDS." → both two_pt |
| 9b | Two-point try followed by a dead-ball foul ("enforced between downs") | the try stands, even though nflverse says `no_play` | [issue](../issues/p1-two-point-dead-ball-penalty.md) |
| 10 | Penalty that does not wipe the play out (declined, offsetting, or enforced after) | credits as if no penalty, penalty yards ignored | "... for 41 yards. Penalty on TEN ..., declined." → full 41 |
| 11 | Replay reversal | the final ruling | "J.Gibbs ... for no gain ... REVERSED. J.Gibbs ... for 1 yard, TOUCHDOWN" → 1 yd + rush_td |
| 12 | Zero yards | the yardage credit is omitted | "J.Cook ... for no gain" → [] |
| 13 | Same player and yardage stat twice on one play | merged into one summed credit | |

### Official yards versus the text

Yardage follows the official stat rules. On two kinds of play these differ from
the "for N yards" printed in the text:

* **An offensive foul enforced from a spot downfield** (holding or an illegal
  block during the run). The gain is credited only up to that spot. For example,
  "J.Mixon left end to HOU 37 for 17 yards ... Offensive Holding, enforced at
  HOU 22" is labeled `rush_yds 2`.
* **A fumble that goes backward.** The gain is measured to where the ball was
  recovered. For example, "J.McLaughlin to DEN 48 for 2 yards. FUMBLES, RECOVERED
  by SEA at DEN 47" is labeled `rec_yds 1`.

The build measures how often this happens. The check covers plays with a single
ball carrier and compares the label with the first "for N yards" in the final
ruling.

| Bucket | Text disagrees (train) | Text disagrees (test) |
|---|---|---|
| normal | 0.0% of 188,476 | 0.0% of 24,558 |
| td | 0.0% of 9,742 | 0.0% of 1,309 |
| challenge | 3.5% | 3.7% |
| penalty_stands | 12.6% | 16.9% |
| fumble | 32.4% | 41.5% |

This is a property of the task, not a labeling error. The cross-check confirms
these labels match the official books. The text usually contains what is needed
to compute the official number: the yard line reached, the gain (which gives the
line of scrimmage), and the spot of enforcement or recovery. Doing so takes
field-position arithmetic, including across midfield. A regex that copies
"for N yards" will be wrong on about a third of fumble plays and one in seven
penalty plays. That makes these buckets good places to see where ML earns its
keep. See [issue](../issues/p1-official-yards-vs-text.md).

## 4. Buckets

Every play gets exactly one bucket. When several conditions apply, the rarest
wins. The order was **measured** on the training seasons, not guessed:

| Precedence | Bucket | Condition | Share of train (standalone) |
|---|---|---|---|
| 1 | lateral | nflverse lateral flag, or "Lateral to" / "Pass back to" in the text | 0.06% |
| 2 | two_point | a two-point try (including one wiped out by a penalty) | 0.35% |
| 3 | challenge | replay review or coach's challenge | 1.01% |
| 4 | interception | interception | 1.19% |
| 5 | fumble | any fumble, lost or not | 1.52% |
| 6 | penalty_stands | a penalty in the play (flag, or "Penalty on" in the text, which catches declined fouls), and the play counts | 2.49% |
| 7 | td | touchdown | 3.78% |
| 8 | penalty_nullified | the play was wiped out | 6.95% |
| 9 | normal | none of the above | 83.6% |

`penalty_stands` uses the text as well as the flag because nflverse sets
`penalty = 0` when every foul was declined, yet the model still has to read past
"Penalty on X, Defensive Holding, declined"
([issue](../issues/p1-declined-penalty-flag.md)).

## 5. Splits and subsets

| File | Seasons | Plays | Purpose |
|---|---|---|---|
| `train.jsonl` | 2015–2022 | 294,016 | training |
| `val.jsonl` | 2023 | 38,102 | early stopping, knob selection |
| `test.jsonl` | 2024 | 37,859 | **frozen eval**, final ladder numbers only |
| `train_50k.jsonl` | from train | 50,000 | sweeps (seed 20151, uniform over plays) |
| `eval_lite.jsonl` | from test | 2,795 | local MPS runs (seed 20241) |

**Plays per bucket:**

| Bucket | train | val | test | train_50k | eval_lite |
|---|---|---|---|---|---|
| lateral | 166 | 16 | 29 | 23 | 29 |
| two_point | 1,039 | 145 | 155 | 182 | 100 |
| challenge | 2,920 | 330 | 335 | 520 | 100 |
| interception | 3,353 | 435 | 393 | 571 | 100 |
| fumble | 4,034 | 523 | 530 | 719 | 100 |
| penalty_stands | 6,739 | 829 | 859 | 1,135 | 100 |
| td | 9,742 | 1,205 | 1,309 | 1,771 | 100 |
| penalty_nullified | 20,287 | 2,474 | 2,794 | 3,463 | 185 |
| normal | 245,736 | 32,145 | 31,455 | 41,616 | 1,981 |

**eval_lite** has two parts, recorded on each line as `eval_lite_part`.

* The `game` part is 18 whole test games (2,409 plays), drawn in a seeded random
  order. Whole games are needed so game-level fantasy-point MAE means something.
* The `topup` part is 386 extra plays, sampled from other games, so that every
  bucket has at least 100 plays. `lateral` contributes all 29 of its plays,
  because the whole test season has no more.

Game-level MAE on eval_lite must use the `game` part only. Per-bucket metrics use
both parts.

**Record format** (one JSON object per line, keys in this order):

```json
{"game_id":"2024_01_ARI_BUF","play_id":40,"season":2024,"week":1,"season_type":"REG",
 "posteam":"BUF","desc":"(14:11) ...","bucket":"normal",
 "label":"{\"nullified\":false,\"credits\":[...]}"}
```

`label` is the exact `PlayLabel.to_json()` string, which is also the training
completion. Lines are sorted by season, game, and play. Every label passes
`PlayLabel.from_json` (asserted during the build).

**Other dataset stats** (test split): 2,809 nullified plays (7.4%); 52,811
credits; credits per play are 0 (30%), 1 (37%), 3 (29%), or more. Label JSON
averages about 100 characters and the `desc` about 99 characters (p99 238).

## 6. The scorer

`score_label(label, config)` returns points per player for one play.
`score_game` and `score_games` aggregate plays per game. Configs:

| Config | pass yd | pass TD | INT | rush/rec yd | rush/rec TD | rec | fumble lost | 2pt |
|---|---|---|---|---|---|---|---|---|
| standard | 0.04 | 4 | −2 | 0.1 | 6 | 0 | −2 | 2 |
| half_ppr | 0.04 | 4 | −2 | 0.1 | 6 | 0.5 | −2 | 2 |
| ppr | 0.04 | 4 | −2 | 0.1 | 6 | 1 | −2 | 2 |
| espn_14team_ppr | 0.04 | 4 | −2 | 0.1 | 6 | 1 | −2 | 2 |

`espn_14team_ppr` is an **assumption**: ESPN's full-PPR defaults, which for v1
stats are identical to `ppr`. ESPN yardage bonuses and kicking or defense
settings are not modelled. If the owner's league differs, edit
`CONFIGS["espn_14team_ppr"]`. Yards are summed per game before multiplying, so
0.04/yd scoring matches official totals exactly. Applied to the official stat
lines, `ppr` reproduces nflverse's `fantasy_points_ppr` (minus return TDs) on 100%
of player-games.

## 7. Validation

### Game-level cross-check

Full detail: [docs/benchmarks/p1-crosscheck.md](../benchmarks/p1-crosscheck.md).

Over 2015–2024 (50,602 player-games, dataset scope plus kneels):

* **99.79%** of player-games agree with the official stats on all ten stats.
* PPR MAE is **0.0035 points** per player-game.
* Passing yards, passing TDs, interceptions, rushing and receiving TDs, and
  two-point conversions agree **100%**.
* Rushing yards agree 100.00%, receptions 99.98%, receiving yards 99.88%,
  fumbles lost 97.66% (99.88% counting lost fumbles that nflverse's fantasy
  formula leaves out).
* **0 unexplained** disagreements. What remains is multi-lateral plays (41
  cells), uncategorized lost fumbles (54), the text spelling one player two ways
  in a game (16), and one fake punt.
* On the dataset scope alone (no kneels), all-stat agreement is 94.93% and MAE
  is 0.0127. The difference is entirely QB kneels.

### Hand audit (model-audited, pending owner spot-check)

Full detail: [docs/benchmarks/p1-audit.md](../benchmarks/p1-audit.md).

200 test plays, stratified by bucket. **Error rate 2.5% overall, 21.7% in
`lateral` (5 of 23), 0% in every other bucket** (0 of 177). Four plays were
marked ambiguous: broken plays whose official yards follow a convention that the
text alone does not determine. Twenty plays are queued for the owner in
`results/p1/owner_spot_check.json`.

## 8. Known noise

| Source | Where | Size | Handling |
|---|---|---|---|
| Only the last lateral is recorded | `lateral` bucket | about 1 in 5 lateral labels wrong (audit); 41 cross-check cells over ten seasons | Own bucket, flagged noisy, error rate published |
| Official yards differ from "for N yards" | `fumble`, `penalty_stands`, some `challenge` | 32–42%, 13–17%, and 3.5% of checked plays | Correct by the official rules; a real difficulty of the task, documented |
| Broken-play conventions (aborted snaps, pitch-backs after a fumble) | `fumble`, `lateral` | 4 of 200 audited plays ambiguous | Documented |
| The text spells one player two ways in a game | any | 4 games in ten seasons | Label follows each play's spelling |
| Lost fumbles nflverse's fantasy formula omits | `fumble`, `lateral` | 54 player-games in ten seasons | Labels charge them; differs from nflverse by −2 points, usually for a lineman |
| Kneels excluded | QB game totals | about −1 yard per kneel | Scope choice; cross-check reports both scopes |

## 9. Issues found during P1-data

Each has a write-up with what happened, how it was found, the root cause, the fix,
and the guarding test.

| Issue | Found by | Status |
|---|---|---|
| [Two-point tries with a dead-ball foul labeled nullified](../issues/p1-two-point-dead-ball-penalty.md) | cross-check | fixed |
| [A player who fumbled twice was not charged](../issues/p1-same-player-fumbles-twice.md) | cross-check | fixed |
| [Credited "D. Thomas" not in the text "D.Thomas"](../issues/p1-name-spacing.md) | name-in-desc check | fixed |
| [Cross-check name normalization broke "A.St. Brown"](../issues/p1-crosscheck-name-normalization.md) | cross-check regression | fixed |
| [Timeouts filed as `no_play`](../issues/p1-timeouts-filed-as-no-play.md) | audit | fixed |
| [Declined fouls have `penalty == 0`](../issues/p1-declined-penalty-flag.md) | bucket tests | fixed |
| [Official yards vs the text](../issues/p1-official-yards-vs-text.md) | build check | documented |
| [nflverse's fantasy formula omits some lost fumbles](../issues/p1-uncategorized-fumbles.md) | cross-check | documented |
| [System prompt says reversed plays are nullified](../issues/p1-prompt-reversal-wording.md) | rule review | **open: owner decision** |

## 10. What the owner should know

1. **Decide the prompt wording before running any baseline.** The system prompt
   says reversed plays are nullified, and the labels say they are not. Prompt-only
   rungs will lose most reversed `challenge` plays for that reason alone.
2. **Spot-check 20 audit plays.** They are in `results/p1/owner_spot_check.json`.
   The audit is labeled provisional until then.
3. **Expect the `fumble` and `penalty_stands` buckets to be hard,** and not
   because the labels are wrong. Copying "for N yards" from the text fails on a
   third of fumble plays. Reading the label against the text on those buckets
   will often look wrong until you do the spot arithmetic.
4. **Read the `lateral` bucket with its 22% label-error rate beside it.** It has
   only 29 test plays.
5. **Check the ESPN scoring assumption** in `scorer.py` against your league.
6. **The frozen eval is `test.jsonl` with the hash at the top of this page.** It
   was frozen *after* the timeout fix, the last change to labels or scope. Any
   later change to `ground_truth.py`, `buckets.py`, `load_pbp.py`, or
   `splits.py` changes the hash. The manifest's `builder_hash` records which
   source produced the files.
7. **Two conventions are easy to flip if you disagree:** dropping zero-yard
   credits (rule 12), and charging lost fumbles that nflverse's fantasy formula
   omits (rule 8). Each is a few lines and has fixtures that will point at every
   case that changes.

## 11. Follow-ups

* Resolve the prompt-wording issue, then let the eval harness take the frozen
  `test.jsonl` hash from `results/p1/dataset_manifest.json`.
* Owner spot-check; update `audit_results.json` if verdicts change.
* Optional: a "text-derivable yards" variant of the labels for an ablation (does
  the model learn spot arithmetic, or only copy "for N yards"?). The build
  already flags the affected plays.
* Optional: include kneels in a v1.1 scope. They are trivial and would make
  dataset-scope game totals match the official ones exactly.
