# p1-eval: ground-truth conventions to reconcile with the dataset builder

## What happened

The eval was built in parallel with the ground-truth builder (`playparse/data/`,
owned by another workstream). To develop and measure the regex, this phase used an
*approximate* labeler (`scripts/regex_dev_estimate.py::approx_label`). Building it
surfaced conventions that the brief does not pin down. Each one changes exact match,
so the real builder and the baselines must agree on them.

## Open questions (the choice made here, and why)

1. **Zero-yard credits.** A completion for no gain is labelled `pass_yds 0, rec 1,
   rec_yds 0`, and a run for no gain is `rush_yds 0`. The structured columns hold 0,
   not null. Dropping zero credits would also make "no gain" look like "no play".
2. **Botched snaps.** `FUMBLES (Aborted)` with no later pass gets `rush_yds 0` for
   the quarterback, because nflverse lists him as the rusher with 0 yards.
3. **Two-point try with a dead-ball penalty "enforced between downs".** nflverse sets
   `play_type = no_play`, so a builder driven by that column marks the play
   nullified. In football terms the conversion counted. The regex credits the
   `two_pt`, so it disagrees with the approximate labeler on these plays. They are
   most of the 2.3% `two_point` misses.
4. **Laterals.** nflverse `receiving_yards` excludes yards gained after a lateral, and
   the lateral columns record only the last lateral. The approximate labeler gives the
   last lateral receiver `rec_yds` equal to `lateral_receiving_yards` and skips middle
   laterals. The regex credits every lateral. The PRD already treats this bucket as
   noisy, and it scores 37.8% exact match here.
5. **Fumble yardage** follows the official rule described in
   `p1-eval-official-yards-vs-stated-gain.md`. This comes from the columns, not from a
   choice made here.

## Next step

When `playparse/data/ground_truth.py` lands, rerun
`python -m playparse.eval.run --rung r0 --data <real train jsonl>` and compare the
per-bucket numbers with `results/r0_dev_train/result.json`. Any bucket that moves
points to a convention mismatch.

## Guarding test

None yet, because this needs the real builder. The regex behaviour for items 1–3 is
pinned by `tests/test_regex_parser.py`.
