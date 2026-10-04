# P1 label audit: 200 test plays read against their labels

> **Status: model-audited, pending owner spot-check.** The PRD asks the owner to
> audit by hand. This pass was done by Claude, reading each play's `desc` against
> its label. Twenty plays are queued for the owner in
> [`results/p1/owner_spot_check.json`](../../results/p1/owner_spot_check.json)
> (listed at the end of this page). Treat the numbers below as provisional until
> that spot-check agrees.

**Date:** 2026-10-04
**Sample:** [`results/p1/audit_sample.jsonl`](../../results/p1/audit_sample.jsonl),
200 plays from `test.jsonl` (2024), stratified by bucket, seed 20240
(`python -m playparse.data.audit sample`).
**Verdicts:** [`results/p1/audit_results.json`](../../results/p1/audit_results.json),
one verdict and note per play.

## Headline

Outside the `lateral` bucket, **no label contradicted its text** (0 of 177). Inside
`lateral`, 5 of 23 labels were wrong, all for the reason the PRD predicted:
nflverse records only the last lateral. Four labels across the sample follow an
official stat convention that the text alone does not determine.

| Bucket | Plays | Correct | Wrong | Ambiguous | Error rate | 95% upper bound |
|---|---|---|---|---|---|---|
| lateral | 23 | 15 | 5 | 3 | **21.7%** | 42% |
| two_point | 23 | 23 | 0 | 0 | 0% | 14% |
| challenge | 22 | 22 | 0 | 0 | 0% | 15% |
| interception | 22 | 22 | 0 | 0 | 0% | 15% |
| fumble | 22 | 21 | 0 | 1 | 0% | 15% |
| penalty_stands | 22 | 22 | 0 | 0 | 0% | 15% |
| td | 22 | 22 | 0 | 0 | 0% | 15% |
| penalty_nullified | 22 | 22 | 0 | 0 | 0% | 15% |
| normal | 22 | 22 | 0 | 0 | 0% | 15% |
| **all** | 200 | 191 | 5 | 4 | **2.5%** | 5.7% |

The error rate counts only "wrong". The upper bound is the 95% Wilson interval's
upper end, a reminder that 22 plays cannot rule out an error rate of a few
percent. The cross-check ([p1-crosscheck.md](p1-crosscheck.md)) is the stronger
evidence for the common buckets because it covers every play.

## How each play was judged

* **Correct.** The label is what the text says, under the labeling rules in
  `playparse/data/ground_truth.py`. This includes the official yardage rules
  when the text gives enough to compute them. For example, "J.Mixon left end to
  HOU 37 for 17 yards ... Offensive Holding ... enforced at HOU 22" is labeled
  `rush_yds 2`. The run started at HOU 20, so the spot of the foul is 2 yards
  downfield. That counts as correct.
* **Wrong.** The label contradicts the text.
* **Ambiguous.** The label matches the official books, but the text alone does
  not determine it. A careful reader would need to know the convention.

## The wrong labels (all `lateral`)

| Audit id | What the text says | What the label says |
|---|---|---|
| 5 | Six laterals: R.Bell 12, E.Saubert −3 then +20, B.Aiyuk 5, J.Mason 1, D.Puni −3, J.Brendel 3 | R.Bell 12 and J.Brendel 2 only |
| 6 | A.Cooper "for −2 yards", lateral to J.Allen "for 9 yards, TOUCHDOWN" | Cooper 0 rec yards, Allen 7 (pass yards and TD right) |
| 8 | Q.Johnston 26, J.Palmer 9, L.McConkey −6, J.Dobbins 10 | Palmer and McConkey missing |
| 9 | C.Kirk 3, G.Davis −7, T.Lawrence −4 | G.Davis missing |
| 18 | Five laterals including Z.Frazier −5 and J.Fields +3 | Frazier and Fields missing |

These cannot be fixed from nflverse's columns, because only the last lateral is
recorded. Fixing them would mean parsing the text, which would make the labels
depend on the thing being evaluated. The decision is to keep them, keep them in
their own bucket, and report the lateral bucket's numbers with this error rate
beside them.

**What it means for evaluation.** On `lateral` plays, a model that reads the text
perfectly will be scored wrong about one time in five. Do not read a low lateral
exact-match score as a model failure without checking against this rate. The
bucket is tiny (29 test plays), so it barely moves any overall number.

## The ambiguous labels

| Audit id | Bucket | The convention |
|---|---|---|
| 11 | lateral | The center fumbles the snap, the QB recovers and runs "for 3 yards". He is credited 1, measured from the line of scrimmage. |
| 13 | lateral | A lateral, then a fumble recovered behind the play. The official yards use a spot the text does not make clear. |
| 15 | lateral | The QB fumbles, recovers, then pitches back. Only the last ball carrier is credited, with the net gain from the line of scrimmage. |
| 97 | fumble | Aborted snap; a running back recovers and runs "for −3 yards". He gets no credit. Compare id 11, where the QB's run after a recovery *is* credited. |

All four involve broken plays where the ball changes hands behind the line. They
are a handful of plays per season and are left as they are.

## What the audit found that the automated checks could not

The first draw of this sample contained six timeouts ("Timeout #2 by BAL at
00:31.") among 22 `penalty_nullified` plays. nflverse files timeouts as
`no_play`, and the label "nullified, no credits" is internally consistent, so no
automated check flagged them. They were 44% of all `no_play` rows. They are now
filtered out at load time, and the dataset was rebuilt before freezing
([issue](../issues/p1-timeouts-filed-as-no-play.md)). The sample was then
redrawn with the same seed. The first seven buckets drew the same plays (their
pools did not change), and `penalty_nullified` and `normal` were re-audited.

## Owner spot-check (20 plays)

The owner should read these 20 and fill in `owner_verdict` in
`results/p1/owner_spot_check.json`. They are chosen to test the model auditor
where it is most likely to be wrong: every non-correct verdict, the trickiest
"correct" ones (spot arithmetic, reversals, unusual two-point tries), and four
plain plays as a control.

| Audit id | Bucket | Why it is on the list |
|---|---|---|
| 5, 6 | lateral | Model said wrong (multi-lateral; lateral yardage split) |
| 11, 13, 15 | lateral | Model said ambiguous (broken-play conventions) |
| 97 | fumble | Model said ambiguous (aborted snap, teammate's run not credited) |
| 1, 19 | lateral | Correct only with fumble-recovery-spot arithmetic |
| 33 | two_point | Pass to one player, lateral to another; who gets the two points |
| 115, 116, 119 | penalty_stands | Spot-foul arithmetic; a touchdown nullified by a holding call |
| 48, 64 | challenge | Reversals that turn a touchdown or a "down by contact" into a lost fumble |
| 46 | challenge | Forward fumble out of the end zone (touchback) counted as lost |
| 96 | fumble | Aborted snap recovered by a teammate with no advance; no credits |
| 24, 37 | two_point | Failed tries (a sack, and a completion that fell short) |
| 128 | penalty_stands | Offsetting fouls; the run stands |
| 177 | penalty_nullified | Plain wiped-out run, as a control |

If the owner disagrees with any verdict, update `audit_results.json` and rerun
`python -m playparse.data.audit summarize` to recompute the table above.
