<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->

# Build datasets and train a candidate

`learn` does everything below in one run. Use the individual commands to build a dataset
with a different view, to retrain on the same data with other settings, or to see what a
stage produced. Datasets and their views are described in the [datasets
reference](../reference/datasets.md); the [release gate](../concepts/release-policy.md)
decides whether a candidate is ever used.

```bash
splinter dataset build 5c0f2a91 --view sft-final
splinter train 9d41e7b3 --steps 200
splinter train 9d41e7b3 --rehearsal 2b6c88f0 --rehearsal-share 0.25
splinter select 4a77d1c5 --exam-set 8e10b2f6
```

## Defaults

Unless `--steps` is given a run's step budget
is three passes over what it learned, within bounds, at one learning rate
suited to a short LoRA run for `train` and `learn` alike (`--lr`, 2e-4),
a LoRA alpha of twice the rank (`--alpha`, so the update's scale does not
change with the rank) and no weight decay (`--weight-decay`), all recorded
in the candidate. Each pass takes every record once in a fresh order, a step
is a mean over its supervised tokens, the rate warms up over a twentieth of
the steps and decays over all of them, and a run that stops on its plateau
first cools the rate down to the floor. A step averages several records - one per
sixteen the data holds, up to eight, or `--records-per-step N` - because an
update on a single long, individual answer is noise the next record undoes.
The budget is a ceiling: the run is watched as it trains and
carries the step with the best monitoring loss, stopping early once that
loss stops improving.

## Regimes, held-out records and what a candidate continues

`train` trains the regime its datasets' objective names - chat datasets
by supervised fine-tuning, preference datasets by DPO against the model
the run starts from (`--beta`, the DPO temperature, defaults to brain's);
one run's datasets are all one or all the other. It concatenates them in
order and holds the newest records out for scoring (a record of the writer's
own text, which has no question to examine, is held out with its family and
scored by nobody): a supervised candidate
reports the held-out loss of base and candidate, a preference candidate
brain's preference score on the held-out pairs (how often, and by how
many nats, it prefers the chosen answer more than its reference does).

## Watching the training and choosing the step

A supervised run is watched as it trains. A share of the training families
(`--monitor-share`, default 0.1, whole families) is set aside as the
monitoring records - carved from the training half, never from the held-out
half the gate and the exam decide on, so no step is chosen on the evidence
it is then judged by - and scored every `--eval-every` steps (default:
sixteen evaluations over the step budget, and at the last step). The
candidate carries the adapter of the evaluation with the lowest monitoring
loss, not the last step's, and the run stops once that loss has gone
`--patience` evaluations (default 4) without improving; `--patience 0` runs
the whole budget and still carries the best, `--eval-every 0` monitors
nothing and carries the last step. `--steps` is the budget, the most steps
the run may take: by default three passes over the examples. The candidate's
report shows the steps trained of the budget, the step carried and why, the
training and monitoring loss at it and the gap between them, the whole curve
(step, mean training loss since the previous evaluation, monitoring loss),
and warns when the gap at the carried step is more than a quarter of the
monitoring loss, when the monitoring loss rose after the step carried, or
when it was still falling at the end of the budget; the release gate and the
exam repeat those warnings beside their verdicts.
`--keep-evaluations` also keeps the adapter of every evaluation with the
candidate, and `splinter select CANDIDATE --exam-set DEV-SUITE` puts each of
them to the dev suite (the families `learn --dev-families` reserves apart from
the final test and from all training) and chooses one by a rule fixed in
advance: an evaluation whose answers state what the source does not hold in
more than 8% of the answers checked is out; of those left, the best share of
answers the judge calls right and every one within a standard error of it are
in the running; of those, the writer's own text of the dev families gets the
lowest loss. Correctness decides, voice breaks the tie it cannot, and an
evaluation that invents is never chosen for its voice. `--adopt` makes the
chosen evaluation a candidate of its own (its held-out score is not carried
over: it was measured of another adapter) to release; `learn --select-on-dev`
does both in the run, between training and the exam.

## Replay and rehearsal

From `policy:<alias>` (the default) it continues the adapter of the
release the alias points at - never the base weights once a release
exists - and a supervised run replays `--replay-fraction` (default 0.25)
of every earlier release's trained-on chat records, a sample drawn with the
run's seed (so the same seed replays the same records) that never includes
what that release held out;
replayed records are never held out, and take a quarter of the training
draws however many there are, so a large replay cannot starve the new
records. `--rehearsal DATASET-ID` mixes a rehearsal dataset (`rehearse`; see
`learn`) in the same way at `--rehearsal-share` of the draws (default 0.25;
the replay's quarter and this share must leave the new records some draws),
never held out, and puts `--monitor-share` of its records into the
monitoring set beside the training families' records, so the step carried is
chosen on the mixture the run trains. The candidate records what it
rehearsed (dataset, share, records trained on and monitored), and the records
mixed in are kept as an artifact the gate's leakage check reads. `--seed N`
seeds the adapter's initialisation, the batch order and the replay draw;
without it a seed is drawn, printed (`no --seed given, using seed N`), written
to the run record and to the candidate (`seed`), and passing it again writes
the same adapter on the same data. Two runs that differ only in the seed show
how much of a measured difference is the draw. A preference run replays and rehearses nothing:
brain's preference trainer trains on its pairs alone. `--from
local:<checkpoint>+<adapter>` continues that adapter instead, with no
replay, and such a candidate cannot replace a champion. The candidate
records its regime; the release gate grades either the same way.

## Seeds

Every random choice a pipeline makes takes a seed the run states. The exam's
families (`--exam-seed`, also on `exam-set create --seed` and, for `practice`,
kept with the families so every later run reads it back and naming another is
refused), the rehearsal's form tasks (`--rehearsal-seed`), the denoise tasks
(`--denoise-seed`), the subjects of `--strip mix:F` (`--mix-seed`) and the
exam's bootstrap intervals (`--interval-seed`) are drawn when no seed is given,
printed (`no --exam-seed given, using seed N (pass --exam-seed N to
reproduce)`), written to the run record (`seeds`) and kept in the artifact they
decide: the reservation, the frozen exam, the rehearsal and task reports, the
dataset manifest and the exam report. The release gate's bootstrap intervals
(the tasks' own evidence, the anchor's drop) are resampled from a seed of their
own, recorded as `gate-interval` and kept in the gate report and so in the
release manifest; `release --interval-seed N` passes it again. A local model
that samples draws a seed for each request that names none; the run that made
the requests records them in order (`runs show` lists the count, the first
sixteen and a digest of all of them, and the whole list is kept in the run's
events), and a seed passed back as the request's replays its decoding. Greedy
decoding draws nothing. Passing the seed again reproduces the choice. Two runs
reserve different families unless they share a seed, so
results are comparable across runs only when each names a recorded one.
