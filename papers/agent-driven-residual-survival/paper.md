<!--
SPDX-License-Identifier: CC-BY-4.0
Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

Swedish Embedded AB implements solutions for reproducible, locally operated
machine-learning research agents for its clients. If your team needs expertise
in survival modelling or agent evaluation, you can procure our services by
sending an email to info@swedishembedded.com.
-->

# Is There Learnable Structure Beyond a Strong Additive Survival Model? A Locally Operated, Supervised Research Agent Tests It on Single-Visit Cohort Data

**Martin Schröder**
Swedish Embedded AB, <info@swedishembedded.com>

*Working draft, started 2026-10-07. Status: milestone 0 (protocol). Every numerical
claim in this document is either reproduced by a command listed in the
Reproducibility section or is explicitly marked as unmeasured.*

## Abstract

Deep survival models have beaten classical baselines where patients number in
the hundreds of thousands and events in the billions. We ask what happens at
the scale of one examination per person (about 56 thousand adults and 8.4
thousand deaths from US national survey cycles linked to mortality records),
and we ask it with a method that makes the experiments themselves auditable:
a locally served 27-billion-parameter coding agent carries out delegated
tasks under the supervision of a stronger model, every run is kept in a
hash-chained event log, and results with help are recorded apart from results
without it. On the survival question, a regularised Cox model that is
nonlinear in every continuous input (written by the agent) is the best model
we measured by point estimate; it is not resolvably better than the same model
without splines when all inputs are used (difference in integrated Brier
score -0.00013, 95% interval -0.00057 to +0.00031), and a deep set encoder
over the same inputs is resolvably worse than it (+0.00111, +0.00019 to
+0.00204). Learning curves from one cross-validation repeat show the Cox
models flat from half of the training subjects and the more flexible learners
still improving, so more data of this kind would help them and not the Cox
models. The experiment that would test for structure beyond the additive model,
a residual neural network, was not delivered by the agent in four attempts and
is reported as not done. On the agent question, two of four substantive tasks
were accepted and two were not. Before the first could even be judged, four
defects of the surrounding infrastructure (a reasoning block spending the
output budget, greedy decoding repeating one action, a server context
overflow, and an unreadable old log format) had to be found and repaired; two
defects of the supervisor's own acceptance checks surfaced later. Each repair
has a regression specification. A candidate adapter trained on nine accepted runs
was judged on eight held-out tasks under a rule fixed beforehand and rejected
(0 of 8 against 1 of 8), leaving the model in use unchanged.

## 1. Introduction

Deep survival and event-sequence models have produced large gains over
classical baselines where patients number in the hundreds of thousands to
millions and timestamped events in the billions. Our own campaign works at a
very different scale: one examination per person, about 8.4 thousand deaths,
and a model of about 85 thousand parameters. Initial results showed that no
baseline we tried was resolvably better than the neural model, and the neural
model was not resolvably better than a regularised additive model.

We therefore reframe the research question from "which neural architecture is
best on this data" to three separable questions:

1. **Nonlinear residual signal.** Does any learnable structure remain once a
   strong additive survival model has been fitted?
2. **Longitudinal signal.** Does repeated-measure information add predictive
   value beyond the current state?
3. **Sufficient data.** Is there enough data to learn what exists?

A single-visit cohort can answer only the first question, and this paper is
organised around answering it with an experimental protocol that is fixed
before the results are seen.

A second, methodological thread runs through the paper. The experiments are
carried out by a locally served coding agent under the supervision of a
stronger model, with every run recorded in an append-only trace. We report
supervisor interventions separately from unaided results, because the claim of
interest is how much of the work the local agent can do without help.

## 2. Related Work

**Foundation models for time-to-event data.** MOTOR pretrains a time-to-event
transformer on tens of millions of patient records and billions of clinical
events and reports gains in time-dependent concordance and label efficiency on
downstream survival tasks (arXiv:2301.03150).
SurvivEHR pretrains a competing-risks next-event model on 7.6 billion coded
events from 23 million primary-care patients (Oxford and Birmingham, 2025,
medRxiv). Delphi-2M trains a generative health-event transformer on 402,799
UK Biobank participants and, in a scaling experiment, finds that about two
million parameters is optimal for that dataset ("Learning the natural history of human disease with generative transformers", 2025). These
results locate the regime where representation learning over event
sequences pays: very many patients, many events per patient. Our cohort has
one examination per person, so we test the weaker claim that nonlinear
structure beyond a strong additive model exists at all.

**Survival evaluation.** We use the Aalen-Johansen estimator for observed
cause-specific risk, inverse-probability-of-censoring weights for
concordance and Brier scores (Uno et al.), and the corrected resampled
t-test of Nadeau and Bengio for repeated cross-validation, whose variance
inflation accounts for the overlap between training sets. Sample-size
guidance for validating a survival prediction model depends on the
distribution of predicted risks, the incidence and censoring, and the
precision wanted (Riley et al., 2022, Statistics in Medicine); we do not
claim a universal events threshold.

*(Further related work to be added and each reference checked against its
source before it is cited: neural versus Cox survival models on benchmark
data, numerical feature embeddings for tabular deep learning, NHANES
mortality-prediction and biological-age studies, coding agents with
execution-based validation.)*

## 3. Data and Estimands

**Cohort.** The ten continuous cycles of the US National Health and
Nutrition Examination Survey (1999 to 2018), linked to the public-use
mortality file with follow-up to 31 December 2019. Every examined adult
eligible for linkage is a subject: 56,253 subjects, 8,355 deaths. Each
subject has one examination; the timeline holds what was measured or answered
then, the eating-time features of the 24-hour dietary recalls, the ages at
which conditions were first diagnosed and recalled body weights as history,
and death in three competing cause groups (cardiovascular, cancer, other).
A locked test set of 8,437 subjects was scored once under an earlier
pre-registration and is not used here.

**What the data cannot show.** Death is the only outcome observed after the
examination. Follow-up is administratively censored, so a cycle examined in
2015 is followed for about four years; a metric at horizon h is computed on
the cycles whose survivors were all followed h years, and the 15-year metric
rests on the 1999 to 2002 cycles alone (1,583 subjects, 345 deaths in the
primary analysis). The mortality file is perturbed by the data owner for some
records. Ages are top-coded. Associations are estimated, not effects of
changing a behaviour.

**Estimands.** For each subject the all-cause cumulative incidence by years 1
to 15, and the incidence of each cause where the model gives it. Discrimination
is Uno's concordance at 5 and 10 years; accuracy is the inverse-probability-of-
censoring-weighted integrated Brier score over 0 to 15 years (IBS) and the Brier
score at 10 years; calibration at 10 years is the weighted intercept and slope
with an Aalen-Johansen observed risk per equal-weight risk group. Metrics are
design-weighted with the survey weights; training is unweighted.

**Folds.** Five repeats of five-fold cross-validation over the non-locked
subjects (25 folds), fixed in a frozen partition file. Every prediction used
here is an out-of-fold prediction, produced by a model fitted on the other
folds of its repeat.

## 4. Methods

### 4.1 Models compared

All models predict the same cumulative-incidence curves from the same inputs
and are evaluated by the same code.

* **age-sex** and **standard**: additive piecewise-exponential hazard models
  on age and sex, or on the conventional risk factors (reference points).
* **additive**: an additive proportional-hazards model (a piecewise-exponential
  generalised additive model) on every input; no interactions. About 16 thousand
  parameters.
* **horizon**: a set encoder over the subject's tokens (variable identity,
  value, time) with attention, trained with the same likelihood and a masked
  value objective; about 86 thousand parameters at its default.
* **Conventional baselines** fitted in Python on the same folds: elastic-net
  Cox on the standard or all inputs, cause-specific Cox, inverse-probability-
  weighted logistic models, gradient-boosted survival trees.
* **Spline Cox** (added during this work): the elastic-net Cox model with a
  cubic B-spline basis on every continuous input, to establish the nonlinear
  additive ceiling.
* **Ensembles and blends**: equal-weight averages of members' predicted curves.

### 4.2 Evaluation protocol

Differences between two models are estimated on the pooled out-of-fold
predictions of one repeat (each subject appears once) with a paired
bootstrap: subjects, or PSUs within strata, are resampled with replacement and
both models are scored on the same resample. Where a result also rests on the
25-fold analysis, the corrected resampled t-test (which inflates the variance
for the overlap between training sets) is used and reported as such, never an
uncorrected test. Model choices that used the folds (a capacity sweep, say) are
made on repeat 0 and confirmed on repeats not used to choose.

### 4.3 Pre-registered decision rule

A deep model is called useful on this cohort only if it improves IBS by at
least 0.001 reproducibly across seeds and folds, with a paired confidence
interval that excludes zero. Failure to meet the rule is reported as a
negative result and is not read as proof of equivalence.

### 4.4 The research agent

The experiments are carried out by a coding agent, the *loop*, that runs on a
locally served language model and is supervised by a stronger model. The
design goal is that a supervisor can audit and reproduce what the agent did,
and that results with help are never mixed with results without it.

**Task contract.** A run starts from a contract that is written before the
first model call and never changed: the task text, the repository and its
baseline revision, the acceptance checks, a list of protected paths, the
limits, the model reference, whether models reached over an API were allowed,
and the digest of the system prompt in effect. Acceptance checks are shell
commands chosen by the supervisor and kept outside the checkout the worker
can edit; protected paths (for example the repository's own tests) may not
change, and a candidate that touches one is rejected even if every check
passes.

**Isolation and tools.** The worker edits a detached git worktree of the
baseline revision, never the repository named by the task. It is given the
file, search and shell tools of the agent runtime it is built on, confined
to that checkout by the runtime's path jail; tools that reach the network,
delegate to another model, or switch the model are withheld. The checkout is
isolation from accidents and not a sandbox: the shell runs as the user.

**Limits.** One attempt is bounded by wall-clock time, generated tokens and
a cap on tool calls (the runtime has no run option for the last, so the loop
counts tool requests and cancels the run at the cap); the run is bounded by
total time, the number of attempts, bounded retries of a model call that
failed in transit, and, for a remote model, a cost cap. A remote call whose
price the provider does not report is counted and never treated as free.

**Judging and retry.** After an attempt the loop reads back from git what
changed (a patch, and for each file its status and content hashes before and
after), checks the protected paths, and runs the acceptance checks in the
candidate checkout. A failed attempt is classified (acceptance failed,
protected path touched, tool-call cap, time, token, provider, cancelled), and
the next attempt starts from a clean checkout and is shown the failure and
the tail of the failing output, never a patch.

**Trace.** Every run keeps an append-only event stream, written in the
agent runtime's own hash-chained log format (each line carries the hash of
the one before, so an edited, reordered, inserted or removed event is
reported when the stream is read, and writers in several processes extend one
chain). An event has a schema version, run, attempt, an id that only grows,
timestamp, parent, type and data, with credentials redacted before anything is
written and payloads over a size bound stored once by content address. The
chain detects accidental damage and edits that do not recompute every later
hash; it is not a defence against someone who can rewrite the whole file. The
runtime's own trajectory of each attempt is persisted beside it. A gap in what
could be observed is an event of its own. The run records which subagent,
skill, command and project-context definitions were in effect, found by the
runtime's own discovery rules, as a digest in the contract and a
`definitions` event, and flags a change on resume. Checkpoints are written
after every attempt; a resumed run does not trust the checkout an
interrupted attempt left, resets it to the baseline and records that it did.

**Stopping a worker that is going nowhere.** An attempt is stopped as
*stagnation* when one call (same tool, same arguments) has returned the same
answer a set number of times (default four); a repeated call whose answer
changes, such as a test run after an edit, is not counted. The retry is told
which call looped, never a patch. This guard exists because of what we
observed (section 5.1): a 27-billion-parameter model decoded greedily
repeated one probe command about forty times and wrote nothing.

**Accounting of help.** A run given any supervisor hint is recorded as
assisted. Unaided means no hint, no remote model and no supervisor repair
during the attempt. Remote models require an explicit opt-in flag.

## 5. Results

Numbers are means over the 25 cross-validation folds unless a repeat is named;
differences are paired on identical folds and carry the corrected resampled
t-test interval. Lower is better for the integrated Brier score (IBS, 0 to 15
years) and the Brier score at 10 years; higher is better for Uno's
concordance (C) at 10 years.

### 5.1 Bringing the agent up: what had to be repaired before a model could be judged

The first delegated task (add a Cox model that is nonlinear in each continuous
input) was attempted on a locally served 27-billion-parameter model (Q8_0
weights, served by the model engine on this machine, no remote call). The
first attempts did not measure the model; they exposed defects of the
infrastructure, each repaired, covered by a specification and rerun:

| Finding | What was observed | Repair |
|---|---|---|
| F-012 | Two attempts ended at the output-token cap with no file written: the model reasoned before every reply (5 to 13 thousand output tokens a call, empty visible text) | a model served on this machine is asked for no reasoning block unless `--thinking` is given |
| F-014, F-015 | With the reasoning block off, the model repeated one probe command about forty times and wrote nothing; the attempt ended at the tool-call cap. At a sampling temperature of 0.7 the same task edited files within five minutes: greedy decoding was the cause | a stagnation stop (one call returning the same answer four times) with feedback naming the call; `--temperature` and `--thinking` are recorded flags |
| F-016 | A follow-up round at a conversation of 58 thousand tokens was refused by the server (prompt plus requested reply above its context) | the provider is built with the context window the server reports, so replies are bounded by the room left |
| F-013 | The new hash-chained event stream could not read runs recorded before it | legacy streams are read and migrated on resume |
| F-017 | My own acceptance script chained two README checks with `&&` under `set -e`, so the requirement was not checked and an accepted patch lacked it | the check was corrected; the README item was carried to a later task |

With these in place the first task was accepted on its second attempt, with no
hint and no remote model. The first attempt wrote a model that refit its spline
knots on whatever rows it was handed, including the held-out rows; the
supervisor's hidden check (predictions for one held-out row must not change
when the other held-out rows are replaced by extreme values) failed with the
difference in the predictions, and the second attempt fitted the knots on the
training rows only. A follow-up task, accepted on its first attempt, added two
specification tests, one of which fails if the spline model is replaced by the
linear one (checked by mutation). A third task, a learning-curve fitting tool,
was not delivered: two attempts ended at the tool-call limit and the time
limit, and the third was cancelled by the supervisor for budget. This is
recorded as a failure of the agent on a task of that size, not as a
result about learning curves; the learning-curve fits reported below were made
by the supervisor with a script kept outside the repository and are labelled
as supervisor analysis.

### 5.2 The nonlinear additive ceiling

| model | IBS 0-15 y | Brier 10 y | C 10 y | calibration slope 10 y |
|---|---|---|---|---|
| age and sex only | 0.05311 | 0.06437 | 0.8554 | 1.04 |
| conventional risk factors (additive) | 0.04952 | 0.05946 | 0.8803 | 1.09 |
| elastic-net Cox, all inputs | 0.04613 | 0.05350 | 0.8986 | 1.08 |
| spline Cox, all inputs (agent-authored) | 0.04600 | 0.05346 | 0.9007 | 1.08 |
| inverse-weighted logistic, all inputs | 0.04605 | 0.05351 | 0.8999 | 1.02 |
| additive piecewise-exponential model, all inputs | 0.04649 | 0.05448 | 0.9000 | 1.09 |
| deep set encoder (`horizon`), all inputs | 0.04711 | 0.05488 | 0.8969 | 0.98 |

Allowing the Cox model to bend in each continuous input did not
resolvably change the integrated Brier score when all inputs are used
(spline minus linear, -0.00013, corrected 95% interval [-0.00057, +0.00031]),
although it raised concordance at 10 years by +0.0021 [+0.0007, +0.0035]. With
only the conventional risk factors the same change helps: -0.00062 [-0.00106,
-0.00019]. The nonlinear additive model is the strongest model in the table by
point estimate, and the deep set encoder is now resolvably worse than it:
+0.00111 [+0.00019, +0.00204] in IBS (p = 0.02) and +0.00142 [+0.00057,
+0.00228] in the Brier score at 10 years. Under the pre-registered rule
(section 4.3) the deep encoder is therefore not useful on this cohort: it does
not improve on the nonlinear additive baseline, it is slightly worse.

### 5.3 Learning curves (supervisor analysis, one repeat)

Mean IBS on the five folds of repeat 0 when a share of each fold's training
subjects is used (the test folds are identical):

| share of training subjects | elastic-net Cox | spline Cox | additive PE model | deep set encoder |
|---|---|---|---|---|
| 10% | 0.04791 | 0.04804 | 0.05318 | 0.05267 |
| 25% | 0.04692 | 0.04713 | 0.04939 | 0.04990 |
| 50% | 0.04630 | 0.04662 | 0.04743 | not run |
| 75% | 0.04617 | 0.04600 | not run | not run |
| 100% | 0.04615 | 0.04603 | 0.04657 | 0.04673 |

The two Cox models are nearly flat from half the data on: doubling the
sample from 50% to 100% lowers the error by 0.00015 (elastic-net Cox) and
0.0006 (spline Cox); from 75% to 100% by 0.00002 and -0.00003. The piecewise-exponential
additive model and the deep encoder are still falling (0.0009 between 50% and
100% for the additive model; 0.0032 between 25% and 100% for the encoder), so
the gap between the learners narrows with more data and the Cox curves
cannot be improved by data alone. Power-law fits `e_inf + a N^-b` to the
fold means are ill-determined for the flat curves (bootstrap interval of the
exponent from the lower bound to 1.5) and moderately determined for the
additive model (exponent 0.67, interval [0.50, 0.87]; asymptote 0.0447,
interval [0.0424, 0.0466]); an extrapolation from four points of one repeat is
a hypothesis for the next data set, not an estimate of what more subjects
would give.

### 5.4 Learning from accepted runs, and a candidate that is rejected

The loop keeps a training record only from a run that was accepted, unaided
and on a local model, and only from the accepted attempt (a rejected patch is
not a target). Of 22 runs offered, 9 produced a record (the rest were
rejected, errored or cancelled; the manifest names the reason for each). Seven
of the nine are the same task, so the set is small and not diverse.

The first training attempt did not run: the trainer refused the whole-
conversation records because the chat template of the student (Qwen3) renders
an assistant turn differently when it is the last message, so no loss mask can
be fixed for a multi-step tool conversation (F-021; the trainer states this
limitation and fails loudly). Trimming stray whitespace from tool-only turns
(F-020) was necessary and not sufficient. A primitive for exact boundaries in
such templates is missing in the model engine and is recorded as open there.
The loop therefore writes, on request, one exchange per accepted run: the task
as it was given and the accepted patch as a single assistant turn.

On those 10 records (8 trained on, 2 held out) the student, an 8-billion-
parameter model run in-process on the GPU, was fine-tuned with a rank-16
low-rank adapter for 30 optimiser steps. The training loss fell from 1.41 to
0.04, a sign of memorising eight records; the held-out loss fell from 1.216
to 1.193. The adapter was written (a file of 175 MB, with its digest) and
registered as a *candidate*, not as the model in use. It was then loaded by
eight fresh loop processes (the run's `model_selected` event names the
adapter) and run, with the model in use, over the same eight seeded tasks of
families absent from the training data, one attempt each, under identical
limits. The rule had been fixed in code before any result: at least eight
paired tasks, the candidate must solve more of them, and a one-sided sign test
over the tasks only one of the two solved must give p at most 0.10. The model
in use solved 1 of 8; the candidate solved 0 of 8 (one task lost, none gained,
p = 1.0). The candidate was rejected with that record, the model in use stayed
`local:Qwen/Qwen3-8B`, and asking to roll back reported that there was no
adapter in use to roll back from.

This proves the mechanics: a real forward and backward pass, an optimiser
step, a saved and reloaded adapter, an evaluation on held-out tasks under a
declared rule, a rejection that leaves the previous version in use. It does
not show that the loop learns: the student's failures at baseline were
malformed `edit_file` diffs repeated until the stagnation stop, and an adapter
trained to write patches in prose is not a remedy for that. The baseline is
one success in eight, so a gain would have needed to be large to be visible.

## 6. Discussion

**What the survival results license.** On this cohort, with the information a
single examination contains, a regularised Cox model with a spline in every
continuous input is the best model we measured by point estimate, and it is not
resolvably better than the same model without splines when all inputs are
used. The deep set encoder is resolvably worse than the spline model by
0.0011 in the integrated Brier score. Failing to find a gain is not proof that
none exists: our power to resolve a true difference of 0.001 is limited
(section 3 and the audit in Appendix B), and the residual-network experiment,
the one that would test "is there structure the additive model misses",
was not completed (section 5.1). What can be said is narrower and still
useful: three differently built learners (a nonlinear Cox model, a logistic
model with inverse-probability-of-censoring weights, and a piecewise-
exponential additive model) land within 0.0005 of one another, and the deep
encoder, whose curve is still falling with more data, has not caught up with
them at this sample size. Whether it would with several times more subjects
is a question for data that are not on this machine.

**What the learning curves license.** The Cox curves are flat from half of the
training subjects onward, so more of the same data will not improve them;
the learners with more flexible hazards are still improving. Both statements
come from one cross-validation repeat and from fold means of a metric whose
folds are positively correlated; the exponents are poorly determined, and the
extrapolated asymptote of the additive model is a hypothesis.

**What the agent results license.** Of four substantive tasks given to the
loop (a nonlinear Cox model, its specification tests, a learning-curve fitting
tool, a residual network), two were accepted (the second after infrastructure
repairs and one retry) and two were not delivered. The accepted work was
checked by the supervisor against hidden checks written before the run and
then read against the task; reading found two gaps (a missing specification
test, and a documentation item that my own check had failed to test). The
failures were informative: the 27-billion-parameter model decoded greedily
loops on one action; with reasoning enabled it spends its output budget before
writing; with a small server context a long conversation is refused; and a
larger multi-part task (a neural network with a fitted likelihood and a
learned multiplier) exceeded what it completed in three attempts. Each
infrastructure defect was repaired in the layer that owned it, with a
regression specification, and a defect of the supervisor's own checks was
found by the same process (F-017, F-019). We report the delivered fraction (2 of 4 tasks; 2 of the 10 attempts
that reached a verdict in the runs after the repairs were accepted, the other 8
were rejected by a check or ran out of a limit) as a property of this model on
these tasks, not as a general rate.

**What the learning-and-promotion results license.** Only that the path
works and can say no: a candidate adapter was trained, reloaded and judged on
held-out tasks under a rule fixed beforehand, and was rejected without
disturbing the model in use. It licenses nothing about whether the loop's
learning improves the agent; that needs more accepted runs, a trainer that
takes multi-step tool conversations, and an evaluation set with a baseline far
from zero. The autonomy stage reached is *supervised local*: the loop works on
local models with the supervisor's diagnostic repairs recorded; no unaided
local evaluation on unseen tasks was run.

## 7. Limitations

* One examination per person: the data cannot answer whether repeated
  measurements add information, which is the regime where sequence
  representations have been reported to help. The accelerometer minute data
  for two cycles (2003 to 2006) have been downloaded but not used; they would
  make a long-sequence experiment possible and are the next step.
* One country and era; mortality is the only outcome observed after the
  examination; follow-up is administratively censored, so the 15-year metric
  rests on the earliest cycles alone.
* Analysis-time reuse of the cross-validation folds: capacity and recipe
  choices for the deep encoder were made on repeat 0 and confirmed on others,
  but baselines, the spline model and the recipe sweep share the same partition;
  all configurations tried are reported and none was selected on a locked test.
* The pooled out-of-fold paired bootstrap that the protocol (section 4.2)
  prefers to the corrected resampled t-test was specified as a loop task and
  not run; every interval above is the corrected resampled t-test interval.
* The agent was evaluated on four tasks and on one local model; the learning
  experiment trained on nine accepted runs, seven of which are the same task.
  Nothing here shows general autonomy.
* The residual-network experiment and the calibration-curve additions
  (integrated calibration index, E50 and E90) were specified and not delivered.
  The audit of the calibration code found no defect in the censoring-aware
  quantities but did find that no summary of the curve was reported.

## 8. Reproducibility

Code revisions: brain f5cde671, sven d4ddb0c, splinter at the commit that
contains this file (`git log -1 -- papers/agent-driven-residual-survival/paper.md`).
Data: the lifecourse build described in `samples/lifecourse/README.md`
(timelines, partition and frozen criteria are content-addressed in
`FROZEN.json`). Commands, in the order they were run:

```sh
# local model served by brain (CUDA); one process on the device at a time
BRAIN_BACKEND=cuda BRAIN_QWEN35_GGUF_CTX=98304 brain serve --openai 127.0.0.1:8788 --models-dir <models>
# a delegated task (sampling temperature recorded in the run's model_selected event)
BRAIN_API_KEY=<key printed by serve> agent-loop --temperature 0.7 run --workspace <clone> \
  --task-file <task.txt> --accept "behaviour=bash <accept.sh>" --accept-visible "harness-tests=..." \
  --protect <paths> --model "remote:brain/unsloth/Qwen3.8-27B-Q8_0" --allow-api-models \
  --max-output-tokens 40000 --max-attempts 3 --follow-ups 3 --json
# the spline baseline on the 25 folds, scored like every arm
python -I samples/lifecourse/baselines/run.py --data <data> --out <data>/baselines --baseline spline-cox-net-all --jobs 16
splinter-lifecourse external --data <data> --baseline spline-cox-net-all
splinter-lifecourse compare --data <data> --a external:spline-cox-net-all --b external:cox-net-all
# learning curves: shares of the training subjects, scored on the same folds
splinter-lifecourse recipe subsample --data <data> --share 0.25
python -I samples/lifecourse/baselines/run.py ... --train-ids <data>/recipe/_subsamples/p25-s1 --repeat 0
splinter-lifecourse recipe score --data <data> --name spline-cox-net-all-p25 --made-of "..."
# learning from accepted runs, and judging the candidate
agent-loop dataset --run <accepted run> ... --out train.jsonl
agent-loop train --dataset train.jsonl --base local:Qwen/Qwen3-8B --steps 60 --rank 16
samples/agent/loop/fixtures/eval-candidate.sh <out> <model-ref> anagrams:1 bisect:1 ...
agent-loop models judge --candidate <version> --baseline-results <dir> --candidate-results <dir>
```

Delegated tasks (task text and hidden acceptance scripts are kept outside the
repository so that a worker's checkout cannot contain them; their SHA-256
prefixes are given):

| task | run | verdict | task / check digest |
|---|---|---|---|
| spline Cox model | run-20261007T203058.568-370c | accepted, attempt 2 of 3 | 76994035 / 54a7583d |
| its specification tests | run-20261007T205231.010-e99c | accepted, attempt 1 (README item unmet, F-017) | 89fad691 / aac259b7 |
| learning-curve tool | run-20261007T205818.780-7b26 | not delivered (limits; cancelled at attempt 3) | 2af4b8c9 / 9eb66011 |
| residual network | run-20261007T215753.920-5a61 | rejected after 3 attempts | 4082373b / 4fe6cc75 |
| residual network, continued, assisted by a hint | run-20261007T225456.244-8b2c | rejected after 2 attempts | 4082373b / 4fe6cc75 |

Earlier delegated attempts of the first task ended before a model could be
judged because of infrastructure defects (section 5.1): runs 6059, a64e (killed
by the supervisor after its first two attempts looped), aa65, 9e1c and c44b.

## Appendix A. Research log

| Date | Milestone | Outcome |
|------|-----------|---------|
| 2026-10-07 | M0 protocol and plan | Paper created; decision rule fixed (section 4.3) |
| 2026-10-07 | M1 loop built | Section 4.4 describes the loop as implemented; specs pass with a scripted model; real-model result pending |
| 2026-10-07 | Reuse of the agent runtime | Event stream moved onto the runtime's hash-chained log; definitions in effect recorded through its discovery; two small additions to the runtime's SDK (`chain`, `workspace`), neither naming this project |
| 2026-10-08 | Learning from accepted runs | Trainer refused multi-step tool conversations (F-021); patch-form records trained an adapter; candidate rejected on eight held-out tasks (section 5.4) |
| 2026-10-07 | First delegated experiment task (spline Cox) | Four infrastructure defects found and repaired before the model could be judged: reasoning block spent the output budget (F-012), identical-probe loops under greedy decoding (F-014, F-015), server context overflow on long conversations (F-016), and the chain could not read old runs (F-013); see 5.1 |

## Appendix B. Audit of an external literature review

An external review of the literature and of our earlier results was supplied
during this work. Its statements are treated as hypotheses: each is checked
against the code, the data or the arithmetic before it changes anything. This
appendix records the outcome. "Checked" means we read the code or recomputed
the number ourselves; "not checked" means we only have the review's word.

| Statement of the review | Status | What we found |
|---|---|---|
| Power: a true difference of 0.001 in IBS needs roughly 2.0 to 2.7 times our present effective information, given the observed paired standard error | Checked (arithmetic) | Reproduced: a two-sided p of 0.05 to 0.09 for a difference of 0.001 gives a standard error of 0.00051 to 0.00059; 80% power at alpha 0.05 needs 0.000357. The ratio of variances is 2.0 to 2.7. It is an approximation, since the p-values come from a corrected resampled test, not from a bootstrap. |
| Our p-values treat the 25 folds as independent | Checked, refuted | The existing paired comparison uses the corrected resampled t-test (variance inflated for the overlap between training sets). The review's concern is valid for an uncorrected test and does not describe this analysis. A subject and PSU bootstrap on pooled out-of-fold predictions remains the stronger check and is reported in Results. |
| Calibration code of the evaluation library (checked by reading it, 2026-10-07) | Checked | `calibration::at_horizon` takes the observed risk of the whole sample and of each equal-weight risk group from an Aalen-Johansen estimate, so a subject censored before the horizon is never a non-event. Its intercept and slope are a logistic regression on the subjects whose outcome by the horizon is known, weighted by the sampling weight over the censoring survival `G` at the event time (or at the horizon for a subject still event-free); a subject censored earlier drops out and the weights stand in for it. `G` is the marginal Kaplan-Meier curve of the training data supplied by the caller, so the correction is valid when censoring does not depend on the predictors; here censoring is administrative and depends on the survey cycle, which is why every horizon metric is restricted to the cycles whose survivors were all followed that long. `auc::at` is the cumulative/dynamic time-dependent AUC with a competing event excluded from the controls, and says so in its documentation. The cause-specific observed risk is Aalen-Johansen, never one minus Kaplan-Meier. **Gap found:** the library reported no summary of the calibration curve itself (integrated calibration index, E50, E90), which is what should be judged for a step-function recalibration where the slope is not interpretable; added through the loop (section 5.4). |
| Our expected calibration error may count subjects censored before the horizon as negatives | Checked, does not apply | In the implementation the observed risk of each equal-weight risk group is an Aalen-Johansen estimate, so a subject censored before the horizon stays in the risk set until censoring and is never a negative. Slope and intercept use inverse-probability-of-censoring weights and drop subjects censored before the horizon. The censoring distribution is the marginal Kaplan-Meier of the training data; censoring here is administrative (the end of the mortality linkage) and depends on the survey cycle, which is why every metric at a horizon is restricted to the cycles whose survivors were all followed that long. |
| A recalibration slope after isotonic or Venn-Abers calibration does not mean what it means for a smooth score | Not checked in code; accepted as a caution | Where we report calibration after isotonic-type recalibration we report the curve, the intercept and the horizon Brier score and do not rely on the slope. |
| Deep survival models beat linear models at the scale of hundreds of thousands to millions of patients and billions of events; no published threshold exists | Partly checked | The scale figures were checked against the primary sources on 2026-10-07: MOTOR pretrains a 143M-parameter model on up to 55M patient records and 9B clinical events and reports a 4.6% improvement in time-dependent C-statistic over the state of the art across 19 tasks and up to 95% better label efficiency; SurvivEHR is trained on over 7.6B coded events from 23M patients; Delphi-2M trained on 402,799 UK Biobank participants and found about 2 million parameters optimal for that dataset. That no sample-size threshold exists was not independently established; we treat it as an open question and measure our own learning curve. |
| NHANES mortality discrimination near 0.90 is close to a ceiling for one examination | Not checked | Consistent with our own measured values. Not independently verified in the literature by us. |
| Gated delta-net mixers have no evidence of benefit on three to twenty sparse visits | Not checked | Our own measurement found no resolvable difference to attention on synthetic data. The line of work is frozen. |
| Licence and access terms of the listed cohorts | Not checked | Taken as a checklist for the data owner to confirm in writing. Nothing was downloaded or requested on this basis. |
