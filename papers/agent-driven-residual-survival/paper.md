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

*Working draft, started 2026-10-07; last revised 2026-10-08 after the secondary
estimands (sections 4.5, 4.6 and 5.5 to 5.10) were added to the first session of
delegated experiments. Every numerical claim in this document is either
reproduced by a command listed in the Reproducibility section or is explicitly
marked as unmeasured. Section 2 (related work) and the residual-network part of
the question are incomplete and say so.*

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
a residual neural network, was not delivered by the agent in five attempts
(two runs) and is reported as not done. The same single examination was then
asked five further questions under rules fixed beforehand: it predicts death
by cause (the full input set beats the conventional risk factors for
cardiovascular, cancer and other deaths), death with a diabetes mention on the
certificate beyond who dies (and a hypertension mention not at all), expected
time lived (calibrated within 0.05 years at ten years), eight prevalent
conditions (AUROC 0.79 to 0.92 with the full input set, 0.67 to 0.85 without
the inputs that define each), and undiagnosed disease only weakly (sensitivity
0.2 to 0.45 at 90% specificity; about nine of ten positive calls for diabetes
are false); accelerometer summaries added a small, narrowly resolved gain to the
ten-year Brier score. On the agent question, two of four substantive tasks
were accepted and two were not. Before the first could even be judged, four
defects of the surrounding infrastructure (a reasoning block spending the
output budget, greedy decoding repeating one action, a server context
overflow, and an unreadable old log format) had to be found and repaired; two
defects of the supervisor's own acceptance checks surfaced later. Each repair
has a regression specification. A candidate adapter trained on ten records from accepted runs
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

Two ways of estimating the difference between two models were planned. The
stronger is on the pooled out-of-fold predictions of one repeat (each subject
appears once) with a paired bootstrap: subjects, or PSUs within strata, are
resampled with replacement and both models are scored on the same resample.
The other is the corrected resampled t-test over the 25 folds (which inflates
the variance for the overlap between training sets), never an uncorrected
test. For the primary comparisons **only the corrected t-test was run**: the
bootstrap was specified as a task for the agent and was not run for lack of
time (section 7). The secondary estimands of sections 5.6, 5.7 and 5.9 use a
cluster bootstrap (cycle, stratum, PSU) on pooled out-of-fold predictions of
one repeat. Model choices that used the folds (a capacity sweep, say) are
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
during the attempt. Remote models require an explicit opt-in flag; a model
served by the model engine on this machine counts as local. Settings the
supervisor chooses (sampling temperature, output and tool-call budgets) are
configuration, written in the run, and do not make a run assisted. In this
paper the two accepted tasks were unaided in that sense (no hint); the
continuation of the residual-network task was assisted (it was given a hint
about a library call) and is labelled so. The infrastructure repairs of
section 5.1 were made by the supervisor between runs, not during an attempt.

### 4.5 Secondary estimands and their pre-registered rules

Added 2026-10-08, before any number for these estimands was computed. They use
the same cohort, folds and corrected resampled t-test as the primary question.
All five are secondary: none changes the rule of section 4.3.

| Id | Estimand | Rule fixed in advance |
|---|---|---|
| T1 | Cumulative incidence of death by cardiovascular, cancer and other causes at 5, 10 and 15 years | A model beats the conventional risk-factor model for a cause iff the 10-year cause-specific Brier difference has a corrected 95% interval below zero. Calibrated iff the bootstrap observed-over-expected interval contains 1 and the slope lies in [0.8, 1.25] |
| T2 | Cumulative incidence of death with diabetes or hypertension listed anywhere on the certificate (the linked file's multiple-cause flags), with death without the flag competing | Models are ranked only with at least 100 flagged events at the horizon; below that, descriptive only. A flag model adds something only if it beats the all-cause risk used as a ranker, on time-dependent AUC, with an interval excluding zero |
| T3 | Restricted mean survival time to 10 years (15 on the 1999-2002 cycles), and the age at which a reference life table gives the same time (mortality-equivalent age) | Calibrated iff the mean absolute gap over risk groups is at most 0.25 years at 10 years and the slope lies in [0.9, 1.1] |
| T4 | Prevalent conditions at the examination: diabetes, hypertension, kidney markers, anaemia, high cholesterol, osteoporosis, depression, sleep problem | Useful iff AUROC exceeds the age-sex model by at least 0.02 with an interval excluding zero. Two input sets are reported side by side: all inputs except those that define the label, and non-definitional inputs only (demographics, body measures, recalled weights, smoking, alcohol, eating times) |
| T5 | Undiagnosed disease among those not reporting the diagnosis | A screening value is claimed only if sensitivity at 90% specificity beats a model of age and body-mass index with a bootstrap interval excluding zero. Inputs are the non-definitional set only |

What each estimand does not claim. A certificate mention is not a diagnosis
and is under-reported; records without multiple-cause data are excluded and
counted; there are few events, so intervals are wide and no subgroup claim is
made. Mortality-equivalent age is a re-expression of one model against one
reference table, not a measure of biological ageing, and its association with
death is partly by construction; only the cause-, flag- and condition-specific
associations are informative. A prevalent condition at the same examination is
a classification, not a forecast, and the labels derived from a measurement
are definitional for the inputs that contain it. "Undiagnosed" is one
measurement above a threshold plus recall of no diagnosis, not a confirmed
diagnosis. Eight conditions and three causes are examined; all are reported
and none is claimed beyond its rule. Label definitions are in Appendix C.

### 4.6 Accelerometer summaries as an input block (pre-registered 2026-10-08)

Written before the accelerometer files were opened. The 2003-2004 and 2005-2006
cycles carried a hip-worn uniaxial accelerometer for seven days, recorded per
minute. Daily summaries are derived by a fixed recipe: non-wear is a run of at
least 60 minutes of zero counts allowing at most two minutes of counts up to
100 inside it; a day is valid with at least 600 minutes of wear; a subject is
included with at least four valid days. Per subject, over valid days: mean counts
per wear minute, minutes per day at or above 2,020 counts (moderate to vigorous
activity), minutes per day below 100 counts (sedentary), minutes per day in
between (light), the highest 30-minute mean count, the probability that a
sedentary minute is followed by an active one (a fragmentation measure), the
number of valid days and the wear time.

The question is whether these summaries add to the same model without them. It
is asked on the subjects of the two cycles with a valid recording, with the
folds of the partition restricted to them: the spline Cox model on all
inputs, against the same model with the summaries appended, trained and scored
on the restricted folds. The block is called useful only if the 10-year Brier
score falls, with the corrected resampled t-test interval of the paired
difference below zero; Uno's concordance at 10 years is reported beside it.
The restricted subsample is small (a few thousand adults and a few hundred
deaths by 10 years), so a failure to meet the rule is a statement about this
sample, not about the signal. A model that reads the minute sequences directly
(rather than summaries) is not compared.

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
| 50% | 0.04630 | 0.04662 | 0.04743 | 0.04792 |
| 75% | 0.04617 | 0.04600 | 0.04687 | 0.04778 |
| 100% | 0.04615 | 0.04603 | 0.04657 | 0.04673 |

The two Cox models are nearly flat from half the data on: doubling the
sample from 50% to 100% lowers the error by 0.00015 (elastic-net Cox) and
0.0006 (spline Cox); from 75% to 100% by 0.00002 and -0.00003. The piecewise-exponential
additive model and the deep encoder are still falling (0.0009 between 50% and
100% for the additive model; 0.0032 between 25% and 100% for the encoder), so
the gap between the learners narrows with more data and the Cox curves
cannot be improved by data alone. Power-law fits `e_inf + a N^-b` to the
fold means are ill-determined for the flat Cox curves (bootstrap interval of
the exponent from the lower bound to about 1.5) and for the deep encoder
(asymptote interval from 0 to 0.0475), and moderately determined for the
additive model (exponent 0.67, interval [0.50, 0.88]; asymptote 0.0447,
interval [0.0423, 0.0466]); an extrapolation from five points of one repeat is
a hypothesis for the next data set, not an estimate of what more subjects
would give. The curve of the additive model is the only one that supports a
statement about data: it is still falling and its fitted asymptote lies below
the full-data error of every Cox model, which a larger sample of this kind
would test.

### 5.4 Learning from accepted runs, and a candidate that is rejected

The loop keeps a training record only from a run that was accepted, unaided
and on a local model, and only from the accepted attempt (a rejected patch is
not a target). Of 22 runs offered, 9 produced a whole-conversation record and 10 a
patch-form record (the rest were rejected, errored or cancelled; the manifest
names the reason for each). Seven of them are the same task, so the set is
small and not diverse.

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

### 5.5 Secondary estimand T1: death by cause

Rules fixed in section 4.5 before the numbers existed. The models are three
cause-specific Cox models fitted per cause on the same inputs as the all-cause
baselines: age and sex alone, the conventional risk factors (`standard`), and
every input (`all`). Each is scored on the 25 folds, by cause, at 5, 10 and 15
years, with the other causes competing. A horizon is scored only on the cycles
whose follow-up reaches it, so the 15-year rows rest on the earliest cycles.
The table gives the 10-year row; all rows, with the 5- and 15-year horizons,
are in the run output (`m6/t1-causes.md`).

| Cause (events per fold) | Model | Brier | Index of prediction accuracy | AUC | O/E | Slope | Brier difference to `standard` |
|---|---|---|---|---|---|---|---|
| Cardiovascular (213) | age, sex | 0.02685 | 0.124 | 0.906 | 0.963 | 1.06 | +0.00090 [+0.00043, +0.00137] |
| | `standard` | 0.02594 | 0.153 | 0.926 | 0.995 | 1.09 | reference |
| | `all` | 0.02522 | 0.177 | 0.938 | 0.984 | 1.08 | -0.00072 [-0.00113, -0.00032] |
| Cancer (147) | age, sex | 0.02268 | 0.043 | 0.862 | 0.948 | 1.10 | +0.00030 [+0.00021, +0.00039] |
| | `standard` | 0.02238 | 0.056 | 0.880 | 0.946 | 1.15 | reference |
| | `all` | 0.02190 | 0.076 | 0.894 | 0.961 | 1.10 | -0.00048 [-0.00069, -0.00027] |
| Other (293) | age, sex | 0.03886 | 0.120 | 0.857 | 0.922 | 1.04 | +0.00087 [+0.00049, +0.00124] |
| | `standard` | 0.03799 | 0.140 | 0.879 | 0.954 | 1.05 | reference |
| | `all` | 0.03618 | 0.181 | 0.911 | 0.942 | 1.07 | -0.00181 [-0.00249, -0.00113] |

Brackets are the corrected resampled t-test interval over the 25 folds. AUC is
the time-dependent AUC at 10 years with event-free subjects as controls; O/E
is observed over expected; the index of prediction accuracy is one minus the
Brier score over that of a constant prediction.

By the rule of section 4.5, `all` beats `standard` for each of the three
causes at 10 years: the interval of the Brier difference lies below zero in
every case. The gain is largest for the heterogeneous residual group (-0.0018)
and about half as large for cardiovascular (-0.0007) and cancer (-0.0005)
death. At 15 years the cardiovascular interval touches zero
([-0.00135, +0.00007]) and the cancer AUC interval spans it, with 82 to 124
events per fold, so the rule is met for the other combinations and not for
those two.

What the rule did not get. The calibration part of the rule asks for a
bootstrap interval of observed over expected; it was not run (it needs the
survey-design bootstrap, which is not implemented for causes), so calibration
is described, not tested. Observed over expected lies between 0.92 and 1.0 for
every model and cause at 10 years: the models predict slightly more deaths than
occur, by up to 8% for the residual group, and the recalibration slopes of 1.04
to 1.15 say the predictions are, if anything, not spread out enough. These are
means over folds; no cause shows a slope outside [0.8, 1.25].

Reading the numbers. The high AUCs are largely age: age and sex alone reach 0.86
to 0.91, and the other inputs add 0.03 to 0.05. The index of prediction
accuracy shows how much is left in absolute terms: cancer death is the least
predictable cause (index 0.076 with all inputs against 0.177 for
cardiovascular death), as expected, since few of the inputs bear on tumours.
The cause groups are those of the public file's underlying-cause recode, which
NCHS perturbs for some records.

### 5.6 Secondary estimand T2: death with diabetes or hypertension listed

The linked file marks deaths on whose certificate diabetes, or hypertension,
appears anywhere. The outcome is death with the mention by year *t*, competing
with death without it. In the cohort 971 deaths carry a diabetes mention and
1,325 a hypertension mention; one death has no multiple-cause data and is
censored at its time. A mention is not a diagnosis, and it is under-reported.

Flagged deaths are few per fold, so predictions are not scored fold by fold:
the out-of-fold predictions of one repeat are pooled (each of about 47,800 subjects
appears once), scored as one set, and a 95% interval comes from resampling
clusters (cycle, stratum, PSU) 1,000 times, on repeat 0. The pooled set has
237 (diabetes) and 304 (hypertension) flagged deaths by 5 years and 384 and 488
by 10 years, above the count of 100 the rule of section 4.5 requires before
models are ranked. Two flag models are compared with the control that the
rule names, the all-cause incidence of the full-input Cox model used as a
ranker: a cause-specific Cox model for death with the mention, and the
all-cause incidence times a penalised logistic estimate of the share of deaths
that carry the mention (assuming the share does not depend on when death comes).
Both use every input.

| Outcome, horizon | All-cause ranker, AUC | Flag Cox, AUC minus ranker | Share model, AUC minus ranker |
|---|---|---|---|
| Diabetes listed, 5 y | 0.948 [0.929, 0.964] | +0.0231 [+0.0118, +0.0368] | +0.0246 [+0.0150, +0.0360] |
| Diabetes listed, 10 y | 0.943 [0.930, 0.955] | +0.0220 [+0.0129, +0.0325] | +0.0253 [+0.0175, +0.0339] |
| Hypertension listed, 5 y | 0.926 [0.904, 0.945] | -0.0008 [-0.0079, +0.0055] | +0.0012 [-0.0033, +0.0062] |
| Hypertension listed, 10 y | 0.935 [0.919, 0.949] | -0.0018 [-0.0069, +0.0037] | +0.0014 [-0.0023, +0.0052] |

By the rule, the diabetes models add something to knowing who dies: both
beat the ranker at both horizons with intervals above zero. With the
conventional inputs only, the Cox model's gain at 10 years does not exclude
zero (+0.0116 [-0.0003, +0.0238]). The hypertension models add nothing the
all-cause ranker does not already have: the differences are within about 0.002
of zero with intervals that span it. Among deaths with multiple-cause data
(7,102 in the pooled set), the implied share carrying the mention separates
those with a diabetes mention from those without with an AUROC of 0.83 (Cox)
to 0.84 (share model), and those with a hypertension mention with 0.61 for
both. Calibration of the diabetes models is close to exact (observed over
expected 0.95 to 1.01); for hypertension the Cox model is at 0.94 to 0.95 and
the share model over-predicts by about 15% (0.83 to 0.86). The index of
prediction accuracy is modest (0.095 and 0.14 for diabetes at 5 and 10 years;
0.043 and 0.065 for hypertension): the high AUCs come largely from age.

Results on repeats 1 to 4 (point estimates only; the repeats pool the same
subjects, so they show stability across training splits, not independent
confirmation) are within 0.003 in AUC of repeat 0 for every model and horizon.
We do not know why the hypertension mention is not predictable from the
examination beyond the risk of dying; one possible reading, which we did not
test, is that it is listed on certificates for reasons tied to the death more
than to the person's blood pressure.

### 5.7 Secondary estimand T3: expected time lived and mortality-equivalent age

A predicted curve gives each subject an expected time lived in the next *τ*
years, the restricted mean survival time (RMST). Calibration compares the mean
prediction in ten equal-weight risk groups with the Kaplan-Meier RMST observed
there, on the pooled out-of-fold predictions of repeat 0, on the cycles whose
follow-up reaches *τ* (23,097 subjects and 3,266 deaths at 10 years; 8,976 and
1,905 at 15 years, which only the 1999 to 2002 cycles reach).

| Model | Mean absolute gap, 10 y (years) | Slope, 10 y | Uno C of the RMST, 10 y | Mean absolute gap, 15 y | Slope, 15 y |
|---|---|---|---|---|---|
| age, sex (Cox) | 0.045 | 0.95 | 0.855 | 0.094 | 0.96 |
| `standard` (cause-specific Cox) | 0.036 | 1.01 | 0.876 | 0.087 | 1.01 |
| `all` (cause-specific Cox) | 0.035 | 1.02 | 0.898 | 0.058 | 1.01 |
| spline Cox, `all` | 0.025 | 1.01 | 0.900 | 0.089 | 1.00 |

By the rule of section 4.5 (a gap of at most 0.25 years and a slope in
[0.9, 1.1]) every model is calibrated at both horizons, by a wide margin on
the gap. The mean predicted and observed 10-year RMST are 9.51 and 9.53 years
for the best models. Repeats 1 and 2 give the same figures within 0.005 years
of gap. This is not a strong test of the models: the 10-year RMST is within 0.1
years of its ceiling for five of ten risk groups, where nothing can be wrong
by much, and the discriminating information sits in the first two groups
(predicted 6.9 and 9.1 years for the full-input model, observed 6.9 and 9.2).

*Mortality-equivalent age.* For each subject, the age at which a Gompertz
life table gives the same expected time over 10 years. The table is fitted per
sex to the training subjects of the subject's fold by weighted maximum
likelihood with delayed entry, so nothing about a test subject touches it.
Ages are clamped to 18 to 85 and the clamps counted: 6% of subjects (1,356 of
23,097 for the full-input Cox model) fall outside the table's range. They sit
at both ends: 5% to 7% of subjects under 30 are healthier than any age in the
range can express, so their equivalent age is the lower bound, and 41% of
those aged 80 and over (10% of the seventies) have an expectation below that of
an 85-year-old, so theirs is the upper bound. For these subjects the
equivalent age is a bound and not a measurement, and the acceleration of the
oldest in particular is censored from above; the hazard ratios below mix
measured and bounded values and should be read with that in mind.

The association of the acceleration (equivalent age minus age) with outcomes
over the whole follow-up, by Cox models adjusted for age and sex, with survey
weights and variance clustered on PSUs:

| Outcome (events) | Full-input Cox | Spline Cox | Age and sex only (control) |
|---|---|---|---|
| All-cause death (5,220) | 1.72 [1.68, 1.77] | 1.67 [1.63, 1.72] | 0.84 [0.61, 1.15] |
| Cardiovascular (1,664) | 1.76 [1.68, 1.84] | 1.73 [1.65, 1.81] | 0.55 [0.31, 0.97] |
| Cancer (1,114) | 1.54 [1.46, 1.62] | 1.50 [1.43, 1.58] | 1.71 [1.11, 2.63] |
| Other (2,442) | 1.80 [1.73, 1.87] | 1.74 [1.68, 1.80] | 0.75 [0.47, 1.20] |
| Diabetes mention (617) | 2.04 [1.89, 2.19] | 1.98 [1.85, 2.13] | 0.38 [0.17, 0.86] |
| Hypertension mention (842) | 1.66 [1.56, 1.77] | 1.63 [1.53, 1.73] | 0.34 [0.15, 0.77] |

Hazard ratios per 5 years of acceleration. They are large and, as the
rule of reading fixed in advance says, partly by construction: the
acceleration is a function of a model fitted to predict death, evaluated on
the same outcome. The control shows what is and is not construction. The same
quantity from a model that sees only age and sex is a function of age and sex,
which the Cox model adjusts for, and has no association with death (0.84,
interval spanning one); the intervals of its other rows are wide, since little
of it remains after adjustment, and are not read. What is informative is the
contrast across outcomes for the full-input models: the acceleration tracks
diabetes-mentioned deaths most closely (about 2.0) and cancer deaths least
(about 1.5), consistent with the cause-specific accuracy of section 5.5.
Nothing here shows that the acceleration measures biological ageing, or that
changing it would change a risk.

### 5.8 Secondary estimand T4: prevalent conditions at the examination

Eight conditions and one variant (diabetes defined with fasting glucose, on the
fasting subsample and its weights) are labelled from the questionnaire and
laboratory files (Appendix C). These are classifications at the same visit, not
forecasts: a label defined by a measurement is not predicted from that
measurement. Models therefore see one of two input sets, reported side by
side: `nondef` (demographics, body measures, recalled weights, smoking, alcohol
and eating times; the same for every label) and `full` (every input except
those that define the label or a drug that treats it). The reference is a
logistic model of age and sex. Models are logistic regression with a penalty
chosen by cross-validation (`logit`) and gradient-boosted trees (`hgb`), fitted
unweighted on the training subjects whose label is known and scored on the
test folds with the survey weights, over the same 25 folds. A guard specifies
that no excluded column reaches a model's design matrix; writing it found that
the display names of a column subset were mislabelled, a defect of the
printing only (the matrix was right), repaired.

| Label (weighted prevalence) | Age, sex | `nondef`, trees | gain over age, sex | `full`, trees | gain over age, sex |
|---|---|---|---|---|---|
| Diabetes (0.114) | 0.742 | 0.843 | +0.102 [+0.094, +0.110] | 0.895 | +0.153 [+0.146, +0.160] |
| Diabetes, fasting subsample (0.124) | 0.745 | 0.834 | +0.089 [+0.076, +0.102] | 0.879 | +0.134 [+0.125, +0.144] |
| Hypertension (0.371) | 0.782 | 0.820 | +0.037 [+0.034, +0.041] | 0.840 | +0.058 [+0.054, +0.062] |
| Kidney markers (0.151) | 0.749 | 0.787 | +0.039 [+0.032, +0.045] | 0.845 | +0.096 [+0.086, +0.107] |
| Anaemia (0.063) | 0.661 | 0.791 | +0.130 [+0.118, +0.143] | 0.919 | +0.259 [+0.245, +0.273] |
| High cholesterol (0.398) | 0.731 | 0.751 | +0.020 [+0.016, +0.023] | 0.790 | +0.059 [+0.053, +0.065] |
| Osteoporosis, age 50 and over (0.060) | 0.848 | 0.852 | +0.004 [-0.002, +0.011] | 0.870 | +0.022 [+0.014, +0.030] |
| Depression, PHQ-9 of 10 or more (0.076) | 0.563 | 0.733 | +0.170 [+0.147, +0.194] | 0.819 | +0.256 [+0.234, +0.278] |
| Sleep trouble told to a doctor (0.270) | 0.603 | 0.672 | +0.068 [+0.060, +0.076] | 0.787 | +0.184 [+0.173, +0.194] |

AUROC, survey-weighted, mean over 25 folds; the gain carries the corrected
resampled t-test interval. By the rule of section 4.5 (a gain of at least 0.02
with an interval excluding zero) the `full` models are useful for all nine
labels, and the `nondef` models for seven: not osteoporosis (gain 0.004), and
not high cholesterol, whose gain of 0.0197 misses the threshold by 0.0003. The
logistic models give almost the same figures as the trees (full: within 0.01
for most labels; the trees add 0.011 to 0.015 for anaemia, kidney markers and
high cholesterol). Calibration slopes lie between 0.94 and 1.21. The conventional risk-factor logistic model, restricted
to what does not define the label, sits between age-sex and `nondef` for most
labels and above `nondef` for kidney markers and high cholesterol.

How to read it. The `full` figures include correlates measured on the same
blood draw as the label: the red-cell and white-cell indices for anaemia, the
liver enzymes and uric acid for diabetes. They show what the examination
contains, not what a person's history predicts. The `nondef` figures are the
more interesting: from age, body size, weight history, smoking, alcohol, income
and eating times, diabetes is ranked with an AUROC of 0.84, depression with
0.73 and anaemia with 0.79. Labels are complete-case: a subject with an
unknown label is dropped, not counted negative, and the weights are the
examination weights, which are not exact for labels from the laboratory
subsamples (urine albumin in particular). Bone density was not used for
osteoporosis, whose label is the recalled diagnosis in those aged 50 and over,
because its reference values could not be verified against a source.

### 5.9 Secondary estimand T5: undiagnosed disease

Among subjects who do not report the diagnosis (and, for diabetes and
hypertension, are not treated), the label is the measured criterion: HbA1c of
6.5% or more, blood pressure of 140/90 or more, an eGFR below 60 or an
albumin-to-creatinine ratio of 30 mg/g or more, total cholesterol of 240 mg/dL
or more. It is one measurement, not a confirmed diagnosis, and "undiagnosed"
rests on recall. A screen uses the `nondef` inputs only, since a screening tool
cannot use the laboratory value it replaces, and a call is a score at or above
the threshold that gives 90% specificity among the training subjects, estimated
on cross-fitted predictions so that a flexible model's training fit does not
flatter it. The comparator is a model of age and body-mass index. The rule of
section 4.5 claims value only if sensitivity at that threshold beats the
comparator with a cluster-bootstrap interval excluding zero. Folds of repeat 0
are pooled; predictive values are restated at the weighted prevalence of the
screened population.

| Undiagnosed (screened, cases, prevalence) | Screen | Sensitivity | Specificity | Positive predictive value | Minus age, BMI at the training threshold [95% CI] | Minus age, BMI at matched 90% specificity [95% CI] |
|---|---|---|---|---|---|---|
| Diabetes (39,837; 1,127; 2.0%) | age, BMI | 0.332 | 0.915 | 0.073 | comparator | comparator |
| | `nondef` logistic | 0.305 | 0.943 | 0.099 | -0.027 [-0.064, +0.013] | +0.072 [+0.031, +0.114] |
| | `nondef` trees | 0.330 | 0.933 | 0.091 | -0.002 [-0.047, +0.039] | +0.048 [+0.000, +0.092] |
| Hypertension (30,467; 3,325; 9.4%) | age, BMI | 0.337 | 0.922 | 0.308 | comparator | comparator |
| | `nondef` logistic | 0.326 | 0.938 | 0.352 | -0.011 [-0.029, +0.007] | +0.035 [+0.011, +0.053] |
| | `nondef` trees | 0.308 | 0.937 | 0.335 | -0.028 [-0.046, -0.012] | +0.018 [-0.004, +0.037] |
| Kidney markers (32,408; 5,188; 12.4%) | age, BMI | 0.369 | 0.928 | 0.422 | comparator | comparator |
| | `nondef` logistic | 0.350 | 0.940 | 0.454 | -0.019 [-0.030, -0.008] | +0.021 [+0.008, +0.033] |
| | `nondef` trees | 0.359 | 0.940 | 0.458 | -0.010 [-0.021, +0.001] | +0.022 [+0.007, +0.035] |
| High cholesterol (27,695; 2,901; 9.8%) | age, BMI | 0.077 | 0.937 | 0.116 | comparator | comparator |
| | `nondef` logistic | 0.171 | 0.914 | 0.176 | +0.094 [+0.074, +0.116] | +0.081 [+0.056, +0.105] |
| | `nondef` trees | 0.223 | 0.902 | 0.198 | +0.146 [+0.123, +0.170] | +0.106 [+0.076, +0.130] |

By the rule as written, only high cholesterol is claimed: the `nondef` models
beat age and body-mass index at the training threshold with intervals above
zero, and no screen beats the comparator for diabetes, hypertension or kidney
markers. That reading is confounded: the training-derived thresholds gave test
specificities of 0.90 to 0.94, not 0.90 (they were chosen on unweighted
cross-fitted training predictions and applied to weighted test data), and a
model that is more specific is less sensitive by construction. The final column
re-reads every screen at exactly 90% specificity on the pooled set (a threshold
no one could set in advance; **post hoc, not pre-registered**). There the
`nondef` models are ahead of age and body-mass index for all four conditions by
2 to 11 points of sensitivity (the trees for hypertension do not exclude zero).
Both readings agree on the size of the claim: sensitivity at 90% specificity is
0.41 to 0.45 for diabetes, hypertension and kidney markers and about 0.2 for cholesterol,
and at the real prevalence of 2% for diabetes about nine of ten positive calls
are false. For cholesterol, sex alone accounts for a part of the gain over age
and body-mass index (age-sex gains 0.035 at the training threshold). We claim no
clinical screening value for any of these.

### 5.10 Accelerometer summaries as an input block

The rule is in section 4.6. Of the 12,422 examinees of the 2003-2004 and
2005-2006 cycles with at least four valid days of recording, 6,952 are in the
cohort (adults eligible for linkage); 5,877 of them are outside the locked
test, and 806 died within ten years. The summaries have plausible levels: the
median adult wears the monitor 14 hours a day, averages 285 counts per minute
and spends 15 minutes a day at or above 2,020 counts. An early version of the
extraction called every day of every subject valid with 24 hours of wear,
because the transport file stores a zero count as 5.4e-79 and no zero minute
was ever seen; the specification for the extraction now includes that value.

| Metric (10 years unless stated) | Without the block | With the block | Paired difference [95% CI, corrected] |
|---|---|---|---|
| Brier score | 0.05186 | 0.05097 | -0.00089 [-0.00167, -0.00011] |
| Uno concordance | 0.9028 | 0.9034 | +0.0006 [-0.0016, +0.0028] |
| Uno concordance, 5 years | 0.9100 | 0.9123 | +0.0022 [-0.0017, +0.0061] |
| Calibration slope | 1.112 | 1.104 | -0.008 [-0.044, +0.028] |

By the rule of section 4.6 the block is useful: the interval of the Brier
difference lies below zero. The evidence is thin and the effect small. The
upper end of the interval is within 0.0001 of zero, the gain is under 2% of the
Brier score, and discrimination does not resolvably change. The comparison is
on 5,877 subjects, so both arms are trained on about 4,700 and are worse
estimated than the models of the other sections (the integrated Brier score
over 15 years cannot be computed: no cycle here is followed that long). One
reading, which we did not test, is that what a hip accelerometer says about the next ten
years is largely already in the examination, in age, body size, blood tests
and self-reported health, with a small remainder. A sequence model on the
minute counts, which the roadmap asked for, was not tried.

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

**What the secondary estimands license.** One examination predicts more than
whether a person dies. For death by cause, adding the full set of inputs to the
conventional risk factors improves the cause-specific Brier score for
cardiovascular, cancer and other deaths (section 5.5), most for the residual
group. For death with a diabetes mention on the certificate it adds to knowing
who dies (section 5.6), and for a hypertension mention it adds nothing. Expected
time lived is well calibrated by every model (section 5.7), and the
mortality-equivalent age built on it tracks diabetes-mentioned and
cardiovascular deaths more closely than cancer deaths, with a control that
separates what is construction from what is information. Conditions at the
examination are ranked well from the questionnaire and measurements alone
(section 5.8), and the screens for undiagnosed disease built on non-laboratory
inputs are weak: a sensitivity of 0.2 to 0.45 at 90% specificity, a positive
predictive value of 9% for diabetes at its real prevalence (section 5.9). None
of these is a forecast of incident disease: the data have no follow-up for
it. What the examination shows is that one visit contains a calibrated
mortality forecast by cause, an estimate of expected time lived, and
classifications of conditions, some of whose labels and inputs come from the
same blood draw, and that is the extent of it.

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
  for two cycles (2003 to 2006) were reduced to eight daily summaries
  (section 5.10); a model of the minute sequences themselves, which would be a
  long-sequence experiment, was not tried.
* One country and era; mortality is the only outcome observed after the
  examination; follow-up is administratively censored, so the 15-year metric
  rests on the earliest cycles alone.
* Analysis-time reuse of the cross-validation folds: capacity and recipe
  choices for the deep encoder were made on repeat 0 and confirmed on others,
  but baselines, the spline model and the recipe sweep share the same partition;
  all configurations tried are reported and none was selected on a locked test.
* The pooled out-of-fold paired bootstrap that the protocol (section 4.2)
  prefers to the corrected resampled t-test was specified as a loop task and
  not run for the primary comparisons: every interval of sections 5.2 to 5.5,
  5.7 (calibration), 5.8 and 5.10 is a corrected resampled t-test interval.
  Sections 5.6, 5.7 (associations) and 5.9 use a cluster bootstrap on pooled
  out-of-fold predictions of one repeat, which the other repeats do not
  confirm independently, since they pool the same subjects.
* The secondary estimands (sections 5.5 to 5.10) were written down before their
  numbers existed, but they are many: nine cause-horizon cells, two flags, eight
  conditions and four screens are reported, each against its own rule, and none
  is adjusted for the number of looks. A rule met by a narrow margin (the
  accelerometer block, the cholesterol gain of 0.0197) should be read as such.
* The calibration clause of the cause rule (a bootstrap interval of observed
  over expected) was not run; the matched-specificity reading of the screens is
  post hoc; weights for labels from laboratory subsamples other than fasting
  glucose are the examination weights, which is not exact; the bone-density
  definition of osteoporosis was not used; creatinine is standardised by the
  correction each cycle's laboratory documentation states, and estimated
  filtration rate by an equation quoted from the literature, not verified against
  a second implementation.
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
# secondary estimands: the multiple-cause flags beside the timelines, then death by cause (T1)
splinter-lifecourse build --nhanes <nhanes> --mortality <mortality> --data <new dir>   # timelines.jsonl identical to the frozen one
python -I samples/lifecourse/baselines/run.py --data <data> --out <data>/baselines --baseline cs-cox-agesex --jobs 16
splinter-lifecourse causes --data <data> --model cs-cox-agesex --model cs-cox-standard --model cs-cox-all --reference cs-cox-standard --score
# T2: death with a diabetes or hypertension mention
python -I samples/lifecourse/baselines/flags.py --data <data> --out <data>/baselines --model flag-cox-all --model flag-share-all --jobs 16
splinter-lifecourse flags --data <data> --ranker cs-cox-all --model flag-cox-all --model flag-share-all --repeat 0
# T3: expected time lived, mortality-equivalent age, and its association with outcomes
splinter-lifecourse labels --nhanes <nhanes> --data <data>
python -I samples/lifecourse/baselines/conditions.py --data <data> --out <data>/conditions --jobs 16
splinter-lifecourse prevalence --data <data> --model agesex --model conv-logit --model nondef-logit --model full-logit --model nondef-hgb --model full-hgb
splinter-lifecourse screen --data <data> --model agebmi --model agesex --model nondef-logit --model nondef-hgb --comparator agebmi --repeat 0
splinter-lifecourse lifeexp --data <data> --model cs-cox-all --model spline-cox-net-all --repeat 0 --tau 10 --aa-dir <aa dir>
python -I samples/lifecourse/baselines/assoc.py --data <data> --aa <aa dir>/aa-cs-cox-all-r0-t10.jsonl
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
| 2026-10-08 | Secondary estimands pre-registered (section 4.5); T1 death by cause (section 5.5) | The full-input cause-specific Cox model beats the conventional risk-factor model for each of three causes at 10 years; calibration described, its bootstrap not run |
| 2026-10-08 | T2 death with a diabetes or hypertension mention (section 5.6) | Diabetes models beat the all-cause ranker (AUC +0.022 to +0.025, intervals above zero); hypertension models do not |
| 2026-10-08 | T3 expected time lived and mortality-equivalent age (section 5.7) | Every model calibrated by the section 4.5 rule; acceleration associates with cause and flag as described, and its age-sex control shows no association |
| 2026-10-08 | T4 prevalent conditions and T5 undiagnosed-disease screens (sections 5.8, 5.9) | Models beat age-sex for nine labels (full inputs) and seven (non-definitional); by the rule only cholesterol is a screen, at matched specificity all four are ahead of age and body-mass index (post hoc) |
| 2026-10-08 | Accelerometer summaries as an input block (sections 4.6, 5.10) | Rule fixed before the files were opened; the 10-year Brier score falls by 0.0009 with an interval just below zero, discrimination unchanged; an extraction defect (zero counts stored as 5.4e-79) found by checking the summaries' levels |
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

## Appendix C. Label definitions for the secondary estimands

Thresholds are those of the cited guidelines (ADA for glucose and HbA1c, JNC 7
and ACC/AHA for blood pressure, KDIGO for kidney markers, WHO for anaemia and
bone density). The codebooks of each cycle were used to verify variable names,
units and codes only. A refused or "don't know" answer gives a missing label,
never a negative; a condition a cycle did not measure is absent for that cycle.

| Label | Definition | Notes from the codebooks |
|---|---|---|
| diabetes | DIQ010 = 1, or DIQ050 = 1, or DIQ070 (DID070 in 2005) = 1, or HbA1c >= 6.5%, or fasting glucose >= 126 mg/dL with fasting >= 8 h | Glucose exists on the fasting subsample only and needs the fasting weights |
| hypertension | mean systolic >= 140 or diastolic >= 90, or BPQ050A = 1 | 130/80 as a sensitivity analysis |
| kidney markers | eGFR (CKD-EPI 2021, race-free) < 60, or urine albumin/creatinine >= 30 mg/g | Creatinine variable differs in 2001 (LBDSCR); one measurement, not the three months KDIGO requires |
| anaemia | haemoglobin < 13 g/dL (men), < 12 (women not pregnant) | |
| high cholesterol | total cholesterol >= 240 mg/dL, or BPQ100D = 1, or BPQ090D = 1 | BPQ080 is skipped when cholesterol was never checked |
| osteoporosis | OSQ060 = 1, or femoral-neck T-score <= -2.5 at age 50 or over | Not asked in 2011 and 2015; bone density only in some cycles |
| depression | PHQ-9 >= 10 | 2005 onward |
| sleep problem | SLQ050 = 1 | 2005 onward |
| undiagnosed (diabetes, hypertension, kidney markers, high cholesterol) | the measured criterion above, among those whose self-report of the diagnosis is not "yes" | "Never checked" counts as not diagnosed for cholesterol |

## Appendix D. Subjects per label

From `lifecourse labels` on the 56,253 subjects. Unknown is a refused or
unanswered question, a condition the cycle did not measure, or (for the
undiagnosed labels) a subject outside the screened population.

| Label | Positive | Negative | Unknown |
|---|---|---|---|
| diabetes | 8,049 | 45,568 | 2,636 |
| diabetes, fasting subsample | 3,826 | 20,109 | 32,318 |
| hypertension | 22,257 | 31,943 | 2,053 |
| hypertension, 130/80 | 28,441 | 25,816 | 1,996 |
| kidney markers | 8,413 | 34,839 | 13,001 |
| anaemia | 4,741 | 47,139 | 4,373 |
| high cholesterol | 20,620 | 29,180 | 6,453 |
| osteoporosis (age 50 and over) | 2,415 | 34,770 | 19,068 |
| depression | 3,164 | 33,030 | 20,059 |
| sleep trouble | 9,886 | 30,494 | 15,873 |
| undiagnosed diabetes | 1,335 | 45,568 | 9,350 |
| undiagnosed hypertension | 3,896 | 31,943 | 20,414 |
| undiagnosed kidney markers | 6,097 | 31,990 | 18,166 |
| undiagnosed high cholesterol | 3,421 | 29,180 | 23,652 |

Deaths with a diabetes or hypertension mention: 971 and 1,325 of 8,355; one
death has no multiple-cause data.
