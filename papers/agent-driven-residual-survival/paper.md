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

*(Pending. To be written once milestone results exist. The paper will report
(i) whether a deep survival model adds held-out predictive information over a
spline/additive Cox baseline on a single-examination cohort, with uncertainty
that respects the cross-validation design; (ii) how much of the experimental
work a locally served language-model agent can carry out unaided; and (iii)
what each finding does and does not license.)*

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

*(Pending. To cover: time-to-event foundation models trained on large
longitudinal record sets; neural versus Cox survival models on benchmark
data; numerical feature embeddings for tabular deep learning; sample-size
requirements for survival-model calibration and validation; inverse
probability of censoring weighting for evaluation; NHANES mortality
prediction and biological-age studies; coding agents with execution-based
validation.)*

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

**Trace.** Every run keeps an append-only JSONL event stream (schema version,
run, attempt, an id that only grows, timestamp, parent, type, data), with
credentials redacted before anything is written and payloads over a size
bound stored once by content address. The agent runtime's own trajectory of
each attempt is persisted beside it. A gap in what could be observed is an
event of its own. Checkpoints are written after every attempt; a resumed run
does not trust the checkout an interrupted attempt left, resets it to the
baseline and records that it did.

**Accounting of help.** A run given any supervisor hint is recorded as
assisted. Unaided means no hint, no remote model and no supervisor repair
during the attempt. Remote models require an explicit opt-in flag.

## 5. Results

*(Pending.)*

## 6. Discussion

*(Pending.)*

## 7. Limitations

*(Pending. To include: single examination per person; one country and era;
mortality-only outcomes; analysis-time design reuse; agent evaluated on a
small number of tasks.)*

## 8. Reproducibility

*(Pending: exact commands, run identifiers, artifact locations, code
revisions of the three repositories.)*

## Appendix A. Research log

| Date | Milestone | Outcome |
|------|-----------|---------|
| 2026-10-07 | M0 protocol and plan | Paper created; decision rule fixed (section 4.3) |
| 2026-10-07 | M1 loop built | Section 4.4 describes the loop as implemented; specs pass with a scripted model; real-model result pending |

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
| Our expected calibration error may count subjects censored before the horizon as negatives | Checked, does not apply | In the implementation the observed risk of each equal-weight risk group is an Aalen-Johansen estimate, so a subject censored before the horizon stays in the risk set until censoring and is never a negative. Slope and intercept use inverse-probability-of-censoring weights and drop subjects censored before the horizon. The censoring distribution is the marginal Kaplan-Meier of the training data; censoring here is administrative (the end of the mortality linkage) and depends on the survey cycle, which is why every metric at a horizon is restricted to the cycles whose survivors were all followed that long. |
| A recalibration slope after isotonic or Venn-Abers calibration does not mean what it means for a smooth score | Not checked in code; accepted as a caution | Where we report calibration after isotonic-type recalibration we report the curve, the intercept and the horizon Brier score and do not rely on the slope. |
| Deep survival models beat linear models at the scale of hundreds of thousands to millions of patients and billions of events; no published threshold exists | Not checked | We did not verify the cited studies. We use the statement only as a reason to expect our data to be limiting, and test the expectation with learning curves. |
| NHANES mortality discrimination near 0.90 is close to a ceiling for one examination | Not checked | Consistent with our own measured values. Not independently verified in the literature by us. |
| Gated delta-net mixers have no evidence of benefit on three to twenty sparse visits | Not checked | Our own measurement found no resolvable difference to attention on synthetic data. The line of work is frozen. |
| Licence and access terms of the listed cohorts | Not checked | Taken as a checklist for the data owner to confirm in writing. Nothing was downloaded or requested on this basis. |
