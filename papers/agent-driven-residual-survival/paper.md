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

*(Pending: cohort, follow-up, outcomes, survey design variables, exclusions.)*

## 4. Methods

### 4.1 Models compared

*(Pending: elastic-net Cox; spline/additive Cox; boosted survival baseline;
neural piecewise-exponential model; additive-plus-residual model with a
residual multiplier initialised at zero.)*

### 4.2 Evaluation protocol

*(Pending: one set of paired out-of-fold predictions; subject and PSU
bootstrap of paired differences in integrated Brier score (IBS), concordance
and time-dependent AUC; the entire model-selection procedure repeated inside
the resampling loop where selection used the same folds; design-weighted
versus sample-level estimates reported separately.)*

### 4.3 Pre-registered decision rule

A deep model is called useful on this cohort only if it improves IBS by at
least 0.001 reproducibly across seeds and folds, with a paired confidence
interval that excludes zero. Failure to meet the rule is reported as a
negative result and is not read as proof of equivalence.

### 4.4 The research agent

*(Pending: architecture of the loop, tool set, local and remote model routing,
budgets, trace format, supervision accounting.)*

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

## Appendix B. Audit of an external literature review

*(Pending: a claim-by-claim record of which statements from an external review
we verified against primary sources, which we reproduced numerically, and
which remain unverified.)*
