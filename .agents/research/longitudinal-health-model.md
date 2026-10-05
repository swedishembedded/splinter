# longitudinal health model

**Status: plan, nothing implemented.** The design, the data findings and the
phase order for training a continuous-time longitudinal health model with
explicit time-to-event prediction: the model is a new brain architecture
(`horizon`, planned in brain's own roadmap), Splinter trains, measures and
releases it, and sven runs the agentic parts (dataset intake). Every
mechanism this needs in Splinter is generic: it trains a model of this kind on
any timeline dataset, health or not. Written 2026-10-05 from the data on disk,
the three codebases as they are, and a literature survey (papers saved under
the shared resources tree, `papers/health-fm/`, 44 PDFs).

## 1. What we are building, stated honestly

Input: a well-defined record of one person (measurements with values and
times, diagnoses, medication, lifestyle including diet timing). Output, for any
horizon up to the longest follow-up the data support:

- a cumulative incidence curve per outcome (all-cause and cause-specific death
  first, disease onset where the data carry it), with an interval;
- a predictive distribution of each biomarker at a future time, conditional on
  being alive then;
- a latent state that updates when a new visit arrives, without retraining.

"Exact measurements 10-20 years ahead" is not an attainable output and the
plan does not promise it: the target is calibrated distributions whose width
grows honestly with the horizon. Evidence that this is the right target: the
best published model of this family keeps a mean AUC of about 0.76 internally,
about 0.70 at 10 years and 0.67 on an external cohort, and its 20-year
generated trajectories barely beat an age-sex baseline.

## 2. What the data on disk actually contain

Measured on the downloaded files (`resources/health/`):

| NHANES cycle | Participants | Eligible for mortality linkage (adults) | Deaths to 2019 | Day-1 diet recall | Day-2 recall | Median follow-up (months) |
|---|---|---|---|---|---|---|
| 1999-2000 | 9,965 | 5,445 | 1,675 | 8,843 | - | 234 |
| 2001-2002 | 11,039 | 5,987 | 1,624 | 9,882 | - | 212 |
| 2003-2004 | 10,122 | 5,610 | 1,420 | 9,033 | 8,352 | 189 |
| 2005-2006 | 10,348 | 5,561 | 1,027 | 9,349 | 8,429 | 165 |
| 2007-2008 | 10,149 | 6,219 | 1,126 | 9,254 | 7,837 | 141 |
| 2009-2010 | 10,537 | 6,510 | 861 | 9,754 | 8,405 | 118 |
| 2011-2012 | 9,756 | 5,849 | 628 | 8,519 | 7,605 | 94 |
| 2013-2014 | 10,175 | 6,100 | 467 | 8,661 | 7,573 | 71 |
| 2015-2016 | 9,971 | 5,974 | 276 | 8,505 | 7,027 | 47 |
| 2017-2018 | 9,254 | 5,809 | 145 | 7,640 | 6,639 | 24 |
| **Total** | **101,316** | **59,064** | **9,249** | **89,440** | **61,867** | |

Consequences that shape the design:

1. **Prospective outcomes are death only.** The public linkage gives vital
   status, a leading-cause recode (about ten groups) and diabetes and
   hypertension flags from multiple causes of death. Incident disease after
   the exam is not observed. Disease onset exists only retrospectively (2).
2. **NHANES is retrospectively longitudinal.** The questionnaires record ages
   at events before the exam. Counts for 2009-2010 alone: age told of
   hypertension 2,212; diabetes 739; arthritis 1,674; coronary disease 173;
   angina 250; heart attack 261; stroke 227; age started smoking 2,866; weight
   ten years ago 4,518; weight at age 25 5,407; heaviest weight 6,505 with its
   age 6,397. A single exam therefore becomes a life-course timeline of onset
   ages and weight history. Two caveats are designed in: recall error, and
   **survivor sampling** (only people alive at the exam are in it), which the
   likelihood handles by conditioning on survival to entry (brain roadmap,
   `horizon` section 4).
3. **Long horizons are thin.** Only 1999-2004 participants reach about 15-20
   years. A 20-year metric is a 1999-2002 metric; a temporal split cannot test
   long horizons at all. This is reported, not hidden.
4. **Diet timing is real.** `DRD020` (1999-2002) and `DR1_020`/`DR2_020`
   (2003-2018) give the clock time of every eating occasion. Rather than
   hand-derived "fasting windows" only, each recall day enters as its own
   sequence of eating events (time of day, energy, macronutrients), so the
   model can learn timing features nobody named; the hand-derived window,
   overnight fast and late-eating share are kept as interpretable probes and
   baselines. Two recall days give within-person variability.
5. **Dense sensors exist for some cycles.** 2011-2014 accelerometry hourly and
   daily summaries (`PAXHR_*`, `PAXDAY_*`) are downloaded; minute-level and
   the 2003-2006 raw files are separate archives not yet fetched.
6. **Small randomised trials** (Bath, Queen Mary 5:2 and TRE trials, TIMET
   table data, the WashU trial's omics) carry repeated measures over weeks to
   a year and an assigned intervention. They anchor the interventional part;
   they cannot anchor 10-year mortality.
7. **Restricted cohorts** (CARDIA, WHI, UK Biobank, Whitehall II, the Harvard
   cohorts, DPP, Look AHEAD) need applications. Each comes with its own terms,
   several forbidding commercial use, so every record and derived artifact
   carries its source's terms from the first commit of the data path (section
   6, S5).

Facts to confirm against NCHS documentation before they are relied on: the
rule for pooling 1999-2002 four-year and later two-year exam weights, and the
disclosure perturbation NCHS applies to the public-use mortality file.

## 3. The model (brain `horizon`)

The full specification is brain's roadmap. In one paragraph: each exam is a
SET of variable tokens (value fused with its variable by FiLM over soft bins;
a value below a detection limit is a state of the token, a variable not
measured contributes no token) summarised into a visit token; visits, events
and interventions go through an interchangeable temporal backbone - a Gated
DeltaNet recurrence whose decay is scaled by physical elapsed time and relaxes
towards an age-conditioned population state between visits, or attention with
rotary angles from real-valued time - and the state at a query time feeds
cause-specific piecewise-constant hazards (closed-form cumulative incidence
with death competing, left truncation and right censoring exact) and a
heteroscedastic, detection-limit-aware measurement head. Two clocks enter
every hazard: attained age (the timescale) and calendar time (period effects:
mortality fell between 1999 and 2019 and a model without calendar time
mis-calibrates whichever era it was not fitted on).

What is new relative to the literature (section 7): direct-horizon
competing-risk incidence with exact delayed entry in a foundation-model
objective; retrospective onset ages scored under survivor conditioning; a
mean-reverting physical-time state for extrapolation; detection-limit
likelihoods; survey-design weights inside the loss and the evaluation.

## 4. Training plan

1. **Baseline first.** `horizon` with no hidden layers is a proportional
   hazards model on the age timescale; it and an age-sex-calendar-only model
   are fitted before anything deep. Nothing ships that does not beat them on
   the locked metrics.
2. **Pretraining on NHANES timelines**: event loss on prospective death
   (all-cause and cause groups) and retrospective onsets, masked-value
   reconstruction within each exam, weighted by the survey weights normalised
   per source.
3. **Longitudinal sources** (the trials now, restricted cohorts later): the
   measurement loss at observed future visits and the latent transition loss.
4. **Uncertainty**: seeded ensembles, conformalised incidence, an
   out-of-distribution score (latent density per source, count of absent
   variables) that triggers abstention.
5. **Interventional module, separate from prediction** (section 5).

Model size follows the data, not ambition: about 60,000 linked adults and
9,249 deaths support a model of a few million parameters at most (the
published UK Biobank model has 2.2M). A model that small should make seeds
and ensembles affordable; the training cost per seed is measured in phase 2,
not assumed.

## 5. Prediction is not intervention

The foundation model estimates `P(future | history)`. It does not estimate
`P(future | do(fast 16 h))`: people change diet BECAUSE of disease, so an
observational model can learn the arrow backwards. The interventional module
is built later and separately:

- questions are written as emulated target trials (eligibility, time zero,
  strategy, outcome), which rules out immortal-time definitions such as
  "adherent at year 2";
- the observational model's predicted effect on short-term biomarker change
  is corrected on the trials with a low-dimensional bias term fitted to the
  randomised contrast (the Kallus-Puli-Shalit shape), because 36-300-person
  trials can estimate a correction, not a model;
- trial analyses use the model as a prognostic covariate (PROCOVA / H-AIPW),
  which keeps type-I error and only gains precision;
- any mortality-scale "what if" is labelled as extrapolation through a
  surrogate, because no available trial measures it.

## 6. Who does what

**brain** (generic model): the `horizon` crate, its kernels and gradient
checks, survival evaluation arithmetic (concordance, IPCW Brier and its
integral, D-calibration, calibration slope and intercept, all weighted), the
SDK pipeline and the serving capability. Brain's roadmap lists the phases.

**Splinter** (generic campaign machinery, then one sample):

- **S1 Record sources.** A source parser for tabular record files (SAS
  transport `.xpt`, CSV) and their codebooks in `splinter-knowledge`, with the
  digests and provenance every source has today.
- **S2 Timeline datasets.** A `timeline-v1` dataset format and manifest in
  `splinter-data`; a split that keeps a group whole, stratifies (by cycle,
  source, outcome), and freezes a locked test set before any training through
  the existing frozen ledger.
- **S3 A trainer by capability.** The model adapter advertises the objectives
  it can train (the open item in phase 8); `timeline` becomes one, backed by
  brain's `horizon` SDK. Outputs are immutable artifacts with manifests, as
  adapters are today.
- **S4 A metric gate.** The release gate today is pass/fail per item with a
  sign test. A continuous-metric gate adds: per-subject (or per-cluster)
  paired bootstrap of a metric difference (`bootstrap_interval` exists),
  improvement on one declared metric, non-inferiority bounds on the others
  (calibration slope, D-calibration, per-subgroup), and "unmeasured is a
  failure" as today.
- **S5 Terms travel with data.** Every source records its licence or data-use
  terms; lineage carries them to datasets and releases; a release refuses a
  combination its sources' terms forbid.
- **S6 Intake agent.** Harmonising hundreds of variables across ten NHANES
  cycles (renamed variables, changed units and assays) and later cohorts is
  agent work with deterministic admission: the generator role reads a
  codebook page through a sven typed call and proposes a mapping (variable to
  concept, unit, conversion, kind, detection limits) with the codebook text it
  relied on; code admits it only if the quoted text is in the codebook, the
  converted distribution overlaps the concept's other sources, and the ranges
  are physical. Rejected and admitted proposals are experiences like any other.
- **Sample `samples/lifecourse`**: the health-specific part - which NHANES
  files and variables, the concept list, the timeline builder (prospective
  mortality and retrospective onsets), the frozen benchmark, and its README
  stating what was measured.

**sven**: nothing new is required for S1-S6. The intake agent is a typed call
on the existing SDK (`Method<T>`, a role, a deadline, postconditions). A sven
change is made only if running S6 exposes a real gap (codebook pages larger
than a context is the likely one), and then as a generic capability.

## 7. Evaluation protocol, fixed before training

- **Locked test**: 15% of linked adults, stratified by cycle and death, never
  used for any decision, frozen by digest. A development split for model
  selection; leave-one-cohort-out once a second cohort exists.
- **Metrics at 5, 10 and 15 years** (20 where the 1999-2002 cycles allow):
  integrated and horizon IPCW Brier score as the primary, Uno and Antolini
  concordance, D-calibration, calibration slope and intercept against
  Aalen-Johansen estimates; survey-weighted with design-based intervals
  (bootstrap over strata and PSUs); by sex, age band, race and ethnicity, and
  cycle.
- **Biomarker forecasts**: CRPS and interval coverage on the trials' follow-up
  visits and on masked values.
- **Promotion**: an update is promoted only when the primary metric improves
  with a paired interval above zero and no non-inferiority bound fails.
  Concordance alone never promotes (it is not a proper scoring rule).

## 8. Phase order across the three repositories

1. Splinter S1 + S2 and the `lifecourse` timeline builder; brain `horizon`
   phase 1 (contract, hazard head, kernels). Acceptance: the builder reproduces
   the counts in section 2; the brain NLL matches a reference survival library
   on a fixture; gradcheck green on both backends.
2. Baselines trained and evaluated through S3 and S4 on the frozen split.
3. brain phases 2-4 (set encoder, heads, backbones, evaluation arithmetic);
   the first deep model against the baselines.
4. S6 intake agent widens the variable set; diet-timing and accelerometer
   encoders; the trials as longitudinal sources.
5. Uncertainty, serving, and the interventional module.
6. Restricted cohorts as access is granted, each under S5.

## 9. Risks and open questions

- Self-reported onset ages are noisy and survivor-sampled; if the retrospective
  term hurts the prospective metrics on the development split it is dropped,
  and that result is recorded.
- Calendar time is confounded with follow-up length in NHANES (early cycles
  have the long follow-up); its effect is estimated on short horizons where
  every cycle contributes.
- The survey weights make the population estimate right and the training
  signal noisier; both weighted and unweighted fits are evaluated with the
  weighted metrics.
- Small trials will not move a deep model; they correct it (section 5).
