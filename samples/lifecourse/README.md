<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright (c) 2026 Martin Schröder <info@swedishembedded.com> -->
<!--
Swedish Embedded AB implements long-horizon risk prediction from cohort and
survey data for its clients. If your team needs expertise in building,
validating and proving time-to-event models on real health records, you can
procure our services by sending an email to info@swedishembedded.com.
-->

# lifecourse - one examination, fifteen years ahead

Can a model read one health examination of a person - measurements,
laboratory values, answers, eating times, what they recall of their past -
and say, calibrated, how likely they are to die, and of what, within five to
fifteen years, better than the conventional risk factors can?

## Data

The ten continuous NHANES cycles (1999-2018) and the public-use linked
mortality file, followed to 31 December 2019. Every adult eligible for linkage
and examined is a subject: their timeline (`timeline-v1`) holds what was
measured or answered at the examination, the eating-time features of their
24-hour recalls, the ages at which conditions were first diagnosed and
recalled body weights as history, and their death in three competing cause
groups (cardiovascular, cancer, other) inside the follow-up window. The
variables, their renames across cycles, unit changes and refusal codes are
written down in `src/concepts.rs`.

What the data cannot do, stated once:

- Death is the only outcome observed after the examination; a diagnosis made
  later is not in the public data.
- Follow-up ends administratively, so a cycle examined in 2015 is followed for
  about four years. A metric at horizon `h` is computed on the cycles whose
  survivors were all followed `h` years: the 15-year metric rests on the
  1999-2002 cycles only.
- NCHS perturbs follow-up and cause of death for some records of the
  public-use mortality file.
- Ages are top-coded (85 before 2007, 80 after).
- Associations, not effects: nothing here says what changing a behaviour
  would do.

## Arms

| Arm | Model | Inputs |
|---|---|---|
| `age-sex` | additive proportional hazards | age and sex |
| `standard` | additive proportional hazards | the conventional risk factors at the examination |
| `additive` | additive proportional hazards | everything |
| `horizon` | brain's set encoder | everything |

All arms share the encoding, the competing-risk piecewise-exponential
likelihood, the training procedure (early stopping on a tenth of the training
subjects) and the evaluation.

## Conventional baselines

Secondary comparators, never one of the criteria and never scored on the
locked test. `baselines/run.py` fits each of these in Python
(scikit-survival, scikit-learn) on the training subjects of every fold of
every repeat of the partition and writes its out-of-fold predictions; the
`external` command scores them with the arms' own metrics, so that
`compare` can set any of them beside `horizon`, `additive`, `standard` and
`age-sex` on identical folds.

| Baseline | Model | Inputs |
|---|---|---|
| `km` | Kaplan-Meier survival and Aalen-Johansen incidence of the training subjects, the same for everyone | none |
| `cox-net-standard`, `cox-net-all` | elastic-net Cox (the ridge-like and the mixed penalty both tried), Breslow baseline | conventional risk factors; everything |
| `cs-cox-agesex`, `cs-cox-standard`, `cs-cox-all` | one such Cox model per cause, the others censoring, joined into cumulative incidence by the Aalen-Johansen formula | age and sex; conventional risk factors; everything |
| `spline-cox-net-standard`, `spline-cox-net-all` | the elastic-net Cox model with a cubic B-spline basis on every continuous input (knots fitted on the training subjects, constant beyond their range), the nonlinear additive ceiling | conventional risk factors; everything |
| `logit-ipcw-standard`, `logit-ipcw-all` | logistic regression for death by 5, 10 and 15 years, weighted by the inverse probability of remaining uncensored, joined by a monotone interpolation | conventional risk factors; everything |
| `logit-ipcw-yearly-all` | the same at every year from 1 to 15 | everything |
| `gbs-all`, `gbs-fast-all` | scikit-survival's gradient-boosted survival model (Cox loss), at two learning rates; the number of trees is chosen on the validation share | everything |

Inputs are those of the arms: `standard` is the conventional risk factors of
`src/concepts.rs`, `all` is every examination concept, the eating-time
features, the years since each recalled diagnosis and the recalled weights.
Missing values are imputed with the training subjects' median, skewed
non-negative measurements are log-transformed, columns are scaled, answers
are one-hot encoded and a missingness indicator is added, all from the
training subjects of the fold only. Every hyperparameter (penalty,
regularisation, number of trees) is chosen on a validation share held out of
the training subjects, and each fold's choice is recorded in its `.meta.json`
and in the baseline's `manifest.json` with the digests of the timelines and
the partition, the seed and the software versions. A horizon of the logistic
model is fitted on the training subjects of the cycles followed that long,
the rule the evaluation applies. Training is unweighted, as the arms'
is; the metrics are survey-weighted, as everywhere.

A prediction file is one JSON line per test subject, `subject_id`, `cif` (the
predicted probability of death from any cause by year 1 to 15) and optionally
`cause_cif` (the same for each cause). `external` refuses a file whose
subjects are not exactly the fold's, or in which a value is not a finite,
non-decreasing probability. Predictions are read as linear between years and
held after year 15. The scores are kept beside the predictions, not under
`runs/`.

```bash
python -I baselines/run.py --data <data> --out <data>/baselines --jobs 8   # all baselines, all folds
lifecourse external --data <data> --baseline gbs-all                      # score its prediction files
lifecourse compare  --data <data> --a horizon --b external:gbs-all
lifecourse compare  --data <data> --a external:cox-net-all --b additive
```

`report` adds a row per scored baseline to its cross-validation section.
`python -I baselines/test_baselines.py` holds the harness's own checks.

## Secondary estimands

Questions the same cohort can answer besides all-cause mortality, with rules
fixed in the paper (section 4.5) before any result. New outcomes and labels
live in sidecar files keyed by subject, so `timelines.jsonl` and its frozen
digest are untouched: `build` writes `causes.jsonl` (the linked file's
underlying-cause recode and its diabetes and hypertension multiple-cause flags,
with deaths lacking multiple-cause data marked) and `labels` writes
`conditions.jsonl`.

| Id | Question | Predictions | Scoring |
|---|---|---|---|
| T1 | death by cardiovascular, cancer and other causes at 5, 10 and 15 years | `cs-cox-*` baselines | `causes`: IPCW Brier, AUC, calibration per fold, paired against a reference |
| T2 | death with diabetes or hypertension listed on the certificate | `baselines/flags.py` (`flag-cox-*`, `flag-share-*`) | `flags`: pooled out-of-fold, against the all-cause ranker, cluster bootstrap |
| T3 | expected time lived over 10 or 15 years and the mortality-equivalent age | any baseline's all-cause curves | `lifeexp`; `baselines/assoc.py` for the association with outcomes |
| T4 | eight prevalent conditions at the examination | `baselines/conditions.py` | `prevalence`: per fold, against the age-and-sex model |
| T5 | undiagnosed diabetes, hypertension, kidney markers, high cholesterol | `baselines/conditions.py` | `screen`: sensitivity at a training-chosen threshold, pooled, against age and body-mass index |

| T6 | accelerometer summaries, and a network on the minute series, as extra inputs (2003 to 2006) | `baselines/accel_experiment.py` (summaries from `baselines/accel.py`) | `external --subset`, then `compare` |

A label defined by a measurement is not predicted from it: `conditions.py`
holds the list of inputs each label excludes, with a specification that none
reaches a model. Label definitions, with the codebook variables and the
standardisation of serum creatinine per cycle, are in `src/conditions.rs` and
the paper's Appendix C.

```bash
lifecourse labels --nhanes <dir> --data <data>                  # conditions.jsonl (build has written causes.jsonl)
python -I baselines/run.py --data <data> --out <data>/baselines --baseline cs-cox-agesex --jobs 8
lifecourse causes --data <data> --model cs-cox-agesex --model cs-cox-standard --model cs-cox-all --score
python -I baselines/flags.py --data <data> --out <data>/baselines --model flag-cox-all --model flag-share-all --jobs 8
lifecourse flags --data <data> --ranker cs-cox-all --model flag-cox-all --model flag-share-all
lifecourse lifeexp --data <data> --model cs-cox-all --tau 10 --aa-dir <dir>
python -I baselines/assoc.py --data <data> --aa <dir>/aa-cs-cox-all-r0-t10.jsonl
python -I baselines/conditions.py --data <data> --out <data>/conditions --jobs 8
lifecourse prevalence --data <data> --model agesex --model nondef-hgb --model full-hgb
lifecourse screen --data <data> --model agebmi --model nondef-logit --model nondef-hgb
```

Their own checks: `python -I baselines/test_flags.py`, `test_assoc.py`,
`test_conditions.py`, `test_accel.py`, `test_residual.py` and `test_curves.py`.

Beyond them, three commands evaluate any model whose predictions are files
(`<baseline>` or `recipe:<name>`):

```bash
lifecourse bootstrap   --data <data> --a spline-cox-net-all --b cox-net-all --repeat 0   # pooled out-of-fold paired difference, cluster bootstrap
lifecourse calibration --data <data> --model cs-cox-all --repeat 0                      # observed/expected and slope with intervals, ICI, E50, E90 per cause
lifecourse external    --data <data> --baseline accel-plus --subset                      # files made on a subsample of each fold
```

## The residual network and the learning curves

`baselines/residual.py` registers `resnet-cox-all-w<width>-l<layers>`: the
spline Cox model plus a multilayer perceptron whose output starts at zero,
trained on the Cox partial likelihood with the spline predictor as an offset,
the number of epochs chosen on a validation share among epochs that include
zero (the spline model itself). `-only-<block>` (`exam`, `diet`, `history`,
`questionnaire`) gives the network one block of inputs and `-shuffled` is the
leakage control; `ResidualSeq` puts a convolutional network on a per-subject
series. It needs `torch`, which the other baselines do not; without it these
names are simply not registered. `baselines/curves.py` reads the per-fold scores
of subsamples (`recipe subsample`, `run.py --train-ids`, `recipe score`) and
fits the learning curves:

```bash
python -I baselines/run.py --data <data> --out <data>/baselines --baseline resnet-cox-all-w64-l2 --jobs 16
python -I baselines/curves.py --data <data> --repeat 0 --repeat 1 --bootstrap 500
```

## Recipes

Secondary, never one of the criteria and never scored on the locked test:
`recipe` asks whether the deep encoder's score is a training-recipe problem
or a ceiling of the data. A recipe is the horizon spec as `key=value`
overrides of the arm's defaults (shape, learning rate, batch, masking rate,
patience, steps, knots, a training subsample, a bootstrap resample; the keys
are listed in `src/recipe.rs`). What brain's `TimelineSpec` does not expose
cannot be set, and asking for it is an error: weight decay is one such
hyperparameter.

Everything is kept under `<data>/recipe/<name>/`, never in `runs/`: the
out-of-fold predictions in the format of the `external` arm and a score per
fold. Every recipe, every average of recipes and every baseline is scored
from prediction files by the same function, so they sit on one footing and
`compare`, which reads them as `recipe:<name>`, pairs them on identical folds.

```bash
lifecourse recipe run     --data <data> --name d32 --set d_model=32 --set d_ff=64 --seed 1 --repeat 0
lifecourse recipe blend   --data <data> --name d32-ens --member d32-s1 --member d32-s2
lifecourse recipe subsample --data <data> --share 0.25     # ids for baselines/run.py --train-ids
lifecourse recipe score   --data <data> --name cox-net-all-p25 --made-of "..."   # score files written elsewhere
lifecourse recipe summary --data <data> --model additive --model recipe:d32-s1 --repeat 0
lifecourse compare --data <data> --a recipe:d32-s1 --b additive --repeat 0
```

A member trained with seed `s` is kept as `<name>-s<s>`; a blend may mix
recipes and `external:<baseline>` members. A subsample keeps a subject when
a hash of the seed and its id falls below the share, so a smaller share is
always inside a larger one.

## Protocol

Fixed before the first model was trained, and pinned by digest
(`FROZEN.json`): the subjects, the partition and the criteria
(`criteria.json`). The partition holds out a locked test (15%, stratified by
cycle, death and age band) that only a final candidate is scored on, and lays
out 5 repeats of 5 grouped stratified folds over the rest. Arms are compared
on identical folds with the corrected resampled t-test; the claim is made on
the locked test, with intervals that resample the survey's own clusters.
`src/report.rs` holds every criterion: an integrated Brier score better than
the standard risk factors' with an interval below zero, calibration slope
and intercept, D-calibration, no subgroup worse beyond a stated bound, and a
model trained on shuffled outcomes doing no better than age and sex.

One amendment was made after the freeze, before any arm scored the locked
test, and is pinned as its own file beside the untouched criteria
(`lifecourse amend`, `src/report.rs`): D-calibration is reported but no
longer decides, because under this much censoring its chi-square null does
not hold and the test cannot reject even a wrong model. Removing it makes
no criterion easier to pass.

Secondary, and not one of the criteria: each run also turns its ten-year
risk into a Venn-Abers interval, calibrated on the subjects it held out for
early stopping (`src/intervals.rs`), and the report shows that interval's
calibration and Brier score beside the raw prediction's.

Also secondary: `temporal` trains on the non-locked subjects of the cycles
before 2009 and scores those of 2009 and later, which tests whether a model
holds up across calendar time rather than only across random splits. The
later cycles are followed for less time, so five years is the longest
horizon it scores.

And `ensemble` scores the locked test by the mean of an arm's models trained
with several seeds on the same subjects as `final` (reusing the model
`final` saved), and reports how far the members disagree about each
subject's ten-year risk: the uncertainty that comes from training alone.

## Commands

```bash
lifecourse build   --nhanes <dir> --mortality <dir> --data <out>
lifecourse freeze  --data <out>
lifecourse cv      --data <out> --arm horizon            # every fold; --repeat/--fold to choose
lifecourse cv      --data <out> --arm horizon --permute  # the leakage check
lifecourse compare --data <out> --a horizon --b standard        # either side may be external:<baseline>
lifecourse external --data <out> --baseline km                  # secondary: score a baseline's predictions
lifecourse labels|causes|flags|lifeexp|prevalence|screen ...    # secondary estimands, see above
lifecourse final   --data <out> --arm horizon            # once; --reason to score again
lifecourse amend   --data <out>                                 # pin the post-freeze amendments
lifecourse report  --data <out>
lifecourse temporal --data <out> --arm horizon --split 2009      # secondary: calendar shift
lifecourse ensemble --data <out> --arm horizon --members 5       # secondary: seeded ensemble on the locked test
lifecourse recipe  run|blend|subsample|score|summary --data <out>   # secondary: named training variants
lifecourse intake  --nhanes <dir> --mortality <dir> --out <dir>   # add --base-url/--api-key/--model to score a model
```

`final` also saves the model it scored in `runs/<arm>-s<seed>-locked-model/`
with a manifest of its data, partition and file digests: the directory
`brain horizon predict --weights` and `BRAIN_HORIZON_DIR` serve.

`--nhanes` is a directory of `<cycle start year>/*.xpt` as CDC distributes
them, `--mortality` the `NHANES_<y>_<y+1>_MORT_2019_PUBLIC.dat` files. Set
`BRAIN_BACKEND`/`BRAIN_DEVICE` to choose brain's device.

## Intake

The exam concepts in `src/concepts.rs` were harmonised by hand: each NHANES
variable with its factor to the concept's unit and the range of real
measurements. `intake` uses that as ground truth for the intake agent's
admission rules (`splinter-knowledge`'s `harmonize::admit`). For each cycle,
the codebook entry (`*.htm`) of each concept's variable, and the cohort's
values of it, are checked against each concept's values from the other
variables or cycles. The true mapping must be admitted, and each corruption
of it rejected.

Over the ten cycles (274 variable-cycle cases), every true mapping was
admitted. Every unit error (factor times or divided by ten) and every
mapping that read documented refusal codes as values was rejected. A
variable assigned to a different concept was admitted in 812 of 7398
pairs: the rules check a unit and a distribution, so quantities that share
both (ALT and AST, two of the mg/dL laboratory values) cannot be told
apart by data alone. Which concept a variable measures rests on the
proposal's quotation of the codebook, and is scored separately when a
model is given.

## Results

Scored 2026-10-05 on the locked test (8,437 subjects; the 15-year primary
uses the 1,583 of them in the two cycles followed that long, 345 deaths).
The full tables are in `report.md` beside the data; `runs/REVISIONS` records
the code each run was built from.

**The pre-registered claim is not made.** Three of the five requirements
fail:

| requirement | result | met |
|---|---|---|
| IBS 1-15 y, horizon minus standard, interval below zero | -0.0012, interval [-0.0046, +0.0020] | no |
| calibration slope at 10 y within [0.9, 1.1] | 0.861 (predictions too extreme) | no |
| calibration intercept interval covers zero | [-0.291, +0.041] | yes |
| no subgroup worse than standard by more than 0.002 | other Hispanic (96 subjects) worse by 0.0030 | no |
| model on shuffled outcomes no better than age and sex | 0.0742 vs 0.0531 (CV) | yes |

What the evidence does show:

- Every arm ranks risk well and the deep model is better than the
  conventional risk factors on every accuracy and discrimination summary
  (not on calibration slope), but not by enough to resolve
  on 345 deaths. Locked test, IBS 1-15 y: horizon 0.0514, standard 0.0526,
  age and sex 0.0571, additive on all inputs 0.0492. Ten years, with
  design-based intervals: Brier 0.0584 [0.0519, 0.0651] against 0.0621
  [0.0555, 0.0692]; Uno C 0.893 [0.875, 0.910] against 0.877 [0.858,
  0.895].
- Cross-validation (25 folds, the 47,816 non-locked subjects) agrees in
  direction and is far more precise: IBS 0.0471 horizon, 0.0495 standard,
  0.0531 age and sex, 0.0465 additive; Uno C at 10 y 0.897, 0.880, 0.855,
  0.900. The training seed moves horizon's fold scores by about half the
  gap to standard (secondary, three seeds).
- The additive model on the same inputs is as good as the deep set encoder
  or better, in cross-validation and on the locked test: one examination per
  subject carries mostly additive signal, and the encoder adds no measurable
  value over it here.
- Secondary, not pre-registered: the mean of five seeds of horizon scores
  IBS 0.0499 on the locked test, and Venn-Abers recalibration on held-out
  subjects brings its slope from 0.861 to 0.990; trained on the cycles before
  2009 and scored on the later ones, five-year Uno C is 0.868 for horizon,
  0.880 additive, 0.854 standard.

The locked test is now spent for this claim: any further model scored on it
is an exploratory result, not a test of a prediction.
