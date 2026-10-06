# health: a gated, traceable release of a risk model

Does a candidate risk model replace the one in place only when held-out
evidence says it should, and can the release be traced to every participant
record it was trained on?

A controlled sample on a synthetic cohort whose hazards are known (brain's
synthetic population: two competing causes of death, `death:cvd` driven by age
and the risk factor `x1`, `death:cancer` by age, `x2` and a prior diagnosis),
written on Splinter's SDK alone. Nothing in it is health data; `real` points
the same loop at a directory of `timeline-v1` files you hold.

## Run it

Training runs on the device `BRAIN_BACKEND` names; wrap every command that
trains in the shared device lock if the device is shared.

```bash
cargo build --release -p splinter-health
H="target/release/splinter-health --dir health-run"

$H synth                                  # source file + declaration: digest, usage terms, outcomes supplied
$H import                                 # one immutable episode per participant, under opaque keys
$H split                                  # train / validation (early stopping) / test; leakage-checked, stored
$H train --as baseline --steps 3 --max-tokens 16   # an untrained baseline: the first champion
$H train --as v1 --max-tokens 16                    # the candidate, calibrated at 5 and 10 years
$H eval --as e1 --candidate v1 --champion baseline  # scored on the test part, requirements registered first
$H release --evaluation e1 --unrestricted           # released only if the gate passes (exit 3: rejected)

$H train --as weak --steps 5 --max-tokens 16        # a deliberately bad candidate
$H eval --as e2 --candidate weak --champion release:health
$H release --evaluation e2 --unrestricted           # rejected and recorded; the alias stays

$H predict                                # the released file, loaded by plain brain: forecast a patient history, append a checkup, forecast again
$H lineage                                # release -> run -> datasets -> episodes -> source lines -> file digests
```

`train` also passes brain's optional heads and states through to the training
configuration; with none of them the model is the default one:

```bash
$H train --as v2 --visits 4 --mixer gated-delta-net --blocks 2     # a stack of sequence mixers over the visits
$H train --as v3 --visits 4 --backbone attention                   # attention over the visits
$H train --as v4 --next-events dx,death:cvd --next-events-weight 0.5   # which event comes first, as a training signal
$H train --as v5 --forecasts 2 --forecast-weight 0.5               # a head for later measurements
$H train --as v6 --calibrate-at 5,10 | --no-calibration           # which horizons are calibrated, or none
```

Each command prints a JSON report (and keeps it under `health-run/reports/`).
Terms: `synth --usage research_only` (or `restricted_DUA`) declares terms that
forbid redistribution; `release --unrestricted` on such data is rejected by the
data-policy check, and without `--unrestricted` the release is restricted and
allowed. Terms nobody declared are `unknown` and permit nothing, not even
training.

## What is measured

Calibration comes first. The validation part is divided by participant group
into the units early stopping reads and the units the calibration is fitted
on, so the calibrators never see what the weights were chosen on and neither
sees a test unit. A (code, horizon) with too few validation events is not
calibrated, and is listed in the training report and in the lineage.

On the test part only, for the candidate and the champion on the same units
(a different unit set, a test part of another split, or a model that was fitted
or early-stopped on any test unit is refused before scoring), with brain's
survival arithmetic (brain's own `TimelineModel::evaluate` for each outcome
code; the all-cause union and the paired differences are computed here):

- per outcome code and for the all-cause union, per horizon: Uno's C,
  time-dependent AUC, IPCW Brier, calibration (slope, intercept, observed over
  expected, expected calibration error against the Aalen-Johansen observed
  risk), and the integrated Brier score over a grid;
- the held-out event likelihood;
- candidate minus champion for each, with percentile intervals from a bootstrap
  that resamples whole groups (the keyed group id), not records;
- the same integrated Brier difference on named subgroups;
- calibration is judged on the calibrated risk where the model has one at that
  code and horizon and on the raw risk otherwise, and the evaluation record says
  which for each; a horizon brain declared uncalibrated has no calibration numbers
  at all, so its requirement fails as unmeasured, never as zero;
- serving, from the shipped file unpacked by the system `tar` and loaded by
  plain brain: its predictions against the model that was scored on every test
  unit, and against the model as trained on probe units; brain's batched
  forecast of patient histories against one forecast per history; the share of
  test units whose support score is above the edge of the training support
  (units `brain serve` would withhold an answer for); and that every served
  probability is finite and inside [0, 1], intervals are ordered, survival
  never rises and cumulative incidence never falls.

A horizon with fewer than ten events is absent, never zero. The requirements
are data written before scoring and registered in the evaluation: the all-cause
integrated Brier difference and the held-out likelihood must improve (interval
excludes zero), calibration must lie in bands at every judged horizon (slope
0.7 to 1.4, intercept -0.5 to 0.5, observed over expected 0.8 to 1.25, ECE at
most 0.05), no subgroup may get worse by more than 0.002, the served file must
reproduce the trained one and a batch must equal single requests, at most a
tenth of the test units may be ones the support would withhold, every served
probability and curve must be valid, and the terms must permit the release asked
for. A
candidate that fails any is rejected, recorded, and leaves the alias where it
was.

With calibration on half of the validation part (840 units at the default
cohort size), the default candidate (seed 1) passes performance, retention and
serving and is rejected by one calibration band: the all-cause intercept at the
first horizon, judged on the raw risk because brain calibrates outcome codes,
not their union, lies just outside its band. Seed 2 is rejected on the
calibrated risk of both causes at the first horizon. The bands were not changed
to make a run pass; a larger cohort (`synth --n`) or a lower-variance
calibration is what would.

The bands are a design choice and the test is finite: with these defaults a
candidate trained with another seed can miss a band by a little and be
rejected, which is the gate working. Seeds and data are fixed in the commands
above so a run is reproducible; the integration tests train twice and require
identical data membership, configuration, lineage and metrics.

## Predict: a patient history

`predict` unpacks the released file with the system `tar`, loads it with brain
alone and sends brain's patient-history format
(`{"as_of", "calendar", "events": [{"time", "code", "value", "unit"}]}`: a record
with a value is a measurement, one without is an event) for one test subject.
Appending a checkup means sending the whole history again with one more record
and the as-of time moved to it; nothing is kept between calls. The report prints
both forecasts as brain returns them: the as-of time, the model's identity (the
digests of the weights and the configuration, the brain version), what the
history covered and which variables it lacked, the curves over the model's knots,
the risk at each requested horizon (raw, and calibrated with its Venn-Abers
interval only where the model was calibrated for exactly that horizon, otherwise
those keys are absent), the support assessment with its warnings, the data-quality
warnings about the history, and whether the model abstained.

```bash
$H predict --subject 0 --var x1 --value 2.0 --unit sd --after 2 --horizons 5,10
```

The synthetic cohort states `sd` as the unit of `x1`, so the model records it: a
checkup stated in another unit is refused by name, a malformed unit (empty, padded
with whitespace, longer than 32 characters) is refused when the history is read,
and a measurement with no unit is accepted as the model's. A checkup already in the
history (the same time, code, value and unit) changes nothing: the forecast is
bit-identical and the duplicate is reported among the warnings.

## Layout

| File | What |
|---|---|
| `src/synth.rs` | the cohort and its declaration |
| `src/source.rs` | declarations, digests, usage terms, the policy file |
| `src/ontology.rs`, `ontology/` | the outcome ontology as data (read-only copy), the datasets listed for each outcome, the NHANES cause recode with golden tests |
| `src/steps.rs` | import, split, train, eval, release as calls into the SDK |
| `src/predict.rs` | plain brain on the released file: a patient history forecast before and after an appended checkup |
| `src/lineage.rs` | the lineage as text |
| `src/real.rs` | the same loop on a directory of timeline-v1 files |

## Real data

```bash
echo '{"terms": {"*": "research_only"}, "subgroups": [{"name": "female", "var": "sex", "level": "female"}]}' > policy.json
$H real --timelines "$TIMELINES_DIR" --policy policy.json   # imports timelines*.jsonl, splits
```

`TIMELINES_DIR` is a directory with one `timelines*.jsonl` of `timeline-v1`
records, for instance the NHANES timelines the `lifecourse` sample builds. The
files are imported as the ontology's `nhanes_mortality` dataset (`--dataset`
changes it; the outcomes asked for must be listed for it). With no `--policy`
the terms are `unknown`: `train` refuses until someone who holds the data's
terms declares them. No data is kept in this repository.

## Fasting-trial transition analysis

`analysis/fasting_transitions.py` asks, for each readable fasting trial taken
alone (the protocols differ, so trials are never pooled), whether the assigned
arm improves out-of-sample prediction of end-of-intervention outcomes beyond
baseline covariates. It is an evaluation of learned signal, not evidence of a
treatment policy and not a claim about lifespan.

Inputs are a directory holding `bath_if_rct/`, `timet/` and `queen_mary_tre/`
as downloaded (read-only). The Queen Mary file is a single-arm pilot, so it
gets a table and an uncontrolled within-person summary but no arm model.
Per outcome with enough complete cases it runs repeated cross-validation with
ridge (alpha chosen on training rows only) for baseline-only, baseline plus
arm, and arm x baseline-value models, and reports the paired gain of the arm
with a participant bootstrap interval beside the unadjusted randomised effect
and the smallest difference the trial could detect. Observed adherence
measures are post-randomisation, kept under their own role, and enter only an
exploratory model that is labelled non-causal. Missing values are dropped per
outcome and counted, never imputed.

```bash
python -I samples/health/analysis/fasting_transitions.py --self-test
python -I samples/health/analysis/fasting_transitions.py \
  --data-dir "$HEALTH_DATA" --out-dir "$OUT_DIR"   # tables, results.json, report.md
```
