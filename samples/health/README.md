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

Training runs on the device `--backend` and `--device` name; wrap every command that
trains in the shared device lock if the device is shared.

```bash
cargo build --release -p splinter-health
H="target/release/splinter-health --dir health-run"

$H synth                                  # source file + declaration: seed, digest, usage terms, outcomes supplied (40000 participants)
$H import                                 # one immutable episode per participant, under opaque keys
$H split                                  # train / validation (early stopping) / test; leakage-checked, stored with its seed
$H train --as baseline --steps 3 --max-tokens 16   # an untrained baseline: the first champion
$H train --as v1 --max-tokens 16                    # the candidate, calibrated at 5 and 10 years (brain's logistic default)
$H eval --as e1 --candidate v1 --champion baseline  # scored on the test part, requirements registered first
$H release --evaluation e1 --unrestricted           # released only if the gate passes (exit 3: rejected)

$H train --as weak --steps 5 --max-tokens 16        # a deliberately bad candidate
$H eval --as e2 --candidate weak --champion release:health
$H release --evaluation e2 --unrestricted           # rejected and recorded; the alias stays

$H predict                                # the released file, loaded by plain brain: forecast a patient history, append a checkup, forecast again
$H preregistration                        # the cohort, seeds and calibration rule fixed before the experiment
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
$H train --as v7 --calibration-kind venn-abers --calibration-units 6000   # another calibrator, or more calibration units
```

A seed that is not given (`synth`, `split`, `train`, `eval`) is drawn and
printed on stderr, and the command's output states it: the declaration, the
split record and the candidate's report carry `seed`, and passing it back
reproduces the draw.

To reproduce the numbers below, pass the seeds they were measured with:

```bash
$H synth --seed 7
$H split --seed 3
$H train --as baseline --steps 3 --max-tokens 16 --seed 1
$H train --as v1 --max-tokens 16 --seed 1
$H train --as weak --steps 5 --max-tokens 16 --seed 1
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
sees a test unit. The calibration gets at least 3000 units when the part
allows (`--calibration-units`; never less than half of the part, and early
stopping keeps at least a quarter), and brain's default kind, a logistic map of
the raw risk with no interval, unless `--calibration-kind` says otherwise. The
candidate record keeps the calibration units and the validation events behind
every code and horizon; a horizon with too few of them is not calibrated, and
is listed with its reason in the training report and in the lineage.

On the test part only, for the candidate and the champion on the same units
(a different unit set, a test part of another split, or a model that was fitted
or early-stopped on any test unit is refused before scoring), with brain's
survival arithmetic (brain's own `TimelineModel::evaluate` for each outcome
code; the all-cause union and the paired differences are computed here):

- per outcome code and for the all-cause union, per horizon: Uno's C,
  time-dependent AUC, IPCW Brier, calibration (slope, intercept in the large, observed over
  expected, expected calibration error against the Aalen-Johansen observed
  risk), and the integrated Brier score over a grid;
- the held-out event likelihood;
- candidate minus champion for each, with percentile intervals from a bootstrap
  that resamples whole groups (the keyed group id), not records;
- the same integrated Brier difference on named subgroups;
- calibration is judged on the calibrated risk where the model has one at that
  code and horizon and on the raw risk otherwise (a horizon the validation data
  could not support is left raw and judged raw), and the evaluation record says
  which for each horizon, with the test events it was judged on;
- serving, from the shipped file unpacked by the system `tar` and loaded by
  plain brain: its predictions against the model that was scored on every test
  unit, and against the model as trained on probe units; brain's batched
  forecast of patient histories against one forecast per history; the share of
  test units whose support score is above the edge of the training support
  (units `brain serve` would withhold an answer for); and that every served
  probability is finite and inside [0, 1], intervals (where the calibrator has
  them) are ordered, survival never rises and cumulative incidence never falls.

A horizon with fewer than ten events is absent, never zero. The requirements
are data written before scoring and registered in the evaluation: the all-cause
integrated Brier difference and the held-out likelihood must improve (interval
excludes zero), calibration must lie in the bands below at every JUDGED horizon,
no subgroup may get worse by more than 0.002, the served file must reproduce
the trained one and a batch must equal single requests, at most a tenth of the
test units may be ones the support would withhold, every served probability and
curve must be valid, and the terms must permit the release asked for. A
candidate that fails any is rejected, recorded, and leaves the alias where it
was.

## Pre-registration: the calibration rule of the release gate

This rule, the cohort and the seeds were fixed in code (`src/steps.rs`,
`src/synth.rs`, `splinter-eval`'s `TimelinePlan`) and written here before the
experiment was run, and are not changed after seeing a result.

Why. A calibration band applied to a point estimate is only as good as the
events behind the estimate. The recalibration slope of a model that is exactly
calibrated spreads by a tenth to a fifth on a few hundred events, so on a test
part with tens of events the band is dominated by noise and rejects good
models, and passes bad ones, at random; a calibrator fitted on a few dozen
events is as noisy. The earlier gate judged every horizon of a test part of
about 1800 units and rejected candidates for that reason. The rule below
judges a horizon only where the test part can tell, and says so where it
cannot.

1. Calibration uses brain's default kind (logistic) unless the request says
   otherwise. The validation part is divided between early stopping and
   calibration so that calibration gets at least 3000 units when the cohort
   allows (configurable); the calibration units and the events per code and
   horizon are recorded in the candidate record.
2. A (code, horizon) is JUDGED only if the test part holds at least 200 events
   of that code by that horizon (`--min-test-events`). Where it is, the bands
   apply to the point estimates: recalibration slope in [0.7, 1.4], observed
   over expected in [0.8, 1.25], expected calibration error at most 0.05, and
   the intercept at the average risk (`intercept_in_the_large`) in [-0.5, 0.5].
3. A horizon with fewer test events is not judged and is recorded as
   UNCERTIFIED: not zero, not failed. It is not exposed as a calibrated risk
   output of the release. The release manifest records the certified horizons
   per code; `predict` gives calibrated values only at those, and the raw risk
   with a note elsewhere.
4. The REQUIRED pairs must be certified for a release: by default the 5-year
   horizon of every outcome code (`--require`). A required pair the test part
   cannot judge for lack of events fails closed: the candidate is rejected with
   that reason, so a cohort too small to certify cannot release.
5. Everything else in the gate (performance, retention, serving, policy) is
   unchanged.

The cohort is `synth`'s default of 40000 participants, large enough that the
required horizons are certifiable (the test part is 30 percent of it; the
events of each code by each horizon are counted on the test part and kept in
the evaluation record). The seeds are {1, 2, 3, 4, 5}, the training seed
(`train --seed`); the data (`synth --seed 7`), the split (`split --seed 3`)
and the bootstrap seed are fixed. Each seed is trained, evaluated against the baseline champion and put through a release
attempt, in a run directory of its own, and every outcome is reported, released
or rejected. Every run states the seeds it used, so it can be repeated; the
integration tests train twice and require identical data membership,
configuration, lineage and metrics. A candidate rejected by a band is the gate
working: the bands are not changed to make a run pass.

## Predict: a patient history

Only certified horizons carry calibrated values. `predict` reads the release
manifest's certified horizons and removes the calibrated risk elsewhere. A
consumer that loads the file with plain brain (`brain serve`) is not held to
the manifest: brain answers a calibrated risk at every horizon its
`calibration.json` holds, so such a consumer must read the certified horizons
from the manifest itself.

`predict` unpacks the released file with the system `tar`, loads it with brain
alone and sends brain's patient-history format
(`{"as_of", "calendar", "events": [{"time", "code", "value", "unit"}]}`: a record
with a value is a measurement, one without is an event) for one test subject.
Appending a checkup means sending the whole history again with one more record
and the as-of time moved to it; nothing is kept between calls. The report prints
both forecasts as brain returns them: the as-of time, the model's identity (the
digests of the weights and the configuration, the brain version), what the
history covered and which variables it lacked, the curves over the model's knots,
the risk at each requested horizon (raw, and calibrated, with an interval
where the calibrator has one, only where the release certified that code and
horizon; elsewhere the calibrated keys are removed and a `note` says the horizon
is uncertified and the raw risk is given), the support assessment with its warnings, the data-quality
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
  --data-dir "$HEALTH_DATA" --out-dir "$OUT_DIR"   # tables, results.json, report.md; the seed is drawn, printed and recorded
```
