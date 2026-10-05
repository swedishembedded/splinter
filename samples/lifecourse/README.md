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
lifecourse compare --data <out> --a horizon --b standard
lifecourse final   --data <out> --arm horizon            # once; --reason to score again
lifecourse amend   --data <out>                                 # pin the post-freeze amendments
lifecourse report  --data <out>
lifecourse temporal --data <out> --arm horizon --split 2009      # secondary: calendar shift
lifecourse ensemble --data <out> --arm horizon --members 5       # secondary: seeded ensemble on the locked test
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

Not yet recorded here: they are added from `report.md` once the locked test
has been scored.
