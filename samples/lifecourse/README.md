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

## Commands

```bash
lifecourse build   --nhanes <dir> --mortality <dir> --data <out>
lifecourse freeze  --data <out>
lifecourse cv      --data <out> --arm horizon            # every fold; --repeat/--fold to choose
lifecourse cv      --data <out> --arm horizon --permute  # the leakage check
lifecourse compare --data <out> --a horizon --b standard
lifecourse final   --data <out> --arm horizon            # once; --reason to score again
lifecourse report  --data <out>
```

`--nhanes` is a directory of `<cycle start year>/*.xpt` as CDC distributes
them, `--mortality` the `NHANES_<y>_<y+1>_MORT_2019_PUBLIC.dat` files. Set
`BRAIN_BACKEND`/`BRAIN_DEVICE` to choose brain's device.

## Results

Not yet recorded here: they are added from `report.md` once the locked test
has been scored.
