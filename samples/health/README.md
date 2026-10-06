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
$H train --as v1 --max-tokens 16                    # the candidate
$H eval --as e1 --candidate v1 --champion baseline  # scored on the test part, requirements registered first
$H release --evaluation e1 --unrestricted           # released only if the gate passes (exit 3: rejected)

$H train --as weak --steps 5 --max-tokens 16        # a deliberately bad candidate
$H eval --as e2 --candidate weak --champion release:health
$H release --evaluation e2 --unrestricted           # rejected and recorded; the alias stays

$H predict                                # the released file, loaded by plain brain; append a checkup; predict again
$H lineage                                # release -> run -> datasets -> episodes -> source lines -> file digests
```

Each command prints a JSON report (and keeps it under `health-run/reports/`).
Terms: `synth --usage research_only` (or `restricted_DUA`) declares terms that
forbid redistribution; `release --unrestricted` on such data is rejected by the
data-policy check, and without `--unrestricted` the release is restricted and
allowed. Terms nobody declared are `unknown` and permit nothing, not even
training.

## What is measured

On the test part only, for the candidate and the champion on the same units
(a different unit set, a test part of another split, or a model that was fitted
or early-stopped on any test unit is refused before scoring), with brain's
survival arithmetic:

- per outcome code and for the all-cause union, per horizon: Uno's C,
  time-dependent AUC, IPCW Brier, calibration (slope, intercept, observed over
  expected, expected calibration error against the Aalen-Johansen observed
  risk), and the integrated Brier score over a grid;
- the held-out event likelihood;
- candidate minus champion for each, with percentile intervals from a bootstrap
  that resamples whole groups (the keyed group id), not records;
- the same integrated Brier difference on named subgroups;
- serving: the shipped file, unpacked and loaded by plain brain, must predict
  what the model as trained predicted on probe units.

A horizon with fewer than ten events is absent, never zero. The requirements
are data written before scoring and registered in the evaluation: the all-cause
integrated Brier difference and the held-out likelihood must improve (interval
excludes zero), calibration must lie in bands at every judged horizon (slope
0.7 to 1.4, intercept -0.5 to 0.5, observed over expected 0.8 to 1.25, ECE at
most 0.05), no subgroup may get worse by more than 0.002, the served file must
reproduce the trained one, and the terms must permit the release asked for. A
candidate that fails any is rejected, recorded, and leaves the alias where it
was.

The bands are a design choice and the test is finite: with these defaults a
candidate trained with another seed can miss a band by a little and be
rejected, which is the gate working. Seeds and data are fixed in the commands
above so a run is reproducible; the integration tests train twice and require
identical data membership, configuration, lineage and metrics.

## Layout

| File | What |
|---|---|
| `src/synth.rs` | the cohort and its declaration |
| `src/source.rs` | declarations, digests, usage terms, the policy file |
| `src/ontology.rs`, `ontology/` | the outcome ontology as data (read-only copy), the datasets listed for each outcome, the NHANES cause recode with golden tests |
| `src/steps.rs` | import, split, train, eval, release as calls into the SDK |
| `src/predict.rs` | plain brain on the released file |
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
