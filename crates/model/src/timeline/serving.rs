// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verification that a released risk model
// serves what was evaluated: the same numbers from the shipped file, the same
// answer alone or in a batch, honest abstention outside its training support,
// and probabilities that are probabilities. If your team needs expertise in
// serving correctness for prediction models, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Serving correctness, measured on held-out units.
//!
//! A model that scores well in memory can still ship wrong: the file may not
//! reproduce it, a batch may not agree with a single request, the units it is
//! asked about may lie outside anything it was trained on, or a curve may not
//! be one. [`measure_serving`] puts numbers on each of those for two loads of
//! the same model - the one that was scored (`reference`) and a fresh one
//! (`served`) the caller unpacked and loaded with plain brain - and the
//! measurement crate's pre-registered requirements decide them. Nothing here
//! decides; an unmeasured quantity is `None`, never zero, and a mismatch in
//! shape is `None` too, never a small difference.
//!
//! * identity: every prediction of `reference` against `served`, raw and
//!   calibrated, with the calibrated interval, on every unit;
//! * batching: brain's batch forecast of the units' patient histories against
//!   one forecast per history ([`history_of`] writes a unit as the history a
//!   caller would send);
//! * abstention: the share of the units whose support score is above the edge
//!   of the training support, which `brain serve` would withhold;
//! * validity: every served probability finite and inside [0, 1], intervals
//!   ordered, survival never rising and cumulative incidence never falling.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};

use super::training::TimelineError;
use super::{PatientHistory, Subject, TimelineModel};
use brain::timeline::ForecastRequest;

/// The units whose patient histories are forecast in a batch and one by one
/// unless the caller asks for another number.
pub const DEFAULT_BATCH_SAMPLE: usize = 32;
/// Points of the fine time grid the validity of the curves is checked on, in
/// addition to the model's knots and the requested horizons.
const GRID_POINTS: usize = 40;
/// How far a curve may step the wrong way before it is not monotone: float
/// noise, not a model.
const MONOTONE_SLACK: f64 = 1e-9;

/// What to measure.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ServingSpec {
    /// The horizons the served forecasts and calibrated risks are taken at;
    /// each inside the model's last knot.
    pub horizons: Vec<f64>,
    /// The first this many units are forecast as histories, batched and one by
    /// one.
    pub batch_sample: usize,
}

impl ServingSpec {
    /// Measure at `horizons` with [`DEFAULT_BATCH_SAMPLE`] histories.
    #[must_use]
    pub fn new(horizons: Vec<f64>) -> Self {
        Self {
            horizons,
            batch_sample: DEFAULT_BATCH_SAMPLE,
        }
    }
}

/// What was measured about serving.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ServingMeasured {
    /// Units measured.
    pub units: usize,
    /// The largest absolute difference between the two loads' predictions;
    /// `None` when they disagree in shape (a value one has and the other
    /// lacks), which is no difference to report and never zero.
    pub identity_max_abs_diff: Option<f64>,
    /// Histories forecast both ways.
    pub histories: usize,
    /// The largest absolute difference between a batched forecast and the
    /// same history forecast alone; `None` when the two disagree in shape (one
    /// abstained, a field differs).
    pub batch_max_abs_diff: Option<f64>,
    /// Units whose support score is above the edge of the support, over units
    /// assessed. `None` when the model records no support (unknown is not
    /// zero).
    pub abstention_rate: Option<f64>,
    /// Probabilities that are not finite or not inside [0, 1] (including
    /// interval ends and an interval whose ends are out of order).
    pub invalid_probabilities: usize,
    /// Curves (a unit's survival, or one code's cumulative incidence) that
    /// step the wrong way.
    pub non_monotone_curves: usize,
}

fn failed(what: &str, e: impl std::fmt::Display) -> TimelineError {
    TimelineError::Brain(format!("serving check: {what}: {e}"))
}

/// The patient history a caller would send for `subject`, as the JSON of
/// brain's update format: the measurements at or before its entry and the
/// events strictly before it, on the model's own clock, with the calendar
/// time stated. Later records are outcomes, not history, and are not sent.
#[must_use]
pub fn history_json(subject: &Subject) -> Json {
    let mut records = Vec::new();
    for o in subject.observations.iter().filter(|o| o.t <= subject.entry) {
        let mut record = json!({"time": o.t, "code": o.var, "value": o.value});
        if let Some(unit) = &o.unit {
            record["unit"] = json!(unit);
        }
        records.push(record);
    }
    for e in subject.events.iter().filter(|e| e.t < subject.entry) {
        records.push(json!({"time": e.t, "code": e.code}));
    }
    json!({
        "id": subject.subject_id,
        "as_of": subject.entry,
        "calendar": subject.calendar_at_entry,
        "events": records,
    })
}

/// [`history_json`] read back by brain's own parser, which refuses what the
/// format does not allow.
pub fn history_of(subject: &Subject) -> Result<PatientHistory, TimelineError> {
    parse_history(&history_json(subject))
}

/// One patient history read by brain's parser: an error names the record and
/// the field that is wrong.
pub fn parse_history(history: &Json) -> Result<PatientHistory, TimelineError> {
    PatientHistory::parse_all(&history.to_string())
        .map_err(|e| TimelineError::Request(format!("patient history: {e}")))?
        .pop()
        .ok_or_else(|| TimelineError::Request("patient history: none was given".into()))
}

/// The largest absolute difference between two JSON documents of one shape:
/// numbers differ by their distance, everything else must be equal. NaN for
/// any difference in shape, so a mismatch is never read as agreement.
fn json_max_abs_diff(a: &Json, b: &Json) -> f64 {
    match (a, b) {
        (Json::Number(x), Json::Number(y)) => match (x.as_f64(), y.as_f64()) {
            (Some(x), Some(y)) => (x - y).abs(),
            _ => f64::NAN,
        },
        (Json::Array(x), Json::Array(y)) if x.len() == y.len() => x
            .iter()
            .zip(y)
            .map(|(x, y)| json_max_abs_diff(x, y))
            .fold(0.0, nan_max),
        (Json::Object(x), Json::Object(y)) if x.len() == y.len() => x
            .iter()
            .map(|(k, v)| y.get(k).map_or(f64::NAN, |w| json_max_abs_diff(v, w)))
            .fold(0.0, nan_max),
        _ if a == b => 0.0,
        _ => f64::NAN,
    }
}

/// The larger of two differences, NaN if either is: `f64::max` would drop it.
fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// The times the curves are read at: the model's knots, the horizons and an
/// even grid up to the last knot, ascending.
fn time_grid(model: &TimelineModel, horizons: &[f64]) -> Vec<f64> {
    let last = model.config().knots.last().copied().map_or(0.0, f64::from);
    let mut times: Vec<f64> = model
        .config()
        .knots
        .iter()
        .copied()
        .map(f64::from)
        .collect();
    times.extend(horizons);
    times.extend((1..=GRID_POINTS).map(|k| last * k as f64 / GRID_POINTS as f64));
    times.sort_by(f64::total_cmp);
    times.dedup();
    times
}

/// Every number `model` serves for `subjects` as a list the same model loaded
/// elsewhere must reproduce: per unit, the survival and each code's raw
/// cumulative incidence over `times`, then at each horizon the calibrated
/// risk and its interval ends (`None` where the model has none).
fn served_numbers(
    model: &TimelineModel,
    subjects: &[Subject],
    times: &[f64],
    horizons: &[f64],
) -> Result<Vec<Option<f64>>, TimelineError> {
    let mut out = Vec::new();
    for p in model.predict(subjects)? {
        for t in times {
            out.push(Some(p.survival(*t)));
            for code in model.codes() {
                out.push(p.cif(code, *t));
            }
        }
        for code in model.codes() {
            for h in horizons {
                out.push(p.calibrated_cif(code, *h));
                let interval = p.cif_interval(code, *h);
                out.push(interval.map(|i| i.0));
                out.push(interval.map(|i| i.1));
            }
        }
    }
    Ok(out)
}

/// The largest difference between two lists of served numbers; a value one
/// load has and the other lacks, or lists of different lengths, are NaN.
fn max_diff(a: &[Option<f64>], b: &[Option<f64>]) -> f64 {
    if a.len() != b.len() {
        return f64::NAN;
    }
    a.iter().zip(b).fold(0.0, |worst, pair| {
        nan_max(
            worst,
            match pair {
                (Some(x), Some(y)) => (x - y).abs(),
                (None, None) => 0.0,
                _ => f64::NAN,
            },
        )
    })
}

fn is_probability(x: f64) -> bool {
    x.is_finite() && (0.0..=1.0).contains(&x)
}

/// Validity of what `model` serves for `subjects`: the count of probabilities
/// that are not probabilities, and of curves that are not monotone.
fn validity(
    model: &TimelineModel,
    subjects: &[Subject],
    times: &[f64],
    horizons: &[f64],
) -> Result<(usize, usize), TimelineError> {
    let (mut invalid, mut non_monotone) = (0, 0);
    for p in model.predict(subjects)? {
        let survival: Vec<f64> = times.iter().map(|t| p.survival(*t)).collect();
        invalid += survival.iter().filter(|x| !is_probability(**x)).count();
        non_monotone += usize::from(survival.windows(2).any(|w| w[1] > w[0] + MONOTONE_SLACK));
        for code in model.codes() {
            let cif: Vec<f64> = times
                .iter()
                .map(|t| p.cif(code, *t).unwrap_or(f64::NAN))
                .collect();
            invalid += cif.iter().filter(|x| !is_probability(**x)).count();
            non_monotone += usize::from(cif.windows(2).any(|w| w[1] < w[0] - MONOTONE_SLACK));
            for h in horizons {
                if let Some(risk) = p.calibrated_cif(code, *h) {
                    invalid += usize::from(!is_probability(risk));
                }
                if let Some((lo, hi)) = p.cif_interval(code, *h) {
                    invalid += usize::from(!is_probability(lo) || !is_probability(hi) || lo > hi);
                }
            }
        }
    }
    Ok((invalid, non_monotone))
}

/// Batched against single forecasts of the first `spec.batch_sample` units'
/// histories: `(histories, largest difference)`.
fn batch_agreement(
    served: &TimelineModel,
    subjects: &[Subject],
    spec: &ServingSpec,
) -> Result<(usize, f64), TimelineError> {
    let histories: Vec<PatientHistory> = subjects
        .iter()
        .take(spec.batch_sample)
        .map(history_of)
        .collect::<Result<_, _>>()?;
    let request = ForecastRequest::new(spec.horizons.iter().copied());
    let batched = served
        .forecast_histories(&histories, &request)
        .map_err(|e| failed("batched forecast", e))?;
    let mut worst = 0.0;
    for (history, together) in histories.iter().zip(&batched) {
        let alone = served
            .forecast_history(history, &request)
            .map_err(|e| failed("single forecast", e))?;
        let (a, b) = (
            serde_json::to_value(together).map_err(|e| failed("forecast", e))?,
            serde_json::to_value(&alone).map_err(|e| failed("forecast", e))?,
        );
        worst = nan_max(worst, json_max_abs_diff(&a, &b));
    }
    Ok((histories.len(), worst))
}

/// The share of `subjects` whose support score is above the edge of the
/// support, `None` when the model records no support or there are no units.
fn abstention_rate(
    model: &TimelineModel,
    subjects: &[Subject],
) -> Result<Option<f64>, TimelineError> {
    let assessed = model.assess(subjects)?;
    if assessed.is_empty() || assessed.iter().any(|a| a.supported.is_none()) {
        return Ok(None);
    }
    let withheld = assessed
        .iter()
        .filter(|a| a.supported == Some(false))
        .count();
    Ok(Some(withheld as f64 / assessed.len() as f64))
}

/// Measures serving correctness on `subjects` (held-out units): `reference`
/// is the model whose numbers were scored, `served` a fresh load of the same
/// file by plain brain. Refused when there are no units or a horizon is not
/// inside the model's knots.
pub fn measure_serving(
    reference: &TimelineModel,
    served: &TimelineModel,
    subjects: &[Subject],
    spec: &ServingSpec,
) -> Result<ServingMeasured, TimelineError> {
    if subjects.is_empty() {
        return Err(TimelineError::Request(
            "serving check: there are no held-out units to measure on".into(),
        ));
    }
    let times = time_grid(served, &spec.horizons);
    let identity = max_diff(
        &served_numbers(reference, subjects, &times, &spec.horizons)?,
        &served_numbers(served, subjects, &times, &spec.horizons)?,
    );
    let (histories, batch_max_abs_diff) = batch_agreement(served, subjects, spec)?;
    let (invalid_probabilities, non_monotone_curves) =
        validity(served, subjects, &times, &spec.horizons)?;
    Ok(ServingMeasured {
        units: subjects.len(),
        identity_max_abs_diff: (!identity.is_nan()).then_some(identity),
        histories,
        batch_max_abs_diff: (!batch_max_abs_diff.is_nan()).then_some(batch_max_abs_diff),
        abstention_rate: abstention_rate(served, subjects)?,
        invalid_probabilities,
        non_monotone_curves,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_difference_in_shape_is_never_agreement() {
        let a = json!({"x": [1.0, 2.0], "name": "m", "gone": null});
        assert_eq!(json_max_abs_diff(&a, &a), 0.0);
        let b = json!({"x": [1.0, 2.5], "name": "m", "gone": null});
        assert_eq!(json_max_abs_diff(&a, &b), 0.5);
        for other in [
            json!({"x": [1.0], "name": "m", "gone": null}),
            json!({"x": [1.0, 2.0], "name": "n", "gone": null}),
            json!({"x": [1.0, 2.0], "name": "m", "gone": 0.0}),
            json!({"x": [1.0, 2.0], "name": "m", "here": null}),
        ] {
            assert!(json_max_abs_diff(&a, &other).is_nan(), "{other}");
        }
        assert!(nan_max(1.0, f64::NAN).is_nan() && nan_max(f64::NAN, 1.0).is_nan());
    }

    #[test]
    fn a_value_one_load_lacks_is_a_mismatch_not_zero() {
        assert_eq!(max_diff(&[Some(0.5), None], &[Some(0.5), None]), 0.0);
        assert!(max_diff(&[Some(0.5)], &[None]).is_nan());
        assert!(max_diff(&[Some(0.5)], &[]).is_nan());
        assert_eq!(max_diff(&[Some(0.5)], &[Some(0.25)]), 0.25);
    }

    #[test]
    fn a_unit_is_written_as_the_history_a_caller_would_send() {
        let (subjects, _) = super::super::synthetic::population(3, 1);
        let s = &subjects[0];
        let h = history_of(s).unwrap();
        assert_eq!(h.id, s.subject_id);
        assert!((h.as_of - s.entry).abs() < 1e-12);
        assert!((h.calendar_at_as_of - s.calendar_at_entry).abs() < 1e-9);
        assert!(h.warnings.is_empty(), "{:?}", h.warnings);
    }
}
