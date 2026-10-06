// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verification that a released risk model
// predicts, from its shipped file, what was trained and evaluated, for its
// clients. If your team needs expertise in serving correctness for released
// models, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The probe a shipped file is held to: a model's cumulative incidences on a
//! few units at a few horizons, taken as trained and compared with what the
//! packed, unpacked and reloaded file predicts.

use super::training::TimelineError;
use super::{Subject, TimelineModel};

/// The largest absolute difference between two equal-length lists of
/// predictions; NaN when they differ in length (a mismatch is never zero).
#[must_use]
pub fn max_abs_difference(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::NAN;
    }
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// [`max_abs_difference`] over values a model may not have: NaN when the
/// lists differ in length or when one side has a value the other lacks.
#[must_use]
pub fn max_abs_difference_optional(a: &[Option<f64>], b: &[Option<f64>]) -> f64 {
    if a.len() != b.len() {
        return f64::NAN;
    }
    let mut worst = 0.0f64;
    for pair in a.iter().zip(b) {
        match pair {
            (Some(x), Some(y)) => worst = worst.max((x - y).abs()),
            (None, None) => {}
            _ => return f64::NAN,
        }
    }
    worst
}

/// The calibrated risk behind each value [`probe_values`] gives, in the same
/// order; `None` where the model has no calibrated risk at that code and
/// horizon (never zero). Empty when the model has no calibration at all.
pub fn probe_calibrated(
    model: &TimelineModel,
    subjects: &[Subject],
    codes: &[String],
    horizons: &[f64],
) -> Result<Vec<Option<f64>>, TimelineError> {
    if model.calibration().is_none() {
        return Ok(Vec::new());
    }
    let predictions = model.predict(subjects)?;
    let mut values = Vec::with_capacity(subjects.len() * codes.len() * horizons.len());
    for p in &predictions {
        for code in codes {
            for h in horizons {
                values.push(p.calibrated_cif(code, *h));
            }
        }
    }
    Ok(values)
}

/// A model's predictions the release serves are checked against: each of
/// `codes`' cumulative incidence at each horizon for `subjects`, flattened in
/// subject, code, horizon order.
pub fn probe_values(
    model: &TimelineModel,
    subjects: &[Subject],
    codes: &[String],
    horizons: &[f64],
) -> Result<Vec<f64>, TimelineError> {
    let predictions = model.predict(subjects)?;
    let mut values = Vec::with_capacity(subjects.len() * codes.len() * horizons.len());
    for p in &predictions {
        for code in codes {
            for h in horizons {
                values.push(p.cif(code, *h).ok_or_else(|| {
                    TimelineError::Request(format!("the model predicts no code {code:?}"))
                })?);
            }
        }
    }
    Ok(values)
}
