// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verification that a shipped risk model, as a
// consumer unpacks and loads it, serves what was trained and evaluated. If
// your team needs expertise in serving correctness for released prediction
// models, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The serving-correctness inputs of the predictive gate for a timeline
//! candidate.
//!
//! The shipped file is unpacked into a fresh directory by the system's own
//! `tar` (the file is a plain ustar archive, so a consumer needs nothing of
//! Splinter's to open it) and loaded by plain brain. That load is then
//! measured ([`splinter_model::timeline::serving`]) against the model whose
//! numbers were scored, on the held-out test units: identity of every
//! prediction, batched against single forecasts, the share of units the
//! model's own support would withhold, and the validity of every served
//! probability and curve. The file must also still predict, on the probe
//! units the candidate was packed with, what the model as trained predicted.
//! Each number is filed under the name the pre-registered requirement reads;
//! one that could not be measured is not filed, and its requirement fails.

use std::path::Path;

use serde::{Deserialize, Serialize};
use splinter_eval::metric_gate::Evidence;
use splinter_eval::timeline_metrics::{
    SERVE_ABSTENTION_RATE, SERVE_BATCH_MAX_ABS_DIFF, SERVE_INVALID_PROBABILITIES,
    SERVE_MAX_ABS_DIFF, SERVE_NON_MONOTONE_CURVES,
};
use splinter_model::timeline::probe::{
    max_abs_difference, max_abs_difference_optional, probe_calibrated, probe_values,
};
use splinter_model::timeline::serving::{measure_serving, ServingMeasured, ServingSpec};
use splinter_model::timeline::{Subject, TimelineModel};
use splinter_orchestrator::error::OrchestratorError;

use super::records::TimelineCandidate;
use super::train::lift;

/// What the serving check measured, kept in the evaluation record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ServingCheck {
    /// The files the unpacked directory holds, in name order: every file the
    /// candidate was packed with, and nothing else.
    pub files: Vec<String>,
    /// The largest difference between the unpacked file's predictions on the
    /// candidate's probe units and the model as trained then; `None` when the
    /// probe could not be reproduced.
    pub probe_max_abs_diff: Option<f64>,
    /// The measurements on the held-out test units.
    pub measured: ServingMeasured,
}

/// Unpacks `bundle` into the new directory `dest` with the system `tar`: a
/// consumer's way in, independent of Splinter's own reader.
pub fn unpack_with_tar(bundle: &Path, dest: &Path) -> Result<Vec<String>, OrchestratorError> {
    std::fs::create_dir_all(dest).map_err(|source| OrchestratorError::Io {
        path: dest.to_path_buf(),
        source,
    })?;
    let status = std::process::Command::new("tar")
        .arg("-xf")
        .arg(bundle)
        .arg("-C")
        .arg(dest)
        .status()
        .map_err(|e| {
            OrchestratorError::Train(format!("running tar to unpack {}: {e}", bundle.display()))
        })?;
    if !status.success() {
        return Err(OrchestratorError::Train(format!(
            "tar could not unpack {}",
            bundle.display()
        )));
    }
    let mut files: Vec<String> = std::fs::read_dir(dest)
        .map_err(|source| OrchestratorError::Io {
            path: dest.to_path_buf(),
            source,
        })?
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .collect();
    files.sort();
    Ok(files)
}

/// What the unpacked file predicts on the candidate's probe units against
/// what the model as trained predicted, raw and calibrated; `None` when the
/// probe units are not among `early_stopping` or the two disagree in shape.
fn probe_difference(
    record: &TimelineCandidate,
    served: &TimelineModel,
    early_stopping: &[Subject],
) -> Result<Option<f64>, OrchestratorError> {
    let probe: Vec<Subject> = record
        .probe
        .subjects
        .iter()
        .filter_map(|id| early_stopping.iter().find(|s| &s.subject_id == id).cloned())
        .collect();
    if probe.len() != record.probe.subjects.len() {
        return Ok(None);
    }
    let (codes, horizons) = (&record.config.codes, &record.probe.horizons);
    let raw = max_abs_difference(
        &probe_values(served, &probe, codes, horizons).map_err(lift)?,
        &record.probe.values,
    );
    let calibrated = max_abs_difference_optional(
        &probe_calibrated(served, &probe, codes, horizons).map_err(lift)?,
        &record.probe.calibrated,
    );
    Ok((!raw.is_nan() && !calibrated.is_nan()).then(|| raw.max(calibrated)))
}

/// Measures the serving of `record`'s packed `bundle`: unpacked into a fresh
/// `dest`, loaded by plain brain and held against `scored` (the model whose
/// numbers the evaluation reports) on `test`, and against the model as
/// trained on the probe units among `early_stopping`.
pub fn check_serving(
    record: &TimelineCandidate,
    bundle: &Path,
    dest: &Path,
    scored: &TimelineModel,
    test: &[Subject],
    early_stopping: &[Subject],
    horizons: &[f64],
) -> Result<ServingCheck, OrchestratorError> {
    let files = unpack_with_tar(bundle, dest)?;
    let mut packed = record.files.clone();
    packed.sort();
    if files != packed {
        return Err(OrchestratorError::Train(format!(
            "the unpacked file holds {files:?} but the candidate was packed with {packed:?}"
        )));
    }
    let served = TimelineModel::load(dest).map_err(|e| lift(e.into()))?;
    Ok(ServingCheck {
        files,
        probe_max_abs_diff: probe_difference(record, &served, early_stopping)?,
        measured: measure_serving(scored, &served, test, &ServingSpec::new(horizons.to_vec()))
            .map_err(lift)?,
    })
}

impl ServingCheck {
    /// Files every measured number under the name its requirement reads. A
    /// difference that could not be measured has no entry. Identity is the
    /// worse of the two measurements, so either failing fails it.
    pub fn add_to(&self, evidence: &mut Evidence) {
        let m = &self.measured;
        if let Some(identity) = self.probe_max_abs_diff.zip(m.identity_max_abs_diff) {
            evidence
                .values
                .insert(SERVE_MAX_ABS_DIFF.into(), identity.0.max(identity.1));
        }
        let mut put = |name: &str, value: Option<f64>| {
            if let Some(v) = value {
                evidence.values.insert(name.into(), v);
            }
        };
        put(SERVE_BATCH_MAX_ABS_DIFF, m.batch_max_abs_diff);
        put(SERVE_ABSTENTION_RATE, m.abstention_rate);
        put(
            SERVE_INVALID_PROBABILITIES,
            Some(m.invalid_probabilities as f64),
        );
        put(
            SERVE_NON_MONOTONE_CURVES,
            Some(m.non_monotone_curves as f64),
        );
    }
}
