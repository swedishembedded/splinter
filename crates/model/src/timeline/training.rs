// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible training of time-to-event risk
// models on longitudinal records, for its clients. If your team needs
// expertise in training risk models whose configuration, data membership and
// seed can be audited, you can procure our services by sending an email to
// info@swedishembedded.com.

//! One training run of brain's timeline model, as a value.
//!
//! [`TimelineTraining`] is everything a run is told apart from its data: the
//! outcome codes and which of them are absorbing, the hazard knots (derived
//! from the training outcome times when the request gives none, so the
//! configuration that was actually trained is a recorded value, never an
//! implicit default), the optional next-event group, forecast head and visit
//! state, the schedule and the seed. [`train_timeline`] fits brain's model on
//! the training subjects only (the vocabulary and the value normalisation are
//! fitted inside brain on exactly the slice it is handed) and stops early on
//! the held-out subjects' event likelihood. No training algorithm is here.

use serde::{Deserialize, Serialize};
use splinter_core::digest::{canonical_json, Digest};

use super::{
    observed, Backbone, Gap, Subject, TimelineModel, TimelineReport, TimelineSpec,
    TimelineSpec as Spec,
};
use brain::timeline::CalibrationSpec;

/// Hazard pieces a derived knot grid has when the request names no count.
pub const DEFAULT_PIECES: usize = 8;
/// The widest ratio between the last knot and the first nonzero one: a day
/// against a few decades, so a piece is never narrower than a dataset can
/// resolve.
pub const MAX_KNOT_SPAN: f64 = 5000.0;
/// The share of outcome times the first knot is drawn from.
const FIRST_KNOT_QUANTILE: f64 = 0.01;
/// The fewest positive outcome times a knot grid can be derived from.
pub const MIN_TIMES_FOR_KNOTS: usize = 10;

/// Why a timeline run could not be set up or trained.
#[derive(Debug, thiserror::Error)]
pub enum TimelineError {
    /// The request is not one a run can be made from.
    #[error("timeline training request: {0}")]
    Request(String),
    /// brain refused or failed the run.
    #[error("brain timeline model: {0}")]
    Brain(String),
    /// A file could not be read or written.
    #[error("{path}: {reason}")]
    Io {
        /// The file or directory.
        path: std::path::PathBuf,
        /// What failed.
        reason: String,
    },
}

impl From<brain::Error> for TimelineError {
    fn from(e: brain::Error) -> Self {
        Self::Brain(e.to_string())
    }
}

/// A group of events modelled for which comes first after the prediction
/// time: a self-supervised signal on every history.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NextEvents {
    /// The event codes in the group.
    pub codes: Vec<String>,
    /// Its weight against the outcome codes.
    pub weight: f32,
}

/// What one training run is told, apart from its data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimelineTraining {
    /// The outcome codes predicted, in the model's order.
    pub codes: Vec<String>,
    /// Which of them end follow-up for all of them.
    pub absorbing: Vec<String>,
    /// Hazard piece boundaries over time since prediction, starting at 0;
    /// derived from the training outcome times when `None` (see
    /// [`derive_knots`]). A resolved configuration always names them.
    pub knots: Option<Vec<f32>>,
    /// The next-event group, if any.
    pub next_events: Option<NextEvents>,
    /// A forecast head on up to this many future measurements per subject,
    /// weighted as given.
    pub forecasts: Option<(u32, f32)>,
    /// Carry a continuous-time state over this many visits.
    pub visits: Option<u32>,
    /// What carries the visits to the prediction time.
    pub backbone: Option<Backbone>,
    /// Train the additive proportional-hazards baseline instead of the set
    /// encoder: the age-and-covariates reference a candidate is compared to.
    pub additive: bool,
    /// Tokens per subject, including the summary token; brain's default
    /// when `None`.
    pub max_tokens: Option<u32>,
    /// Seed of the initial weights, the batches and the masks.
    pub seed: u64,
    /// Optimiser steps at most.
    pub steps: u32,
    /// Subjects per batch.
    pub batch: u32,
    /// Peak learning rate.
    pub lr: f32,
    /// Held-out evaluations without improvement before training stops.
    pub patience: u32,
    /// Steps between held-out evaluations.
    pub eval_interval: u32,
    /// Share of numeric values hidden for the value objective.
    pub mask_rate: f64,
}

impl TimelineTraining {
    /// A run predicting `codes`, `absorbing` of them ending follow-up, with
    /// brain's schedule and seed 1.
    pub fn new<C: Into<String>, A: Into<String>>(
        codes: impl IntoIterator<Item = C>,
        absorbing: impl IntoIterator<Item = A>,
    ) -> Self {
        Self {
            codes: codes.into_iter().map(Into::into).collect(),
            absorbing: absorbing.into_iter().map(Into::into).collect(),
            knots: None,
            next_events: None,
            forecasts: None,
            visits: None,
            backbone: None,
            additive: false,
            max_tokens: None,
            seed: 1,
            steps: brain::timeline::DEFAULT_STEPS,
            batch: brain::timeline::DEFAULT_BATCH,
            lr: brain::timeline::DEFAULT_LR,
            patience: brain::timeline::DEFAULT_PATIENCE,
            eval_interval: brain::timeline::DEFAULT_EVAL_INTERVAL,
            mask_rate: brain::timeline::DEFAULT_MASK_RATE,
        }
    }

    /// The same run with its knots fixed: the explicit ones when the request
    /// gave them, else [`derive_knots`] over the training subjects.
    pub fn resolve(&self, train: &[Subject]) -> Result<Self, TimelineError> {
        let mut resolved = self.clone();
        if resolved.knots.is_none() {
            resolved.knots = Some(derive_knots(train, &self.codes, DEFAULT_PIECES)?);
        }
        Ok(resolved)
    }

    /// The address of the canonical form of this configuration: what a
    /// release records as the training configuration it was made under.
    pub fn digest(&self) -> Result<Digest, TimelineError> {
        canonical_json(self)
            .map(|bytes| Digest::of(&bytes))
            .map_err(|e| {
                TimelineError::Request(format!("the configuration has no canonical form: {e}"))
            })
    }

    fn spec(&self) -> Spec {
        let mut spec = TimelineSpec::new(self.codes.clone(), self.absorbing.clone())
            .additive(self.additive)
            .batch(self.batch)
            .steps(self.steps)
            .lr(self.lr)
            .patience(self.patience)
            .eval_interval(self.eval_interval)
            .mask_rate(self.mask_rate)
            .seed(self.seed);
        if let Some(knots) = &self.knots {
            spec = spec.knots(knots.clone());
        }
        if let Some(n) = self.max_tokens {
            spec = spec.max_tokens(n);
        }
        if let Some(group) = &self.next_events {
            spec = spec.next_events(group.codes.clone(), group.weight);
        }
        if let Some((per_subject, weight)) = self.forecasts {
            spec = spec.forecasts(per_subject, weight);
        }
        if let Some(visits) = self.visits {
            spec = spec.visits(visits);
        }
        if let Some(backbone) = self.backbone {
            spec = spec.backbone(backbone);
        }
        spec
    }

    fn check(&self) -> Result<(), TimelineError> {
        let bad = |why: String| Err(TimelineError::Request(why));
        if self.codes.is_empty() {
            return bad("no outcome code is named".into());
        }
        if let Some(code) = self.absorbing.iter().find(|a| !self.codes.contains(a)) {
            return bad(format!(
                "absorbing code {code:?} is not one of the outcome codes {:?}",
                self.codes
            ));
        }
        if self.steps == 0 || self.batch == 0 {
            return bad(format!(
                "steps ({}) and batch ({}) must both be positive",
                self.steps, self.batch
            ));
        }
        if !(0.0..=1.0).contains(&self.mask_rate) {
            return bad(format!("mask_rate {} is outside [0, 1]", self.mask_rate));
        }
        Ok(())
    }
}

/// The hazard knots for a training set: 0, then `pieces` boundaries spaced
/// geometrically from a low quantile of the observed outcome times (time to
/// the first outcome or to the end of follow-up) to the longest of them, so a
/// dataset of days and a dataset of decades both get pieces where its events
/// are. The first boundary is never closer to zero than `1 / MAX_KNOT_SPAN`
/// of the last, and the last is the longest follow-up seen: the model says
/// nothing beyond what the training data observed.
pub fn derive_knots(
    train: &[Subject],
    codes: &[String],
    pieces: usize,
) -> Result<Vec<f32>, TimelineError> {
    if pieces < 2 {
        return Err(TimelineError::Request(format!(
            "{pieces} hazard piece(s): at least 2 are needed"
        )));
    }
    let names: Vec<&str> = codes.iter().map(String::as_str).collect();
    let mut times: Vec<f64> = observed(train, &names)
        .into_iter()
        .map(|o| o.time)
        .filter(|t| t.is_finite() && *t > 0.0)
        .collect();
    if times.len() < MIN_TIMES_FOR_KNOTS {
        return Err(TimelineError::Request(format!(
            "{} subject(s) have a positive follow-up time; at least {MIN_TIMES_FOR_KNOTS} are \
             needed to derive hazard knots (name them in the request instead)",
            times.len()
        )));
    }
    times.sort_by(f64::total_cmp);
    let last = times[times.len() - 1];
    let quantile = times[((times.len() - 1) as f64 * FIRST_KNOT_QUANTILE).round() as usize];
    let first = quantile.max(last / MAX_KNOT_SPAN);
    if first >= last {
        return Err(TimelineError::Request(format!(
            "every follow-up time is about {last}: there is no spread to place {pieces} pieces on"
        )));
    }
    let ratio = (last / first).powf(1.0 / (pieces - 1) as f64);
    let mut knots = vec![0.0f32];
    for i in 0..pieces {
        let knot = if i + 1 == pieces {
            last
        } else {
            first * ratio.powi(i as i32)
        };
        knots.push(knot as f32);
    }
    if knots.windows(2).any(|w| w[1] <= w[0]) {
        return Err(TimelineError::Request(format!(
            "derived knots {knots:?} do not increase"
        )));
    }
    Ok(knots)
}

/// A trained model and what training did.
pub struct TrainedTimeline {
    /// The model kept (the best on the held-out likelihood).
    pub model: TimelineModel,
    /// brain's report of the run.
    pub report: TimelineReport,
    /// The configuration that was trained, with its knots fixed.
    pub config: TimelineTraining,
}

/// Trains one model: the configuration is resolved on `train` (knots), the
/// vocabulary and normalisation are fitted by brain on `train` alone, and
/// early stopping reads `held_out`. Refused when either is empty, when the
/// two share a subject, or when the configuration is not one a run can be
/// made from.
pub fn train_timeline(
    train: &[Subject],
    held_out: &[Subject],
    config: &TimelineTraining,
) -> Result<TrainedTimeline, TimelineError> {
    config.check()?;
    let shared: Vec<&str> = {
        let ids: std::collections::HashSet<&str> =
            train.iter().map(|s| s.subject_id.as_str()).collect();
        held_out
            .iter()
            .map(|s| s.subject_id.as_str())
            .filter(|id| ids.contains(id))
            .collect()
    };
    if !shared.is_empty() {
        return Err(TimelineError::Request(format!(
            "{} subject(s) are in both the training and the early-stopping set (e.g. {:?})",
            shared.len(),
            shared[0]
        )));
    }
    let config = config.resolve(train)?;
    let (model, report) = TimelineModel::train(train, held_out, &config.spec())?;
    Ok(TrainedTimeline {
        model,
        report,
        config,
    })
}

/// Which risks a trained model is calibrated for, on which events.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationPlan {
    /// The horizons after entry whose risk is calibrated, each inside the
    /// model's last knot.
    pub horizons: Vec<f64>,
    /// The fewest validation events a code needs by a horizon to be
    /// calibrated there (and as many still event-free); brain's own minimum
    /// when `None`.
    pub min_events: Option<usize>,
}

/// What fitting a calibration did.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationOutcome {
    /// The plan it followed.
    pub plan: CalibrationPlan,
    /// Subjects the calibrators were fitted on.
    pub units: usize,
    /// The event minimum that applied.
    pub min_events: usize,
    /// The (code, horizon) pairs calibrated.
    pub calibrated: Vec<(String, f64)>,
    /// The pairs the validation data could not support: not calibrated, and
    /// not served as if they were.
    pub uncalibrated: Vec<Gap>,
}

/// Fits `model`'s calibration on `validation` - subjects it was neither
/// trained nor early-stopped on, and never the test units - and keeps it with
/// the model, so saving writes it beside the weights. Refused when there are
/// no validation subjects or no horizon.
pub fn calibrate_timeline(
    model: &mut TimelineModel,
    validation: &[Subject],
    plan: &CalibrationPlan,
) -> Result<CalibrationOutcome, TimelineError> {
    if validation.is_empty() || plan.horizons.is_empty() {
        return Err(TimelineError::Request(
            "calibration needs validation subjects and at least one horizon".into(),
        ));
    }
    let mut spec = CalibrationSpec::new(plan.horizons.iter().copied());
    if let Some(n) = plan.min_events {
        spec = spec.min_events(n);
    }
    let fitted = model.calibrate(validation, &spec)?;
    Ok(CalibrationOutcome {
        plan: plan.clone(),
        units: fitted.validation_subjects,
        min_events: fitted.min_events,
        calibrated: fitted
            .entries()
            .iter()
            .map(|e| (e.code.clone(), e.horizon))
            .collect(),
        uncalibrated: fitted.uncalibrated().to_vec(),
    })
}

/// The lock of the workspace this crate was built in, as it was at build time.
const LOCK: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.lock"));

/// The commit of brain this build was linked against, read from the lock the
/// workspace was built with; `None` when the lock names none.
#[must_use]
pub fn brain_revision() -> Option<&'static str> {
    let mut lines = LOCK.lines();
    while let Some(line) = lines.next() {
        if line != "name = \"brain\"" {
            continue;
        }
        for field in lines.by_ref() {
            if field.is_empty() {
                break;
            }
            if let Some(source) = field.strip_prefix("source = \"") {
                return source
                    .trim_end_matches('"')
                    .rsplit_once('#')
                    .map(|(_, rev)| rev);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timeline::synthetic;

    fn subjects(n: usize) -> Vec<Subject> {
        synthetic::population(n, 3).0
    }

    #[test]
    fn derived_knots_start_at_zero_end_at_the_longest_follow_up_and_are_log_spaced() {
        let train = subjects(400);
        let codes: Vec<String> = synthetic::CODES.iter().map(|c| (*c).to_string()).collect();
        let knots = derive_knots(&train, &codes, 6).unwrap();
        assert_eq!(knots.len(), 7);
        assert_eq!(knots[0], 0.0);
        let longest = train
            .iter()
            .map(|s| s.at_risk[0].to - s.entry)
            .fold(0.0f64, f64::max);
        assert!(
            (f64::from(knots[6]) - longest).abs() < 1e-3,
            "{knots:?} vs {longest}"
        );
        let ratios: Vec<f32> = knots[1..].windows(2).map(|w| w[1] / w[0]).collect();
        for r in &ratios[..ratios.len() - 1] {
            assert!((r - ratios[0]).abs() < 1e-3 * ratios[0], "{ratios:?}");
        }
        assert!(knots[1] >= knots[6] / MAX_KNOT_SPAN as f32);
    }

    #[test]
    fn knots_cannot_be_derived_from_too_little() {
        let codes = vec!["death:a".to_string()];
        assert!(derive_knots(&subjects(5), &codes, 6).is_err());
        assert!(derive_knots(&subjects(100), &codes, 1).is_err());
    }

    #[test]
    fn a_request_that_cannot_train_is_refused_by_name() {
        let train = subjects(50);
        let held = subjects(60)[50..].to_vec();
        let mut config = TimelineTraining::new(["death:a"], ["death:b"]);
        let why = train_timeline(&train, &held, &config)
            .err()
            .unwrap()
            .to_string();
        assert!(why.contains("death:b"), "{why}");
        config.absorbing.clear();
        let overlap = train_timeline(&train, &train[..5], &config)
            .err()
            .unwrap()
            .to_string();
        assert!(
            overlap.contains("both the training and the early-stopping"),
            "{overlap}"
        );
    }

    #[test]
    fn the_configuration_digest_is_stable_and_sees_every_field() {
        let a = TimelineTraining::new(["x"], ["x"]);
        let mut b = a.clone();
        assert_eq!(a.digest().unwrap(), b.digest().unwrap());
        b.seed += 1;
        assert_ne!(a.digest().unwrap(), b.digest().unwrap());
    }

    #[test]
    fn the_brain_revision_is_a_commit_hash_of_the_lock() {
        let rev = brain_revision().expect("the workspace lock pins brain");
        assert!(
            rev.len() >= 7 && rev.bytes().all(|b| b.is_ascii_hexdigit()),
            "{rev}"
        );
    }
}
