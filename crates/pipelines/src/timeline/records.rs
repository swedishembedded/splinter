// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements traceable training and evaluation records
// for risk models, from a checkpoint back to the participants' source files,
// for its clients. If your team needs expertise in auditable model
// provenance, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The record a timeline candidate leaves behind.
//!
//! It is a document in the experience database, kept by the digest of its
//! canonical form, so it cannot be edited after the fact: a candidate is
//! immutable, its id names its content, and training again never replaces
//! one. A candidate's record is what the release's lineage is read from: the
//! dataset snapshots, the episodes and the source files its training units
//! came from, the configuration it was trained under, the split that cut the
//! data, the proof that statistics were fitted on training units only, the
//! seed, and the commits of brain and Splinter.

use serde::{Deserialize, Serialize};
use splinter_core::dataset::DatasetId;
use splinter_core::digest::Digest;
use splinter_core::terms::Terms;
use splinter_data::split::FitCertificate;
use splinter_data::timeline_dataset::SourceFile;
use splinter_model::timeline::{CalibrationOutcome, TimelineTraining};

/// The `format` of a candidate record.
pub const CANDIDATE_FORMAT: &str = "splinter-timeline-candidate-v1";
/// The architecture a released timeline checkpoint names: brain's.
pub const ARCHITECTURE: &str = "horizon";
/// The document class of candidates.
pub const CANDIDATE_CLASS: &str = "timeline_candidate";

/// What brain's training reported.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainingOutcome {
    /// Optimiser steps run.
    pub steps: u32,
    /// Training loss before the first step.
    pub initial_loss: f32,
    /// Training loss at the last step.
    pub final_loss: Option<f32>,
    /// Weighted event negative log-likelihood on the early-stopping units.
    pub held_out_event_nll: f32,
    /// Trainable parameters.
    pub parameters: usize,
    /// Known tokens dropped because a subject had too many.
    pub truncated_tokens: usize,
}

/// What a candidate's file must still predict when it is unpacked and loaded
/// by plain brain: the predictions of the model as trained, on some units it
/// was not fitted on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Probe {
    /// The early-stopping units the probe was taken on, in order.
    pub subjects: Vec<String>,
    /// The horizons.
    pub horizons: Vec<f64>,
    /// Cumulative incidence by subject, outcome code, horizon.
    pub values: Vec<f64>,
    /// The calibrated risk in the same order; `None` where the model has none
    /// at that code and horizon. Empty for a model with no calibration.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub calibrated: Vec<Option<f64>>,
}

/// The calibration kept with a candidate, and what it was fitted on.
///
/// The validation part of the split is divided by participant group into the
/// units early stopping reads and the units the calibration is fitted on, so
/// the calibrators never see data the model chose its weights on; neither
/// ever sees a test unit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationRecord {
    /// The SHA-256 of the `calibration.json` beside the weights, which the
    /// release manifest names.
    pub digest: Digest,
    /// What fitting did: the plan, the units, the pairs calibrated and the
    /// pairs left uncalibrated for want of events.
    pub outcome: CalibrationOutcome,
    /// Validation units early stopping read.
    pub early_stopping_units: usize,
    /// The address of the sorted ids of the units the calibration was fitted
    /// on.
    pub units_digest: Digest,
}

/// A trained candidate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimelineCandidate {
    /// [`CANDIDATE_FORMAT`].
    pub format: String,
    /// The configuration that was trained, knots fixed.
    pub config: TimelineTraining,
    /// The address of that configuration.
    pub config_digest: Digest,
    /// The seed it was trained with.
    pub seed: u64,
    /// The dataset the model was fitted on.
    pub train: DatasetId,
    /// The dataset early stopping read.
    pub held_out: DatasetId,
    /// The address of the split both are parts of.
    pub split: Digest,
    /// Proof that the vocabulary and normalisation were fitted on training
    /// units only.
    pub fit: FitCertificate,
    /// The episodes the training units were projected from, in file order.
    pub episodes: Vec<Digest>,
    /// The source files those episodes came from.
    pub sources: Vec<SourceFile>,
    /// The terms the training data came under.
    pub terms: Option<Terms>,
    /// The packed checkpoint's artifact address.
    pub checkpoint: Digest,
    /// The SHA-256 of the packed checkpoint, which a release names.
    pub checkpoint_sha256: Digest,
    /// The files the checkpoint holds.
    pub files: Vec<String>,
    /// What training did.
    pub outcome: TrainingOutcome,
    /// The predictions the packed file is held to.
    pub probe: Probe,
    /// The calibration packed with the weights; `None` for a model trained
    /// without one (a baseline, say), which is served and judged raw.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<CalibrationRecord>,
    /// The largest difference between the model as trained and the same
    /// model loaded from the packed file, on the probe: what the round trip
    /// through the file cost.
    pub round_trip_max_abs_diff: f64,
    /// The commit of brain that trained it.
    pub brain_commit: Option<String>,
    /// The commit of Splinter that trained it.
    pub splinter_commit: Option<String>,
}

impl TimelineCandidate {
    /// The candidate's address.
    pub fn digest(&self) -> Result<Digest, serde_json::Error> {
        splinter_core::digest::canonical_json(self).map(|b| Digest::of(&b))
    }

    /// The id a release knows it by: a prefix of its address.
    pub fn id(&self) -> Result<String, serde_json::Error> {
        Ok(format!("tl-{}", &self.digest()?.hex()[..16]))
    }
}
