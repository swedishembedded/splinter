// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The record of how a candidate was trained: what it trained on, what it
//! replayed, and what was measured on held-out records before and after.
//! Release manifests carry it.

use serde::{Deserialize, Serialize};

use crate::dataset::DatasetId;
use crate::digest::Digest;
use crate::release::ReleaseId;

/// One held-out score: teacher-forced loss and token accuracy over the
/// supervised positions of the held-out records.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HeldOutScore {
    /// Mean per-token cross-entropy over the supervised positions; lower is
    /// better. `None` when no position was scored.
    pub loss: Option<f32>,
    /// Fraction of supervised positions, 0.0-1.0, where the greedy argmax
    /// matched the true next token; `None` when no position was scored.
    pub token_accuracy: Option<f64>,
    /// Supervised token positions the two numbers above were computed over.
    pub positions: usize,
    /// Held-out records scored.
    pub records: usize,
    /// Held-out records skipped (too long for the scoring row, or not
    /// encodable).
    pub skipped: usize,
}

/// brain's preference score of a tuned adapter against its reference on a
/// set of pairs.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreferenceScore {
    /// Fraction, 0.0-1.0, of scored pairs where the tuned model prefers the
    /// chosen answer more than the reference does; `None` when no pair was
    /// scored.
    pub accuracy: Option<f32>,
    /// Mean over scored pairs of the reference-normalised log-probability
    /// margin of chosen over rejected, in nats (without `beta`); `None`
    /// when no pair was scored.
    pub mean_margin: Option<f32>,
    /// Pairs scored.
    pub pairs: usize,
    /// Pairs skipped because a candidate does not fit the model's context.
    pub skipped: usize,
}

/// How a candidate was trained, decided by its datasets' objective.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Regime {
    /// Supervised fine-tuning on chat records (`generic-messages-v2`); the
    /// regime of a record that names none.
    #[default]
    Sft,
    /// Direct preference optimisation on chosen/rejected pairs
    /// (`generic-preference-v1`).
    Dpo,
}

/// Where the replayed records came from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplaySource {
    /// The earlier release.
    pub release: ReleaseId,
    /// Its datasets.
    pub datasets: Vec<DatasetId>,
    /// Its records that were trained on, which the sample is drawn from.
    pub available: usize,
    /// Of those, the ones replayed.
    pub sampled: usize,
}

/// The earlier records replayed beside the new ones.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplaySample {
    /// The fraction of each release's training records drawn.
    pub fraction: f64,
    /// The draw's seed.
    pub seed: u64,
    /// Each earlier release trained on chat records, oldest first.
    pub sources: Vec<ReplaySource>,
    /// Records replayed in all.
    pub records: usize,
    /// The digest of the replay file; `None` when nothing was drawn.
    pub digest: Option<Digest>,
}

/// A preference candidate's measurements: brain's preference score of the
/// adapter against the reference it was trained against.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreferenceSummary {
    /// The DPO temperature it trained with.
    pub beta: f32,
    /// The digest of the adapter the reference carried (the one
    /// continued); `None` when the reference was the base alone.
    pub reference_adapter: Option<String>,
    /// On the pairs trained on; `None` when not measured.
    pub train_score: Option<PreferenceScore>,
    /// On the held-out pairs; `None` when not measured.
    pub held_out_score: Option<PreferenceScore>,
}

/// How a candidate was trained, as its release records it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainingSummary {
    /// The reference it was trained from.
    pub from: String,
    /// Optimizer steps.
    pub steps: u32,
    /// LoRA rank asked for.
    pub rank: u32,
    /// Records in the new datasets.
    pub records: usize,
    /// How it was trained.
    #[serde(default)]
    pub regime: Regime,
    /// The base on the held-out records; `None` when not measured (the
    /// preference regime measures preferences instead).
    #[serde(default)]
    pub base_score: Option<HeldOutScore>,
    /// The base with the adapter on the same records; `None` when not
    /// measured.
    #[serde(default)]
    pub tuned_score: Option<HeldOutScore>,
    /// The preference measurements; `None` for the supervised regime.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preference: Option<PreferenceSummary>,
    /// brain's own training record of the adapter.
    pub record: serde_json::Value,
}
