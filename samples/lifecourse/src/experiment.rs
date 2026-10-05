// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One experiment: an arm trained on one set of subjects and scored on
//! another, every number traceable to the dataset, partition, arm and seed.
//!
//! Arms differ only in their inputs and their encoder; the outcome, the
//! training procedure and the evaluation are identical:
//!
//! | Arm | Encoder | Inputs |
//! |---|---|---|
//! | `age-sex` | additive | age and sex |
//! | `standard` | additive | the conventional risk factors ([`STANDARD_RISK_FACTORS`]) at the examination |
//! | `additive` | additive | everything, history included |
//! | `horizon` | set encoder | everything, history included |
//!
//! Training holds out a tenth of its subjects (by a hash of the subject id
//! and the seed) for early stopping; the test subjects are never seen.

use std::collections::BTreeSet;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use splinter_sdk::model::timeline::{Subject, TimelineModel, TimelineSpec};

use crate::build::CODES;
use crate::concepts::{AGE_SEX, STANDARD_RISK_FACTORS};

/// Hazard knots in years since the examination: finer early, to the longest
/// follow-up the linkage gives (about 20.75 years for 1999).
pub const KNOTS: [f32; 16] = [
    0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 21.0,
];
/// Tokens per subject; the builder writes fewer for every participant.
pub const MAX_TOKENS: u32 = 96;
/// Optimiser steps at most; early stopping ends most runs well before.
pub const STEPS: u32 = 6000;
/// Subjects per batch.
pub const BATCH: u32 = 256;

/// What is trained.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Arm {
    /// Additive model on age and sex.
    AgeSex,
    /// Additive model on the conventional risk factors.
    Standard,
    /// Additive model on every input.
    Additive,
    /// The set encoder on every input.
    Horizon,
}

impl Arm {
    /// Its name in results.
    pub fn name(self) -> &'static str {
        match self {
            Arm::AgeSex => "age-sex",
            Arm::Standard => "standard",
            Arm::Additive => "additive",
            Arm::Horizon => "horizon",
        }
    }

    /// The subject as this arm sees it: the restricted arms keep only their
    /// concepts measured at the examination, and no history.
    pub fn view(self, s: &Subject) -> Subject {
        let keep: Option<BTreeSet<&str>> = match self {
            Arm::AgeSex => Some(AGE_SEX.iter().copied().collect()),
            Arm::Standard => Some(STANDARD_RISK_FACTORS.iter().copied().collect()),
            Arm::Additive | Arm::Horizon => None,
        };
        let Some(keep) = keep else { return s.clone() };
        let mut v = s.clone();
        v.observations
            .retain(|o| o.t == s.entry && keep.contains(o.var.as_str()));
        v.events.retain(|e| e.t > s.entry); // outcomes only: no history
        v
    }
}

/// Training options for one run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Training {
    /// Seed of the initial weights, the batches and the early-stopping split.
    pub seed: u64,
    /// Optimiser steps at most.
    pub steps: u32,
}

fn early_stopping_share(id: &str, seed: u64) -> bool {
    let h = splinter_sdk::vocabulary::digest::Digest::of(format!("{seed}:{id}").as_bytes());
    // The first byte of the digest is uniform: one in ten held out.
    u8::from_str_radix(&h.hex()[..2], 16).unwrap_or(0) < 26
}

/// What a run produced besides the model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunInfo {
    /// Steps run.
    pub steps: u32,
    /// Event NLL on the early-stopping subjects.
    pub early_stopping_nll: f32,
    /// Trainable parameters.
    pub parameters: usize,
    /// Tokens left out because a subject had too many.
    pub truncated_tokens: usize,
    /// Subjects trained on and held out for early stopping.
    pub subjects: (usize, usize),
}

/// Train `arm` on `subjects`.
pub fn fit(
    arm: Arm,
    subjects: &[&Subject],
    training: &Training,
) -> Result<(TimelineModel, RunInfo)> {
    let views: Vec<Subject> = subjects.iter().map(|s| arm.view(s)).collect();
    let (held, train): (Vec<Subject>, Vec<Subject>) = views
        .into_iter()
        .partition(|s| early_stopping_share(&s.subject_id, training.seed));
    if train.is_empty() || held.is_empty() {
        return Err(anyhow!(
            "too few subjects to train ({} + {})",
            train.len(),
            held.len()
        ));
    }
    let spec = TimelineSpec::new(CODES, CODES)
        .additive(arm != Arm::Horizon)
        .knots(KNOTS.to_vec())
        .max_tokens(MAX_TOKENS)
        .batch(BATCH)
        .steps(training.steps)
        .seed(training.seed);
    let (model, report) = TimelineModel::train(&train, &held, &spec)?;
    let info = RunInfo {
        steps: report.steps,
        early_stopping_nll: report.held_out_event_nll,
        parameters: report.parameters,
        truncated_tokens: report.truncated_tokens,
        subjects: (train.len(), held.len()),
    };
    Ok((model, info))
}
