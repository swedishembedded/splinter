// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! What the model backend trains.
//!
//! A view projects an objective's records and writes them in that
//! objective's line format; whether a trainer reads that format is the
//! backend's to say, so the answer lives here and not with the views.
//!
//! It also holds the length check of a chat dataset against the training row,
//! made by the service that trains it. A dataset file is read by the trainer
//! itself before a run (`check_dataset`), so nothing here parses one.

use std::path::{Path, PathBuf};

use splinter_data::{Format, Objective};

use crate::endpoint::BrainEndpoints;
use crate::error::PolicyError;
use crate::selection::local_model_name;
use crate::served_train::{check_dataset, CheckSpec};

/// The dataset formats a backend trains from a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingCapabilities {
    /// Supervised fine-tuning on chat records, which also trains
    /// classification rendered as SFT.
    pub chat_sft: bool,
    /// Preference optimisation on (chosen, rejected) pairs.
    pub preference: bool,
    /// KTO on judged completions.
    pub feedback: bool,
    /// Time-to-event training on subject timelines (`timeline-v1`).
    pub timeline: bool,
}

impl TrainingCapabilities {
    /// What brain's public SDK trains from a dataset file: chat fine-tuning
    /// (which also trains a raw text corpus, written as one supervised
    /// assistant turn per text), preference fine-tuning and subject
    /// timelines, KTO on judged completions. It has no trainer that reads contrastive triples for the
    /// policy model or rewarded trajectories.
    pub const BRAIN: Self = Self {
        chat_sft: true,
        preference: true,
        feedback: true,
        timeline: true,
    };

    /// Whether a file in `format` can be trained from.
    #[must_use]
    pub fn supports(self, format: Format) -> bool {
        match format {
            Format::GenericMessagesV2 => self.chat_sft,
            Format::GenericPreferenceV1 => self.preference,
            Format::GenericFeedbackV1 => self.feedback,
            Format::SplinterExportV1 => false,
            Format::TimelineV1 => self.timeline,
        }
    }

    /// The format `objective` is trained from, or why it cannot be.
    pub fn require(self, objective: Objective) -> Result<Format, PolicyError> {
        let format = objective.line_format();
        if self.supports(format) {
            Ok(format)
        } else {
            Err(PolicyError::ObjectiveNotTrainable { objective })
        }
    }
}

/// A chat record too long for the training row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OverlongRecord {
    /// The dataset file it is in.
    pub dataset: PathBuf,
    /// Its 1-based line in the file.
    pub line: usize,
    /// What its metadata names it by (its first experience, else its task),
    /// when it has either.
    pub origin: Option<String>,
    /// Its length in tokens.
    pub tokens: usize,
}

impl std::fmt::Display for OverlongRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.dataset.display(), self.line)?;
        if let Some(origin) = &self.origin {
            write!(f, " ({origin})")?;
        }
        write!(f, " is {} tokens", self.tokens)
    }
}

/// The records of `dataset` longer than `max_block` tokens, in file order.
///
/// A record's length is the service's count ([`check_dataset`]): what the
/// fine-tune of the base checkpoint `model_dir` trains it as, in the mode
/// `thinking` and `pack_answers` name (the trainer's,
/// [`crate::train::FineTune`]). A record the trainer would not read is an
/// error naming its line, as the trainer refuses it.
///
/// # Errors
/// A record is refused, or the service could not be asked.
pub fn overlong_records(
    brain: &BrainEndpoints,
    dataset: &Path,
    model_dir: &Path,
    thinking: bool,
    pack_answers: bool,
    max_block: u32,
) -> Result<Vec<OverlongRecord>, PolicyError> {
    let base = local_model_name(model_dir);
    let report = check_dataset(
        brain,
        dataset,
        &CheckSpec::chat(&base, thinking, pack_answers),
    )?;
    report.refuse_invalid(dataset)?;
    let long: Vec<_> = report
        .per_record
        .iter()
        .filter(|r| r.tokens > max_block as usize)
        .collect();
    if long.is_empty() {
        return Ok(Vec::new());
    }
    let text = std::fs::read_to_string(dataset).map_err(|e| PolicyError::Dataset {
        path: dataset.to_path_buf(),
        reason: e.to_string(),
    })?;
    let lines: Vec<&str> = text.lines().collect();
    Ok(long
        .into_iter()
        .map(|r| OverlongRecord {
            dataset: dataset.to_path_buf(),
            line: r.line,
            origin: lines
                .get(r.line.wrapping_sub(1))
                .and_then(|l| serde_json::from_str(l).ok())
                .and_then(|record| origin_of(&record)),
            tokens: r.tokens,
        })
        .collect())
}

/// What a record's metadata names it by: its first experience, else its task.
fn origin_of(record: &serde_json::Value) -> Option<String> {
    let metadata = record.get("metadata")?;
    let first = metadata
        .get("experiences")
        .and_then(serde_json::Value::as_array)
        .and_then(|e| e.first());
    first
        .or_else(|| metadata.get("task"))
        .map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_owned))
}
