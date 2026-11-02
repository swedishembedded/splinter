// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! What the model backend trains, and the check of a dataset file against
//! its own parser.
//!
//! A view projects an objective's records and writes them in that
//! objective's line format; whether a trainer reads that format is the
//! backend's to say, so the answer lives here and not with the views.
//!
//! It also holds the length check of a chat dataset against the training row.

use std::path::{Path, PathBuf};

use splinter_data::{DatasetCheck, Format, Objective};

use crate::error::PolicyError;
use crate::measure::RecordMeter;
use crate::pictures::{holds_images, validate_pictured_dataset};
use crate::train::{validate_dataset, validate_preference_dataset};

/// The dataset formats a backend trains from a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingCapabilities {
    /// Supervised fine-tuning on chat records, which also trains
    /// classification rendered as SFT.
    pub chat_sft: bool,
    /// Preference optimisation on (chosen, rejected) pairs.
    pub preference: bool,
    /// Time-to-event training on subject timelines (`timeline-v1`).
    pub timeline: bool,
}

impl TrainingCapabilities {
    /// What brain's public SDK trains from a dataset file: chat fine-tuning,
    /// preference fine-tuning and subject timelines. It has no trainer that
    /// reads contrastive triples for the policy model, rewarded trajectories
    /// or a raw text corpus.
    pub const BRAIN: Self = Self {
        chat_sft: true,
        preference: true,
        timeline: true,
    };

    /// Whether a file in `format` can be trained from.
    #[must_use]
    pub fn supports(self, format: Format) -> bool {
        match format {
            Format::GenericMessagesV2 => self.chat_sft,
            Format::GenericPreferenceV1 => self.preference,
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

/// brain's own parsers as the check of a dataset file: the wire schema and
/// supervision boundaries of a chat file, which must supervise something,
/// and a preference file, which must be read whole. A chat file with pictures
/// is read as the trainer of a base that reads them would, except that no
/// base is chosen yet: every picture is taken for an inline `data:` URL and
/// not decoded. The base is checked when it is trained
/// ([`crate::pictures::validate_dataset_on`]).
#[derive(Clone, Copy, Debug, Default)]
pub struct BrainDatasetCheck;

impl DatasetCheck for BrainDatasetCheck {
    fn check(&self, format: Format, pending: &Path, records: usize) -> Result<(), String> {
        match format {
            Format::GenericMessagesV2 if holds_images(pending).map_err(|e| e.to_string())? => {
                validate_pictured_dataset(pending)
            }
            Format::GenericMessagesV2 => {
                let summary = validate_dataset(pending).map_err(|e| format!("{e:#}"))?;
                if summary.trained_messages == 0 {
                    return Err("no message is supervised".into());
                }
                Ok(())
            }
            Format::GenericPreferenceV1 => {
                let summary = validate_preference_dataset(pending).map_err(|e| format!("{e:#}"))?;
                if summary.pairs == records {
                    Ok(())
                } else {
                    Err(format!(
                        "{records} pair(s) were written but brain's parser read {}",
                        summary.pairs
                    ))
                }
            }
            Format::SplinterExportV1 => Ok(()),
            Format::TimelineV1 => {
                let subjects = crate::timeline::read_jsonl(pending).map_err(|e| e.to_string())?;
                if subjects.len() == records {
                    Ok(())
                } else {
                    Err(format!(
                        "{records} subject(s) were written but brain's parser read {}",
                        subjects.len()
                    ))
                }
            }
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
/// A record's length is [`RecordMeter::tokens`]: what the fine-tune of the
/// checkpoint in `model_dir` trains it as, in the mode `thinking` and
/// `pack_answers` name (the trainer's, [`crate::train::FineTune`]). A record
/// the trainer would not read is an error naming its line, as the trainer
/// refuses it. Needs no device.
pub fn overlong_records(
    dataset: &Path,
    model_dir: &Path,
    thinking: bool,
    pack_answers: bool,
    max_block: u32,
) -> Result<Vec<OverlongRecord>, PolicyError> {
    let unreadable = |reason: String| PolicyError::Dataset {
        path: dataset.to_path_buf(),
        reason,
    };
    let meter = RecordMeter::for_model(model_dir, thinking, pack_answers)?;
    let text = std::fs::read_to_string(dataset).map_err(|e| unreadable(e.to_string()))?;
    let mut overlong = Vec::new();
    for (index, line) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let line_no = index + 1;
        let tokens = meter
            .tokens(line)
            .map_err(|reason| unreadable(format!("line {line_no}: {reason}")))?;
        if tokens > max_block as usize {
            let origin = serde_json::from_str(line).ok().and_then(|r| origin_of(&r));
            overlong.push(OverlongRecord {
                dataset: dataset.to_path_buf(),
                line: line_no,
                origin,
                tokens,
            });
        }
    }
    Ok(overlong)
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
