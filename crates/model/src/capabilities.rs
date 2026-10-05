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

use std::path::Path;

use splinter_data::{DatasetCheck, Format, Objective};

use crate::error::PolicyError;
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
/// and a preference file, which must be read whole.
#[derive(Clone, Copy, Debug, Default)]
pub struct BrainDatasetCheck;

impl DatasetCheck for BrainDatasetCheck {
    fn check(&self, format: Format, pending: &Path, records: usize) -> Result<(), String> {
        match format {
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
