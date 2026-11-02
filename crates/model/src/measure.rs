// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training pipelines whose records are sized
// exactly as the trainer will size them, for its clients. If your team needs
// expertise in fitting conversations to a fine-tuning block, you can procure
// our services by sending an email to info@swedishembedded.com.

//! How long a chat training record is: the one count the dataset build
//! bounds records by and the trainer's pre-check refuses them by.
//!
//! The count is brain's ([`brain::ExampleMeter`]): the record is read and
//! encoded with the code the chat fine-tune reads its dataset with, so the
//! chat template's markup, the shape a record is trained in and the rows of
//! its pictures are all in it. Nothing here counts tokens itself.

use std::path::Path;

use crate::PolicyError;

/// Measures dataset lines (`generic-messages-v2` records) for one base, in the
/// mode a fine-tune of it is run in.
pub struct RecordMeter {
    meter: brain::ExampleMeter,
}

impl RecordMeter {
    /// The meter of the checkpoint in `model_dir`. `thinking` and
    /// `pack_answers` are the fine-tune's own ([`crate::train::FineTune`]):
    /// records that carry step reasoning train for thinking mode, and a
    /// dataset built packed keeps each record as one sequence.
    pub fn for_model(
        model_dir: &Path,
        thinking: bool,
        pack_answers: bool,
    ) -> Result<Self, PolicyError> {
        let meter =
            brain::ExampleMeter::for_model(model_dir, thinking, pack_answers).map_err(|e| {
                PolicyError::Load {
                    path: model_dir.to_path_buf(),
                    reason: e.to_string(),
                }
            })?;
        Ok(Self { meter })
    }

    /// The tokens of the longest example the dataset line `line` is trained
    /// as, which the training row has to hold. The reason, when the trainer
    /// would not read the line.
    pub fn tokens(&self, line: &str) -> Result<usize, String> {
        self.meter.tokens(line).map_err(|e| e.to_string())
    }
}

/// Whether the dataset line `line` carries the reasoning of a step
/// (`reasoning_content` on a message), which trains a model for thinking mode.
#[must_use]
pub fn carries_reasoning(line: &str) -> bool {
    // Cheap test first: most lines of a dataset without reasoning never parse.
    if !line.contains("\"reasoning_content\"") {
        return false;
    }
    serde_json::from_str::<serde_json::Value>(line)
        .ok()
        .is_some_and(|record| {
            record["messages"].as_array().is_some_and(|messages| {
                messages.iter().any(|m| {
                    m["reasoning_content"]
                        .as_str()
                        .is_some_and(|text| !text.trim().is_empty())
                })
            })
        })
}
