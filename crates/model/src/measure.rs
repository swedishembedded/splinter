// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-set curation that sizes every
// record exactly as the trainer will, for its clients. If your team needs
// expertise in fitting conversations to a fine-tuning block, you can procure
// our services by sending an email to info@swedishembedded.com.

//! How long a chat training record is: the one count the dataset build
//! bounds records by and a run's row is sized by.
//!
//! The count is brain's: the records go to the service's `check_dataset`
//! action ([`crate::served_train::check_dataset`]), which reads and encodes
//! them with the code the chat fine-tune reads its dataset with, so the chat
//! template's markup, the shape a record is trained in and the rows of its
//! pictures are all in it. Nothing here counts tokens. A meter measures many
//! lines in one check ([`RecordMeter::measure`]) and keeps every count, so a
//! line asked for again costs nothing; a line it has not seen is checked
//! alone.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;

use crate::endpoint::BrainEndpoints;
use crate::selection::local_model_name;
use crate::served_train::{check_published, CheckSpec};
use crate::PolicyError;

/// Measures dataset lines (`generic-messages-v2` records) for one base, in the
/// mode a fine-tune of it is run in, through a brain service.
pub struct RecordMeter<'b> {
    brain: &'b BrainEndpoints,
    base: String,
    thinking: bool,
    pack_answers: bool,
    /// Each line measured, by its text: its tokens, or why the trainer
    /// refuses it.
    known: RefCell<HashMap<String, Result<usize, String>>>,
}

impl<'b> RecordMeter<'b> {
    /// The meter of the base checkpoint `model_dir`, as the service `brain`
    /// names it. `thinking` and `pack_answers` are the fine-tune's own
    /// ([`crate::train::FineTune`]): records that carry step reasoning train
    /// for thinking mode, and a dataset built packed keeps each record as one
    /// sequence.
    #[must_use]
    pub fn for_model(
        brain: &'b BrainEndpoints,
        model_dir: &Path,
        thinking: bool,
        pack_answers: bool,
    ) -> Self {
        Self {
            brain,
            base: local_model_name(model_dir),
            thinking,
            pack_answers,
            known: RefCell::new(HashMap::new()),
        }
    }

    /// Measures every line of `lines` in one check, so later [`Self::tokens`]
    /// calls for them are answered from what was measured.
    ///
    /// # Errors
    /// No artifact store is configured, or the service refused the check.
    pub fn measure(&self, lines: &[&str]) -> Result<(), PolicyError> {
        let unseen: Vec<&str> = {
            let known = self.known.borrow();
            let mut seen = std::collections::HashSet::new();
            lines
                .iter()
                .copied()
                .filter(|l| !known.contains_key(*l) && seen.insert(*l))
                .collect()
        };
        if unseen.is_empty() {
            return Ok(());
        }
        // One record per line, in order: line `n` of the file is unseen[n-1]
        // (no line holds a newline; a JSON record never does).
        let mut text = String::new();
        for line in &unseen {
            text.push_str(&line.replace('\n', " "));
            text.push('\n');
        }
        let digest = self
            .brain
            .artifact_store()?
            .publish_bytes(text.as_bytes())?;
        let report = check_published(
            self.brain,
            &digest,
            Path::new("the records measured"),
            &CheckSpec::chat(&self.base, self.thinking, self.pack_answers),
        )?;
        let mut known = self.known.borrow_mut();
        for length in &report.per_record {
            if let Some(line) = unseen.get(length.line.wrapping_sub(1)) {
                known.insert((*line).to_string(), Ok(length.tokens));
            }
        }
        for refusal in &report.refused {
            if let Some(line) = unseen.get(refusal.line.wrapping_sub(1)) {
                known.insert((*line).to_string(), Err(refusal.error.clone()));
            }
        }
        for line in unseen {
            known
                .entry(line.to_string())
                .or_insert_with(|| Err("the trainer did not read it".into()));
        }
        Ok(())
    }

    /// The tokens of the longest example the dataset line `line` is trained
    /// as, which the training row has to hold. The reason, when the trainer
    /// would not read the line, or the service could not be asked.
    pub fn tokens(&self, line: &str) -> Result<usize, String> {
        if let Some(known) = self.known.borrow().get(line) {
            return known.clone();
        }
        self.measure(&[line]).map_err(|e| e.to_string())?;
        self.known
            .borrow()
            .get(line)
            .cloned()
            .unwrap_or_else(|| Err("the trainer did not read it".into()))
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
