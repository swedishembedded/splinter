// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements post-hoc analysis of stored model
// examinations, for its clients. If your team needs expertise in redoing an
// analysis from what an exam kept, you can procure our services by sending
// an email to info@swedishembedded.com.

//! What the analyses done after an exam need: the records its report kept,
//! the tasks' wording from the stores, and the text a candidate was trained to
//! produce.

use std::path::Path;

use serde::Deserialize;
use splinter_core::digest::Digest;
use splinter_core::experience::PrivilegedKind;

use super::analysis::TaskRecord;
use super::audit::Wording;
use super::memorisation::Corpus;
use crate::reserve::touched_blobs;
use crate::train::load_candidate;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::{io, OrchestratorError};

/// The per-task records of the exam report in `file`: the report as
/// `exam --json` prints it, whether bare or as a recorded run's output.
pub fn read_records(file: &Path) -> Result<Vec<TaskRecord>, OrchestratorError> {
    let text = std::fs::read_to_string(file).map_err(io(file))?;
    let json_start = text.find('{').unwrap_or(0);
    let value: serde_json::Value =
        serde_json::from_str(&text[json_start..]).map_err(|source| OrchestratorError::Json {
            what: file.display().to_string(),
            source,
        })?;
    let ran = [&value["outputs"]["ran"], &value["ran"], &value]
        .into_iter()
        .find(|v| v["records"].is_array())
        .ok_or_else(|| {
            OrchestratorError::Refused(format!(
                "{} holds no exam report with records",
                file.display()
            ))
        })?;
    serde_json::from_value(ran["records"].clone()).map_err(|source| OrchestratorError::Json {
        what: format!("the records of {}", file.display()),
        source,
    })
}

/// The tasks' wording as the stores hold it.
pub struct StoredWording<'a>(pub &'a Context);

impl Wording for StoredWording<'_> {
    fn of(&self, address: &str) -> Option<(String, String)> {
        let task = self.0.tasks().get(&Digest::parse(address).ok()?).ok()?;
        let reference = task
            .privileged
            .iter()
            .find(|p| p.kind == PrivilegedKind::Reference)
            .map(|p| p.content.clone())
            .unwrap_or_default();
        Some((task.instruction.clone(), reference))
    }
}

/// The text `candidate` (and the releases it continues) were trained to
/// produce, as a corpus: the supervised turns of their datasets and the
/// source text the records were built from.
pub fn training_corpus(ctx: &Context, candidate: &str) -> Result<Corpus, OrchestratorError> {
    #[derive(Deserialize)]
    struct Line {
        messages: Vec<Message>,
    }
    #[derive(Deserialize)]
    struct Message {
        content: String,
        #[serde(default)]
        train: bool,
    }
    let trained = load_candidate(ctx, candidate)?;
    let datasets = super::datasets_trained_on(ctx, &trained)?;
    let mut texts: Vec<String> = Vec::new();
    for id in &datasets {
        let stored = ctx.datasets().get(id)?;
        let text = std::fs::read_to_string(&stored.path).map_err(io(&stored.path))?;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            if let Ok(line) = serde_json::from_str::<Line>(line) {
                texts.extend(
                    line.messages
                        .into_iter()
                        .filter(|m| m.train)
                        .map(|m| m.content),
                );
            }
        }
    }
    for blob in touched_blobs(ctx, &datasets)? {
        if let Ok(bytes) = ctx.sources().read_blob(&blob) {
            texts.push(String::from_utf8_lossy(&bytes).into_owned());
        }
    }
    Ok(Corpus::of(texts.iter().map(String::as_str)))
}
