// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The denoise view: a denoise task's corrupted passage and its original,
//! with no solve needed.
//!
//! A denoise task (kind `splinter_lab::denoise::KIND`) carries its answer
//! as its reference, so the task alone is supervision. Candidates are the
//! corpus's tasks, then the tasks of its experiences, whatever their
//! verdicts; each task yields at most one record (a repeat counts as
//! [`Exclusion::Duplicate`]). A task with no single reference is left out.
//! The record is the student's turn (see [`Strip`], keyed by the task's id)
//! and the reference as the supervised reply; its metadata names the task
//! and every experience in the corpus that attempted it.

use std::collections::HashSet;

use splinter_core::digest::Digest;
use splinter_core::experience::{PrivilegedKind, Task};
use splinter_lab::denoise::KIND;

use crate::render::message;
use crate::{
    Corpus, Exclusion, Objective, Projection, Provenance, RecordBody, Strip, View, ViewError,
};

/// The name every denoise record carries.
const NAME: &str = "denoise";

/// Supervised fine-tuning on denoise tasks' references.
#[derive(Clone, Debug, Default)]
pub struct DenoiseView {
    strip: Strip,
}

impl DenoiseView {
    /// The view, showing the student only the instruction.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The same view under `strip`.
    #[must_use]
    pub fn with_strip(self, strip: Strip) -> Self {
        Self { strip }
    }

    fn record(&self, corpus: &Corpus, task: &Task) -> Result<(RecordBody, Provenance), Exclusion> {
        let mut references = task
            .privileged
            .iter()
            .filter(|p| p.kind == PrivilegedKind::Reference);
        let (Some(reference), None) = (references.next(), references.next()) else {
            return Err(Exclusion::NoReference);
        };
        let turn = self
            .strip
            .student_turn(&task.task.id, &task.instruction, &task.privileged)?;
        let messages = vec![
            message("user", &turn, false),
            message("assistant", &reference.content, true),
        ];
        let experiences = corpus
            .entries()
            .iter()
            .filter(|e| e.experience.task.id == task.task.id)
            .map(|e| e.id.clone())
            .collect();
        Ok((
            RecordBody::Chat { messages },
            Provenance {
                experiences,
                task: Some(task.task.id.clone()),
                sources: Vec::new(),
            },
        ))
    }
}

impl View for DenoiseView {
    fn name(&self) -> &str {
        NAME
    }

    fn objective(&self) -> Objective {
        Objective::Sft
    }

    fn project(&self, corpus: &Corpus) -> Result<Projection, ViewError> {
        let mut projection = Projection::new(NAME, Objective::Sft, Some(self.strip.clone()), None);
        let attempted = corpus.entries().iter().map(|e| e.experience.to_task());
        let mut seen: HashSet<Digest> = HashSet::new();
        for task in corpus.tasks().iter().cloned().chain(attempted) {
            if task.task.kind != KIND {
                continue;
            }
            if !seen.insert(task.task.id.clone()) {
                projection.exclude(Exclusion::Duplicate);
                continue;
            }
            projection.take(self.record(corpus, &task));
        }
        Ok(projection)
    }
}
