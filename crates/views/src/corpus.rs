// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What views project from: experiences with their annotations, tasks no
//! one has solved yet, and sources.
//!
//! A corpus is a value, built by the caller from whichever experiences a
//! pipeline stage names (an experience set, say) - so a view's output is a
//! function of exactly what it was given. A relation that names an
//! experience outside the corpus is not followed into the store; the view
//! counts it as [`Exclusion::RelatedMissing`](crate::Exclusion).

use std::collections::HashMap;

use splinter_record::annotation::{
    decide, Annotation, AnnotationBody, Decision, Label, RelationKind,
};
use splinter_record::experience::{Experience, ExperienceId, Task};
use splinter_record::experiences::ExperienceStore;
use splinter_record::source::SourceId;

use crate::ViewError;

/// One experience and everything said about it.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    /// The experience's id.
    pub id: ExperienceId,
    /// The experience.
    pub experience: Experience,
    /// Its annotations, in append order; every one is about it.
    pub notes: Vec<Annotation>,
}

impl Entry {
    /// The decision its verdicts add up to.
    #[must_use]
    pub fn decision(&self) -> Option<Decision> {
        decide(&self.notes)
    }

    /// The experiences it is related to by `kind`, in annotation order.
    pub fn related(&self, kind: RelationKind) -> impl Iterator<Item = &ExperienceId> {
        self.notes.iter().filter_map(move |note| match &note.body {
            AnnotationBody::Relation { kind: k, other } if *k == kind => Some(other),
            _ => None,
        })
    }

    /// Whether step `step` carries `label`.
    #[must_use]
    pub fn labelled(&self, step: u64, label: Label) -> bool {
        self.notes.iter().any(|note| {
            matches!(&note.body, AnnotationBody::StepLabel { step: s, label: l, .. }
                if *s == step && *l == label)
        })
    }
}

/// Experiences, tasks and sources a view projects from.
#[derive(Clone, Debug, Default)]
pub struct Corpus {
    entries: Vec<Entry>,
    index: HashMap<ExperienceId, usize>,
    tasks: Vec<Task>,
    sources: Vec<SourceId>,
}

impl Corpus {
    /// An empty corpus.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The experiences `ids` from `store`, each with its readable
    /// annotations, in the order given.
    pub fn load(store: &ExperienceStore, ids: &[ExperienceId]) -> Result<Self, ViewError> {
        let mut corpus = Self::new();
        for id in ids {
            let experience = store.get(id)?;
            let notes = store.annotations(id)?.annotations;
            corpus.insert(experience, notes)?;
        }
        Ok(corpus)
    }

    /// Adds `experience` with `notes`, its annotations, and returns its id.
    /// Refuses an annotation of another experience, and an experience the
    /// corpus already holds.
    pub fn insert(
        &mut self,
        experience: Experience,
        notes: Vec<Annotation>,
    ) -> Result<ExperienceId, ViewError> {
        let id = experience.id()?;
        if let Some(foreign) = notes.iter().find(|n| n.experience != id) {
            return Err(ViewError::ForeignAnnotation {
                experience: id,
                annotation_of: foreign.experience.clone(),
            });
        }
        if self.index.contains_key(&id) {
            return Err(ViewError::DuplicateExperience(id));
        }
        self.index.insert(id.clone(), self.entries.len());
        self.entries.push(Entry {
            id: id.clone(),
            experience,
            notes,
        });
        Ok(id)
    }

    /// Adds a task as its generator emitted it, solved or not.
    pub fn add_task(&mut self, task: Task) {
        self.tasks.push(task);
    }

    /// Adds a source by its id; a view that reads sources resolves it
    /// through its source store.
    pub fn add_source(&mut self, source: SourceId) {
        self.sources.push(source);
    }

    /// The experiences, in the order they were added.
    #[must_use]
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// The experience `id`, when the corpus holds it.
    #[must_use]
    pub fn get(&self, id: &ExperienceId) -> Option<&Entry> {
        self.index.get(id).and_then(|&i| self.entries.get(i))
    }

    /// The tasks added with [`Corpus::add_task`], in order.
    #[must_use]
    pub fn tasks(&self) -> &[Task] {
        &self.tasks
    }

    /// The sources added with [`Corpus::add_source`], in order.
    #[must_use]
    pub fn sources(&self) -> &[SourceId] {
        &self.sources
    }
}
