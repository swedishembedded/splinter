// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curricula that track what a learner has
// mastered, concept by concept, for its clients. If your team needs
// expertise in curriculum design or knowledge tracing, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The concepts a task exercises.
//!
//! The rule, first match wins:
//!
//! 1. **Declared.** A task that declares concepts
//!    ([`Task::concepts`](splinter_core::experience::Task)) exercises
//!    exactly those.
//! 2. **Sections.** Otherwise each evidence span that names its source part
//!    exercises one (source, section) pair: the part is split by the one
//!    sectioner ([`crate::sections`]) at its media type, and the span
//!    belongs to the last section starting at or before its first byte (the
//!    first section when it starts before any). A part with no section is
//!    one concept, section 0.
//! 3. **Kind.** A task with neither (no evidence naming a part) exercises
//!    its kind as a whole.
//!
//! A task's concepts are distinct and sorted. The resolver reads each part
//! once and keeps its section starts, so resolving many tasks over the same
//! sources reads each part once.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use splinter_core::experience::{Span, Task};
use splinter_core::source::SourceId;
use splinter_record::error::StoreError;
use splinter_record::sources::SourceStore;

use crate::sections::sections;

/// One concept a task exercises; see the module documentation.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "concept", rename_all = "snake_case")]
pub enum Concept {
    /// A concept the task declared by name.
    Declared {
        /// Its name.
        name: String,
    },
    /// A section of a source part.
    Section(SectionRef),
    /// A task kind as a whole, for a task with no other concept.
    Kind {
        /// The kind.
        kind: String,
    },
}

impl std::fmt::Display for Concept {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Declared { name } => write!(f, "declared:{name}"),
            Self::Section(section) => write!(f, "section:{section}"),
            Self::Kind { kind } => write!(f, "kind:{kind}"),
        }
    }
}

/// One section of a source part, by position.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SectionRef {
    /// The source.
    pub source: SourceId,
    /// The part's name within it.
    pub part: String,
    /// The section's position among the part's sections, from zero.
    pub section: usize,
}

impl std::fmt::Display for SectionRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}#{}", self.source, self.part, self.section)
    }
}

/// Resolves tasks' concepts against one source store; see the module
/// documentation.
pub struct ConceptResolver {
    store: SourceStore,
    /// Each part's section starts, by (source, part); `None` for a part that
    /// is not text.
    starts: BTreeMap<(SourceId, String), Option<Vec<usize>>>,
}

impl ConceptResolver {
    /// A resolver reading parts from `store`.
    #[must_use]
    pub fn new(store: SourceStore) -> Self {
        Self {
            store,
            starts: BTreeMap::new(),
        }
    }

    /// The concepts `task` exercises, distinct and sorted. A span into a
    /// part that is not text names no section; with no other concept the
    /// task exercises its kind.
    pub fn concepts(&mut self, task: &Task) -> Result<Vec<Concept>, StoreError> {
        let mut concepts = BTreeSet::new();
        if !task.concepts.is_empty() {
            concepts.extend(
                task.concepts
                    .iter()
                    .map(|name| Concept::Declared { name: name.clone() }),
            );
        } else {
            concepts.extend(
                self.evidence_sections(task)?
                    .into_iter()
                    .map(Concept::Section),
            );
        }
        if concepts.is_empty() {
            concepts.insert(Concept::Kind {
                kind: task.task.kind.clone(),
            });
        }
        Ok(concepts.into_iter().collect())
    }

    /// The sections `task`'s evidence falls in, whatever concepts it
    /// declares: where new tasks for its concepts can be generated from.
    /// Distinct and sorted.
    pub fn evidence_sections(&mut self, task: &Task) -> Result<Vec<SectionRef>, StoreError> {
        let mut sections = BTreeSet::new();
        for span in &task.evidence {
            if let Some(section) = self.section_of(span)? {
                sections.insert(section);
            }
        }
        Ok(sections.into_iter().collect())
    }

    fn section_of(&mut self, span: &Span) -> Result<Option<SectionRef>, StoreError> {
        let Some(part) = &span.part else {
            return Ok(None);
        };
        let key = (part.source.clone(), part.name.clone());
        if !self.starts.contains_key(&key) {
            let starts = self.section_starts(&part.source, &part.name)?;
            self.starts.insert(key.clone(), starts);
        }
        let Some(Some(starts)) = self.starts.get(&key) else {
            return Ok(None);
        };
        let start = usize::try_from(span.start).unwrap_or(usize::MAX);
        let section = starts.iter().rposition(|s| *s <= start).unwrap_or_default();
        Ok(Some(SectionRef {
            source: part.source.clone(),
            part: part.name.clone(),
            section,
        }))
    }

    fn section_starts(
        &self,
        source: &SourceId,
        part: &str,
    ) -> Result<Option<Vec<usize>>, StoreError> {
        let stored = self.store.get_source(source)?;
        let found = stored.part(part).ok_or_else(|| StoreError::UnknownPart {
            source_id: source.clone(),
            part: part.to_string(),
        })?;
        let bytes = self.store.read_blob(&found.content)?;
        let Ok(text) = String::from_utf8(bytes) else {
            return Ok(None);
        };
        Ok(Some(
            sections(&text, &found.media_type)
                .into_iter()
                .map(|s| s.range.start)
                .collect(),
        ))
    }
}
