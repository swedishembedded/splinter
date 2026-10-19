// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements coding agents that work inside a confined
// directory and leave a complete, replayable record of what they did, for
// its clients. If your team needs expertise in agent runtimes or in
// measuring what an agent really did, you can procure our services by

//! Knowledge a worker may retrieve from, frozen for the attempt.
//!
//! A task is sometimes better done with something to look up: a runbook, a
//! codebase's notes, what a person taught. That is retrieval, not training,
//! and it is measured as such: the worker reads a [`Knowledge`] frozen when it
//! starts (the memory cannot change while the attempt runs, and the attempt
//! cannot write to it), and the attempt records the snapshot it read and the
//! units it retrieved ([`MemoryUse`]). Comparing an arm with knowledge against
//! one without, at the same weights, is then a statement about the knowledge
//! ([`splinter_eval::arms`]).
//!
//! Documents are cut into units by sven's deterministic segmenter and found by
//! word. Searching by meaning needs vectors added to the units, which this
//! does not do yet.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use splinter_core::outcome::{MemoryUse, RetrievedUnit};
use sven_sdk::atif::{MemoryRecord, Trajectory};
use sven_sdk::memory::{
    AccessContext, IngestRequest, MemoryError, MemoryStore, Origin, Permission, Provenance,
    SourceId, SpaceId, TrainingUse,
};
use sven_sdk::sources::{segment, SourceKind};
use sven_sdk::MemoryScope;

/// A memory space an attempt may be given to read.
#[derive(Clone)]
pub struct Knowledge {
    store: Arc<dyn MemoryStore>,
    space: String,
}

impl Knowledge {
    /// The space `space` of `store`.
    #[must_use]
    pub fn new(store: Arc<dyn MemoryStore>, space: impl Into<String>) -> Self {
        Self {
            store,
            space: space.into(),
        }
    }

    fn author(&self) -> AccessContext {
        AccessContext::new("splinter-knowledge")
            .writing(SpaceId::new(&self.space), &[Permission::Ingest])
    }

    fn reader(&self) -> AccessContext {
        AccessContext::new("splinter-worker").reading(SpaceId::new(&self.space))
    }

    /// Adds (or replaces) the markdown document `source_id`.
    ///
    /// # Errors
    /// The store's refusal.
    pub async fn add_document(
        &self,
        source_id: &str,
        label: &str,
        text: &str,
    ) -> Result<(), MemoryError> {
        let request = IngestRequest {
            space: SpaceId::new(&self.space),
            source_id: SourceId::new(source_id),
            label: label.to_owned(),
            text: text.to_owned(),
            tags: BTreeSet::new(),
            provenance: Provenance {
                origin: Origin::User,
                captured_at: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0, |d| d.as_secs()),
                training_use: TrainingUse::Unspecified,
            },
            units: segment(text, SourceKind::Markdown),
        };
        self.store.ingest(&self.author(), request).await?;
        Ok(())
    }

    /// How many documents the space holds.
    ///
    /// # Errors
    /// The store's refusal.
    pub async fn source_count(&self) -> Result<usize, MemoryError> {
        Ok(self
            .store
            .list_sources(&self.reader(), &SpaceId::new(&self.space), None)
            .await?
            .len())
    }

    /// A read-only scope frozen on the content the space has now.
    pub(crate) async fn frozen_scope(&self) -> Result<MemoryScope, MemoryError> {
        MemoryScope::new(Arc::clone(&self.store), self.reader(), &self.space)
            .frozen()
            .await
    }
}

/// The memory `trajectory` read: the snapshot and the distinct units its
/// retrievals returned. `None` when it made no retrieval.
#[must_use]
pub fn memory_use(trajectory: &Trajectory) -> Option<MemoryUse> {
    let mut used = MemoryUse::default();
    let mut seen = BTreeSet::new();
    for step in &trajectory.steps {
        let Some(record) = step.extra.as_ref().and_then(MemoryRecord::from_extra) else {
            continue;
        };
        for retrieval in record.retrievals {
            used.retrievals += 1;
            if used.snapshot_id.is_none() {
                used.snapshot_id.clone_from(&retrieval.snapshot_id);
            }
            for result in retrieval.results {
                if seen.insert(result.unit_id.clone()) {
                    used.units.push(RetrievedUnit {
                        unit_id: result.unit_id,
                        revision: result.revision,
                        inserted: result.inserted,
                    });
                }
            }
        }
    }
    (used.retrievals > 0).then_some(used)
}
