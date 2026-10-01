// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The logical query: predicates, no strategy.

use crate::id::{ContentId, RecordId};
use crate::model::{Epistemic, Record, RecordKind};

/// Which records are wanted. Every predicate narrows; kinds given more than
/// once widen only among themselves.
#[derive(Debug, Clone, Default)]
pub struct Query {
    pub(crate) kinds: Vec<RecordKind>,
    pub(crate) family: Option<ContentId>,
    pub(crate) attempt: Option<RecordId>,
    pub(crate) task_instance: Option<ContentId>,
    pub(crate) time: Option<(u64, u64)>,
    pub(crate) epistemic: Option<Epistemic>,
    pub(crate) limit: Option<usize>,
}

impl Query {
    /// Every record.
    pub fn all() -> Self {
        Self::default()
    }

    /// Only records of this kind (or of any kind named, if several).
    pub fn kind(mut self, kind: RecordKind) -> Self {
        self.kinds.push(kind);
        self
    }

    /// Only records of this episode family.
    pub fn family(mut self, family: ContentId) -> Self {
        self.family = Some(family);
        self
    }

    /// Only records of this attempt.
    pub fn attempt(mut self, attempt: RecordId) -> Self {
        self.attempt = Some(attempt);
        self
    }

    /// Only records of this task instance.
    pub fn task_instance(mut self, instance: ContentId) -> Self {
        self.task_instance = Some(instance);
        self
    }

    /// Only records made in `[from_ns, to_ns)`.
    pub fn between(mut self, from_ns: u64, to_ns: u64) -> Self {
        self.time = Some((from_ns, to_ns));
        self
    }

    /// Only records of this epistemic class.
    pub fn epistemic(mut self, epistemic: Epistemic) -> Self {
        self.epistemic = Some(epistemic);
        self
    }

    /// At most this many records, the lowest ids first.
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
}

/// How a query was answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plan {
    /// Zone maps pruned blocks and the rest were scanned.
    Scan,
    /// A persisted index named the records and only their blocks were read.
    Index,
}

/// The answer and what it cost.
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// The records, in id order, each once.
    pub records: Vec<Record>,
    /// How the query was answered.
    pub plan: Plan,
    /// Record blocks in the snapshot.
    pub blocks_total: u64,
    /// Record blocks read to answer.
    pub blocks_read: u64,
}
