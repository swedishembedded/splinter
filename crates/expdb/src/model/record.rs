// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The record envelope: identity, position in the graph and a body.

use serde::{Deserialize, Serialize};

use super::body::{Body, RecordKind};
use crate::id::{ContentId, RecordId};

/// The current version of the body schema.
pub const SCHEMA_VERSION: u16 = 1;

/// One immutable record. The envelope fields are stored as columns so a scan
/// can filter on them without parsing the body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// The record's identity, stable across compaction.
    pub id: RecordId,
    /// When it was made, in nanoseconds since the Unix epoch.
    pub timestamp_ns: u64,
    /// The record before this one on its path through the graph.
    pub parent: Option<RecordId>,
    /// The attempt the record belongs to.
    pub attempt: Option<RecordId>,
    /// The episode family the record belongs to.
    pub family: Option<ContentId>,
    /// The task instance the record belongs to.
    pub task_instance: Option<ContentId>,
    /// The version of the body's schema, so old records stay readable.
    pub schema: u16,
    /// What the record says.
    pub body: Body,
}

impl Record {
    /// A record with no position in the graph yet.
    pub fn new(id: RecordId, timestamp_ns: u64, body: Body) -> Self {
        Self {
            id,
            timestamp_ns,
            parent: None,
            attempt: None,
            family: None,
            task_instance: None,
            schema: SCHEMA_VERSION,
            body,
        }
    }

    /// The record's kind.
    pub fn kind(&self) -> RecordKind {
        self.body.kind()
    }

    /// Sets the parent.
    pub fn with_parent(mut self, parent: RecordId) -> Self {
        self.parent = Some(parent);
        self
    }

    /// Sets the attempt, family and task instance it belongs to.
    pub fn in_attempt(
        mut self,
        attempt: RecordId,
        family: ContentId,
        task_instance: ContentId,
    ) -> Self {
        self.attempt = Some(attempt);
        self.family = Some(family);
        self.task_instance = Some(task_instance);
        self
    }
}
