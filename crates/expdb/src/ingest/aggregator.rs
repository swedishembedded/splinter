// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The aggregator: many small spooled segments from a node's writers become a
//! few large published ones, so the shared filesystem sees few large writes.

use std::collections::BTreeSet;

use crate::backend::Kind;
use crate::database::Database;
use crate::error::{Error, Result};
use crate::format::{seal_segment, Segment};
use crate::id::ContentId;
use crate::model::{Edge, Record};

/// What a drain did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DrainReport {
    /// Spooled segments consumed.
    pub microsegments: usize,
    /// Records published.
    pub records: usize,
    /// Edges published.
    pub edges: usize,
    /// The segments published.
    pub segments: Vec<ContentId>,
}

/// Merges a spool into published segments.
pub struct Aggregator {
    db: Database,
}

impl Aggregator {
    /// An aggregator over `db`'s spool.
    pub fn new(db: &Database) -> Self {
        Self { db: db.clone() }
    }

    /// Merges everything spooled into segments of at most
    /// `max_buffered_records` records, publishes them under `job`, and only
    /// then deletes the spooled files. If the process dies in between, the
    /// next drain merges the same input into the same segments, so the
    /// worst outcome is a repeated manifest, never a lost or doubled record.
    pub fn drain(&self, job: &str) -> Result<DrainReport> {
        let backend = self.db.backend_arc();
        let keys = backend.list(Kind::Spool)?;
        if keys.is_empty() {
            return Ok(DrainReport::default());
        }
        let mut records: Vec<Record> = Vec::new();
        let mut edges: BTreeSet<Edge> = BTreeSet::new();
        for key in &keys {
            let name = key.name().strip_suffix(".seg").unwrap_or(key.name());
            let id = ContentId::parse(name)?;
            let segment = Segment::open_in(backend.clone(), Kind::Spool, id)?;
            records.extend(segment.records()?);
            edges.extend(segment.edges()?);
        }
        records.sort_by_key(|r| r.id);
        records.dedup_by_key(|r| r.id);
        let edges: Vec<Edge> = edges.into_iter().collect();

        let per_segment = self.db.config().max_buffered_records;
        let mut report = DrainReport {
            microsegments: keys.len(),
            records: records.len(),
            edges: edges.len(),
            segments: Vec::new(),
        };
        let mut objects = Vec::new();
        let mut edge_chunks = edges.chunks(per_segment);
        for chunk in records.chunks(per_segment) {
            let id = seal_segment(
                backend.as_ref(),
                chunk,
                edge_chunks.next().unwrap_or(&[]),
                self.db.config(),
            )?;
            objects.push(Segment::open(backend.clone(), id)?.object_ref()?);
            report.segments.push(id);
        }
        for chunk in edge_chunks {
            let id = seal_segment(backend.as_ref(), &[], chunk, self.db.config())?;
            objects.push(Segment::open(backend.clone(), id)?.object_ref()?);
            report.segments.push(id);
        }
        if objects.is_empty() {
            return Err(Error::corrupt(
                "spool",
                "spooled segments held no records and no edges",
            ));
        }
        self.db
            .publish_once(&format!("aggregator-{job}"), objects, Vec::new())?;
        for key in keys {
            backend.remove(&key)?;
        }
        Ok(report)
    }
}
