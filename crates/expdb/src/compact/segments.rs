// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Merging small segments.

use std::collections::BTreeSet;

use super::{group_by_size, CompactReport};
use crate::database::Database;
use crate::error::Result;
use crate::format::{seal_segment, Segment};
use crate::manifest::ObjectRef;
use crate::model::{Edge, Record};

impl Database {
    /// Merges the snapshot's segments smaller than
    /// [`Config::compact_target_bytes`](crate::Config) into larger ones. Each
    /// merged segment holds its inputs' records in id order, once each, so
    /// merging the same inputs always gives the same file.
    pub fn compact_segments(&self) -> Result<CompactReport> {
        let snapshot = self.snapshot()?;
        let sized: Vec<(u64, ObjectRef)> = snapshot
            .segments()
            .into_iter()
            .map(|s| (s.bytes, s))
            .collect();
        let mut report = CompactReport::default();
        for group in group_by_size(&sized, self.config().compact_target_bytes as u64) {
            let mut records: Vec<Record> = Vec::new();
            let mut edges: BTreeSet<Edge> = BTreeSet::new();
            for object in &group {
                let segment = Segment::open(self.backend_arc(), object.id)?;
                records.extend(segment.records()?);
                edges.extend(segment.edges()?);
            }
            records.sort_by_key(|r| r.id);
            records.dedup_by_key(|r| r.id);
            let edges: Vec<Edge> = edges.into_iter().collect();
            let id = seal_segment(self.backend(), &records, &edges, self.config())?;
            let merged = Segment::open(self.backend_arc(), id)?.object_ref()?;
            // A merge that reproduces one of its inputs must not remove it.
            let removed = group.iter().filter(|o| o.id != id).cloned().collect();
            self.publish_once("compactor", vec![merged], removed)?;
            report.groups += 1;
            report.inputs += group.len();
            report.outputs.push(id);
        }
        Ok(report)
    }
}
