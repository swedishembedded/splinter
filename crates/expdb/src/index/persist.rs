// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Writing and merging index runs. Both only ever add files and publish a
//! manifest, like any other writer.

use std::collections::HashSet;

use super::run::Run;
use crate::backend::{Key, Kind};
use crate::database::Database;
use crate::error::Result;
use crate::format::Segment;
use crate::id::ContentId;
use crate::manifest::{ObjectKind, ObjectRef};

const INDEXER_JOB: &str = "indexer";

impl Database {
    fn store_run(&self, run: &Run) -> Result<ObjectRef> {
        let bytes = run.encode(self.config().compression)?;
        let id = ContentId::of(&bytes);
        self.backend()
            .write_once(&Key::new(Kind::Index, &format!("{id}.idx"))?, &bytes)?;
        Ok(ObjectRef {
            kind: ObjectKind::Index,
            id,
            bytes: bytes.len() as u64,
            records: run.rows.len() as u64,
        })
    }

    /// Indexes every segment of the current snapshot no run covers yet, as
    /// one new run. Returns its id, or `None` if everything was covered.
    pub fn build_indexes(&self) -> Result<Option<ContentId>> {
        let snapshot = self.snapshot()?;
        let mut covered = HashSet::new();
        for object in snapshot.index_runs() {
            covered.extend(Run::load(self.backend(), &object.id)?.covers);
        }
        let segments = snapshot
            .segments()
            .into_iter()
            .filter(|s| !covered.contains(&s.id))
            .map(|s| Segment::open(self.backend_arc(), s.id).map(std::sync::Arc::new))
            .collect::<Result<Vec<_>>>()?;
        if segments.is_empty() {
            return Ok(None);
        }
        let object = self.store_run(&Run::scan(&segments)?)?;
        self.publish_once(INDEXER_JOB, vec![object.clone()], Vec::new())?;
        Ok(Some(object.id))
    }

    /// Merges the snapshot's index runs into one, dropping entries of
    /// segments that have left the snapshot. Returns the new run's id, or
    /// `None` if there were fewer than two runs.
    pub fn compact_indexes(&self) -> Result<Option<ContentId>> {
        let snapshot = self.snapshot()?;
        let old = snapshot.index_runs();
        if old.len() < 2 {
            return Ok(None);
        }
        let live: HashSet<ContentId> = snapshot.segments().iter().map(|s| s.id).collect();
        let runs = old
            .iter()
            .map(|o| Run::load(self.backend(), &o.id))
            .collect::<Result<Vec<_>>>()?;
        let object = self.store_run(&Run::merge(&runs, &live))?;
        let removed = old.into_iter().filter(|o| o.id != object.id).collect();
        self.publish_once(INDEXER_JOB, vec![object.clone()], removed)?;
        Ok(Some(object.id))
    }
}
