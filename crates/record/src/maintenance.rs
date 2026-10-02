// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable, content-addressed experience
// stores for learning agents, for its clients. If your team needs expertise
// in training-data lineage or crash-safe storage, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Keeping the experience database small and quick to open: merging the small
//! files every commit leaves behind, indexing what is not indexed, folding the
//! history of finished writers into the catalog, and, when asked, deleting
//! what nothing reaches any more.
//!
//! None of it changes what is stored. It is safe to run beside a running
//! command and from several processes at once.

use serde::Serialize;

use crate::error::StoreError;
use crate::workspace::Workspace;

/// What the database holds, as files.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Storage {
    /// Segments of records.
    pub segments: usize,
    /// Packs of blob chunks.
    pub blob_packs: usize,
    /// Index runs.
    pub index_runs: usize,
    /// Snapshots held alive by a name, such as a dataset's.
    pub pins: usize,
    /// Manifests every open walks, one per commit since the last checkpoint.
    pub history: usize,
}

/// Once opening walks more manifests than this, maintenance checkpoints.
const CHECKPOINT_AFTER: usize = 64;

/// What a maintenance pass did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Maintained {
    /// What the database held before.
    pub before: Storage,
    /// What it holds now.
    pub after: Storage,
    /// Groups of segments merged into one.
    pub segment_groups: usize,
    /// Groups of blob packs merged into one.
    pub blob_groups: usize,
    /// Writers whose history was folded into the catalog.
    pub writers_retired: usize,
    /// Files deleted, when collection was asked for.
    pub removed: Option<usize>,
}

impl Workspace {
    /// What the database holds, as files.
    pub fn storage(&self) -> Result<Storage, StoreError> {
        self.commit()?;
        let db = self.database()?;
        let pins = db.pins()?.len();
        let history = db.history_depth()?;
        self.read(|s| {
            let snapshot = s.snapshot()?;
            Ok(Storage {
                segments: snapshot.segments().len(),
                blob_packs: snapshot.blob_packs().len(),
                index_runs: snapshot.index_runs().len(),
                pins,
                history,
            })
        })
    }

    /// Merges small files, indexes the rest and retires finished writers.
    /// With `collect`, also deletes files nothing reaches and that are older
    /// than the grace period, which keeps every snapshot a name pins and
    /// everything a writer might still be about to publish.
    pub fn maintain(&self, collect: bool) -> Result<Maintained, StoreError> {
        let before = self.storage()?;
        let db = self.database()?;
        let segments = db.compact_segments()?;
        let blobs = db.compact_blobs()?;
        db.build_indexes()?;
        db.compact_indexes()?;
        let absorbed = db.absorb_jobs()?;
        if db.history_depth()? > CHECKPOINT_AFTER {
            db.checkpoint()?;
        }
        let removed = if collect {
            Some(db.gc()?.removed)
        } else {
            None
        };
        self.refresh()?;
        Ok(Maintained {
            before,
            after: self.storage()?,
            segment_groups: segments.groups,
            blob_groups: blobs.groups,
            writers_retired: absorbed.pruned,
            removed,
        })
    }
}
