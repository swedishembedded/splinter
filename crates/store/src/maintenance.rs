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
//! what nothing reaches any more, files and artifacts included.
//!
//! None of it changes what is stored. It is safe to run beside a running
//! command and from several processes at once.

use std::time::Duration;

use serde::Serialize;

use crate::artifacts::ArtifactStore;
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
    /// Files the database tracks as artifacts.
    pub artifacts: usize,
    /// Items written off as lost.
    pub losses: usize,
}

/// Once opening walks more manifests than this, maintenance checkpoints.
const CHECKPOINT_AFTER: usize = 64;

/// How old an unrecorded artifact file must be before collection takes it: a
/// file being written is not yet recorded, and must not be taken.
const ORPHAN_GRACE: Duration = Duration::from_secs(3600);

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
    /// Artifact files no commit made official that were deleted, when
    /// collection was asked for.
    pub orphan_artifacts: Option<usize>,
}

/// A name that holds a snapshot of the database alive.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Pin {
    /// Who holds it, such as `dataset-<hex>`.
    pub holder: String,
    /// The snapshot it keeps readable.
    pub snapshot: String,
}

impl Workspace {
    /// Holds the database as it is now alive under `holder`'s name until
    /// [`Workspace::release_pin`].
    pub fn hold(&self, holder: &str) -> Result<(), StoreError> {
        self.commit()?;
        self.read(|s| s.snapshot()?.pin(holder))
    }

    /// The snapshots held alive, by name.
    pub fn pins(&self) -> Result<Vec<Pin>, StoreError> {
        if !self.initialised() {
            return Ok(Vec::new());
        }
        Ok(self
            .database()?
            .pins()?
            .into_iter()
            .map(|(holder, id)| Pin {
                holder,
                snapshot: id.to_string(),
            })
            .collect())
    }

    /// Lets go of a held snapshot; its files become collectable once nothing
    /// else reaches them. Refused for a name that holds nothing.
    pub fn release_pin(&self, holder: &str) -> Result<(), StoreError> {
        if !self.pins()?.iter().any(|p| p.holder == holder) {
            return Err(StoreError::Recovery(format!(
                "no snapshot is held by {holder}"
            )));
        }
        Ok(self.database()?.unpin(holder)?)
    }

    /// What the database holds, as files.
    pub fn storage(&self) -> Result<Storage, StoreError> {
        self.commit()?;
        let db = self.database()?;
        let pins = db.pins()?.len();
        let history = db.history_depth()?;
        let artifacts = ArtifactStore::new(self, self.root()).list()?.len();
        let losses = self.losses()?.len();
        self.read(|s| {
            let snapshot = s.snapshot()?;
            Ok(Storage {
                segments: snapshot.segments().len(),
                blob_packs: snapshot.blob_packs().len(),
                index_runs: snapshot.index_runs().len(),
                pins,
                history,
                artifacts,
                losses,
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
        let (removed, orphan_artifacts) = if collect {
            let swept = ArtifactStore::new(self, self.root()).sweep_orphans(ORPHAN_GRACE)?;
            (Some(db.gc()?.removed), Some(swept))
        } else {
            (None, None)
        };
        self.refresh()?;
        Ok(Maintained {
            before,
            after: self.storage()?,
            segment_groups: segments.groups,
            blob_groups: blobs.groups,
            writers_retired: absorbed.pruned,
            removed,
            orphan_artifacts,
        })
    }
}
