// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Reading a snapshot through its index: assembling it, counting what it
//! cost, and fetching single records without scanning.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use super::model::Index;
use super::run::Run;
use crate::error::{Error, Result};
use crate::format::{Block, Segment};
use crate::id::{ContentId, RecordId};
use crate::manifest::{ObjectKind, ObjectRef, Snapshot};
use crate::model::Record;

/// Blocks kept decoded for single-record reads.
const BLOCK_CACHE: usize = 64;

/// What reading a snapshot has cost, so tests and operators can see whether
/// an index or a zone map saved a scan.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ScanStats {
    /// Record blocks read and decoded.
    pub blocks_read: u64,
    /// Segments opened.
    pub segments_opened: u64,
}

/// What a snapshot remembers between calls.
#[derive(Default)]
pub(crate) struct SnapshotCache {
    index: Mutex<Option<Arc<Index>>>,
    segments: Mutex<HashMap<ContentId, Arc<Segment>>>,
    blocks: Mutex<HashMap<(ContentId, u32), Arc<Block>>>,
    pub(crate) blocks_read: Arc<AtomicU64>,
    segments_opened: AtomicU64,
}

impl SnapshotCache {
    pub(crate) fn has_index(&self) -> bool {
        locked(&self.index).is_some()
    }
}

fn locked<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Snapshot {
    /// What reading this snapshot has cost so far.
    pub fn stats(&self) -> ScanStats {
        let cache = self.cache();
        ScanStats {
            blocks_read: cache.blocks_read.load(Ordering::Relaxed),
            segments_opened: cache.segments_opened.load(Ordering::Relaxed),
        }
    }

    /// The index runs that are part of the snapshot.
    pub fn index_runs(&self) -> Vec<ObjectRef> {
        self.objects(ObjectKind::Index)
    }

    pub(crate) fn segment(&self, id: ContentId) -> Result<Arc<Segment>> {
        if let Some(found) = locked(&self.cache().segments).get(&id) {
            return Ok(Arc::clone(found));
        }
        let mut segment = Segment::open(self.database().backend_arc(), id)?;
        segment.count_blocks_in(Arc::clone(&self.cache().blocks_read));
        self.cache().segments_opened.fetch_add(1, Ordering::Relaxed);
        let segment = Arc::new(segment);
        locked(&self.cache().segments).insert(id, Arc::clone(&segment));
        Ok(segment)
    }

    /// The snapshot's index. Persisted runs are read as they are; segments
    /// no run covers are scanned, so a missing index costs time, never
    /// correctness. Entries of segments that have left the snapshot are
    /// dropped.
    pub fn index(&self) -> Result<Arc<Index>> {
        if let Some(index) = locked(&self.cache().index).as_ref() {
            return Ok(Arc::clone(index));
        }
        let live: Vec<ContentId> = self.segments().iter().map(|o| o.id).collect();
        let live_set: HashSet<ContentId> = live.iter().copied().collect();
        let backend = self.database().backend();
        let mut runs = Vec::new();
        let mut covered = HashSet::new();
        for object in self.index_runs() {
            let run = Run::load(backend, &object.id)?;
            covered.extend(run.covers.iter().copied().filter(|c| live_set.contains(c)));
            runs.push(run);
        }
        let uncovered = live
            .iter()
            .filter(|s| !covered.contains(s))
            .map(|s| self.segment(*s))
            .collect::<Result<Vec<_>>>()?;
        if !uncovered.is_empty() {
            runs.push(Run::scan(&uncovered)?);
        }
        let merged = Run::merge(&runs, &live_set);
        let index = Arc::new(Index::assemble(std::slice::from_ref(&merged))?);
        *locked(&self.cache().index) = Some(Arc::clone(&index));
        Ok(index)
    }

    fn block(&self, segment: ContentId, block: u32) -> Result<Arc<Block>> {
        if let Some(found) = locked(&self.cache().blocks).get(&(segment, block)) {
            return Ok(Arc::clone(found));
        }
        let decoded = Arc::new(self.segment(segment)?.read_block(block as usize)?);
        let mut cache = locked(&self.cache().blocks);
        if cache.len() >= BLOCK_CACHE {
            cache.clear();
        }
        cache.insert((segment, block), Arc::clone(&decoded));
        Ok(decoded)
    }

    /// One record, read through the index without scanning.
    pub fn get(&self, id: RecordId) -> Result<Option<Record>> {
        let Some(loc) = self.index()?.loc(id) else {
            return Ok(None);
        };
        let block = self.block(loc.segment, loc.block)?;
        let record = block.record(loc.row as usize)?;
        if record.id != id {
            return Err(Error::corrupt(
                format!("index entry for {id}"),
                "the stored row holds another record",
            ));
        }
        Ok(Some(record))
    }
}
