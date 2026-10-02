// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The writer: one per process, owning a unique id, so records get ids
//! without asking anyone and segments are written without contention.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::backend::Kind;
use crate::blob::BlobStore;
use crate::database::Database;
use crate::error::Result;
use crate::format::{seal_segment, seal_segment_in, Segment};
use crate::id::{ContentId, RecordId, WriterId, WriterIdentity};
use crate::manifest::ObjectRef;
use crate::model::{Edge, Record};

/// Distinguishes writers started in one process within one clock reading.
static INCARNATION: AtomicU64 = AtomicU64::new(0);

/// Where a writer's sealed segments go.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Destination {
    /// Straight into the database, named by the writer's own manifest chain.
    Publish,
    /// Into the node-local spool, for an [`Aggregator`](super::Aggregator) to
    /// merge into large segments and publish.
    Spool,
}

/// Buffers records and edges and seals them as segments.
///
/// Memory is bounded: when [`Config::max_buffered_records`](crate::Config)
/// records are buffered the writer seals them before taking another.
pub struct Writer {
    db: Database,
    id: WriterId,
    job_ref: String,
    destination: Destination,
    next_sequence: u64,
    records: Vec<Record>,
    edges: Vec<Edge>,
    blobs: BlobStore,
    published_packs: HashSet<ContentId>,
}

impl Writer {
    /// Starts a writer for `identity`. Each start is a new incarnation, so a
    /// restarted job never reuses a sequence number its predecessor spent.
    pub fn open(
        db: &Database,
        identity: &WriterIdentity,
        destination: Destination,
    ) -> Result<Self> {
        // A fresh incarnation must differ from every other, in this process
        // and in any other process of the same identity, even on a frozen
        // clock: the process id, a counter and the standard library's
        // per-process random keys are all mixed in.
        let incarnation = {
            use std::hash::{BuildHasher, Hasher};
            let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
            hasher.write_u64(db.clock().now_ns());
            hasher.write_u32(std::process::id());
            hasher.write_u64(INCARNATION.fetch_add(1, Ordering::Relaxed));
            hasher.finish()
        };
        let id = identity.writer_id(incarnation);
        let job: String = identity
            .job()
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || matches!(c, '-' | '_') {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        Ok(Self {
            db: db.clone(),
            id,
            job_ref: format!("{job}-{id}"),
            destination,
            next_sequence: 0,
            records: Vec::new(),
            edges: Vec::new(),
            blobs: BlobStore::with_packs(db, &[])?,
            published_packs: HashSet::new(),
        })
    }

    /// This writer's id.
    pub fn id(&self) -> WriterId {
        self.id
    }

    /// The database being written.
    pub fn database(&self) -> &Database {
        &self.db
    }

    /// The id the next record will have.
    pub fn next_id(&mut self) -> RecordId {
        let id = RecordId::new(self.id, self.next_sequence);
        self.next_sequence += 1;
        id
    }

    /// Records waiting to be sealed.
    pub fn buffered(&self) -> usize {
        self.records.len().max(self.edges.len())
    }

    /// The blob store large content is put in.
    pub fn blobs(&mut self) -> &mut BlobStore {
        &mut self.blobs
    }

    /// What the blob store has written so far.
    pub fn blob_stats(&self) -> crate::blob::BlobStats {
        self.blobs.stats()
    }

    /// Buffers a record, sealing first if the buffer is full.
    pub fn push(&mut self, record: Record) -> Result<RecordId> {
        record.body.check_finite()?;
        let id = record.id;
        self.records.push(record);
        self.seal_if_full()?;
        Ok(id)
    }

    /// Buffers an edge, sealing first if the buffer is full.
    pub fn link(&mut self, edge: Edge) -> Result<()> {
        self.edges.push(edge);
        self.seal_if_full()
    }

    fn seal_if_full(&mut self) -> Result<()> {
        if self.buffered() >= self.db.config().max_buffered_records {
            self.flush()?;
        }
        Ok(())
    }

    /// Makes everything buffered durable: blob packs first, then the
    /// segment, then the manifest that names them. Returns the segment's id.
    /// Until the manifest is published, none of it is visible.
    pub fn flush(&mut self) -> Result<Option<ContentId>> {
        let packs: Vec<ContentId> = self
            .blobs
            .flush()?
            .into_iter()
            .filter(|p| !self.published_packs.contains(p))
            .collect();
        let mut add: Vec<ObjectRef> = Vec::new();
        for pack in &packs {
            let key = crate::backend::Key::new(Kind::BlobPack, &format!("{pack}.pack"))?;
            add.push(ObjectRef::blob_pack(*pack, self.db.backend().len(&key)?));
        }
        let mut segment = None;
        if !self.records.is_empty() || !self.edges.is_empty() {
            let (backend, config) = (self.db.backend(), self.db.config());
            match self.destination {
                Destination::Publish => {
                    let id = seal_segment(backend, &self.records, &self.edges, config)?;
                    add.push(Segment::open(self.db.backend_arc(), id)?.object_ref()?);
                    segment = Some(id);
                }
                Destination::Spool => {
                    segment = Some(seal_segment_in(
                        backend,
                        Kind::Spool,
                        &self.records,
                        &self.edges,
                        config,
                    )?);
                    // The spooled file now holds them; only packs remain to publish.
                    self.records.clear();
                    self.edges.clear();
                }
            }
        }
        if !add.is_empty() {
            self.db.publish(&self.job_ref, add, Vec::new())?;
            self.published_packs.extend(packs);
        }
        // Only once the manifest is out is the buffer let go. If publishing
        // failed above, the records are still here, and a retry seals the same
        // segment again (the same bytes, so nothing new is written) and
        // publishes it.
        if self.destination == Destination::Publish {
            self.records.clear();
            self.edges.clear();
        }
        Ok(segment)
    }
}
