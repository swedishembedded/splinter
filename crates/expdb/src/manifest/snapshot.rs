// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Snapshots: a pinned, repeatable view of the database.

use std::collections::HashSet;

use super::model::{ObjectKind, ObjectRef};
use crate::backend::{Key, Kind};
use crate::database::Database;
use crate::error::Result;
use crate::format::Segment;
use crate::id::{ContentId, RecordId};
use crate::model::{Edge, Record};

/// A fixed set of files. Whatever is published or compacted afterwards, a
/// snapshot reads the same records for as long as its files exist, and its
/// files exist for as long as it is pinned.
pub struct Snapshot {
    db: Database,
    id: ContentId,
    heads: Vec<ContentId>,
    live: Vec<ObjectRef>,
    cache: std::sync::Arc<crate::index::SnapshotCache>,
}

impl Database {
    /// A snapshot of everything published so far: the catalog and every job.
    pub fn snapshot(&self) -> Result<Snapshot> {
        let mut heads = self.job_heads()?;
        heads.extend(self.get_ref("catalog")?);
        heads.sort();
        heads.dedup();
        Snapshot::of(self, heads)
    }

    /// The snapshot with the given id, as it was.
    pub fn snapshot_at(&self, id: ContentId) -> Result<Snapshot> {
        Snapshot::of(self, vec![id])
    }

    /// The pins held, as `(holder, snapshot id)`.
    pub fn pins(&self) -> Result<Vec<(String, ContentId)>> {
        let mut pins = Vec::new();
        for key in self.backend().list(Kind::Pin)? {
            let bytes = self.backend().read(&key)?;
            pins.push((
                key.name().to_owned(),
                ContentId::parse(String::from_utf8_lossy(&bytes).trim())?,
            ));
        }
        Ok(pins)
    }

    /// Releases a pin. Its files become collectable once nothing else
    /// reaches them.
    pub fn unpin(&self, holder: &str) -> Result<()> {
        self.backend().remove(&Key::new(Kind::Pin, holder)?)
    }
}

impl Snapshot {
    fn of(db: &Database, heads: Vec<ContentId>) -> Result<Self> {
        let resolved = db.resolve(&heads)?;
        let id = match heads.as_slice() {
            [only] => *only,
            _ => crate::manifest::Manifest::merge(heads.clone()).id()?,
        };
        Ok(Self {
            db: db.clone(),
            id,
            heads,
            live: resolved.live.into_iter().collect(),
            cache: Default::default(),
        })
    }

    /// The snapshot's id, the manifest it names.
    pub fn id(&self) -> ContentId {
        self.id
    }

    /// The database the snapshot reads.
    pub fn database(&self) -> &Database {
        &self.db
    }

    /// The segments that are part of the snapshot.
    pub fn segments(&self) -> Vec<ObjectRef> {
        self.of_kind(ObjectKind::Segment)
    }

    /// The blob packs that are part of the snapshot.
    pub fn blob_packs(&self) -> Vec<ObjectRef> {
        self.of_kind(ObjectKind::BlobPack)
    }

    pub(crate) fn cache(&self) -> &crate::index::SnapshotCache {
        &self.cache
    }

    pub(crate) fn objects(&self, kind: ObjectKind) -> Vec<ObjectRef> {
        self.of_kind(kind)
    }

    fn of_kind(&self, kind: ObjectKind) -> Vec<ObjectRef> {
        self.live
            .iter()
            .filter(|o| o.kind == kind)
            .cloned()
            .collect()
    }

    /// Opens one of the snapshot's segments.
    pub fn open_segment(&self, object: &ObjectRef) -> Result<Segment> {
        let mut segment = Segment::open(self.db.backend_arc(), object.id)?;
        segment.count_blocks_in(std::sync::Arc::clone(&self.cache.blocks_read));
        Ok(segment)
    }

    /// Keeps the snapshot's files alive under `holder`'s name until
    /// [`Database::unpin`]. A snapshot of several heads is first stored as a
    /// merge manifest, so the pin names something that exists.
    pub fn pin(&self, holder: &str) -> Result<()> {
        if self.heads.len() > 1 {
            self.db
                .put_manifest(&crate::manifest::Manifest::merge(self.heads.clone()))?;
        }
        self.db.backend().replace(
            &Key::new(Kind::Pin, holder)?,
            self.id.to_string().as_bytes(),
        )
    }

    /// Every record, once: the first copy wins if two segments hold the
    /// same record, as they do while a compaction's inputs are still named.
    pub fn records(&self) -> Result<Vec<Record>> {
        let mut seen: HashSet<RecordId> = HashSet::new();
        let mut records = Vec::new();
        for object in self.segments() {
            for record in self.open_segment(&object)?.records()? {
                if seen.insert(record.id) {
                    records.push(record);
                }
            }
        }
        Ok(records)
    }

    /// Every edge, once.
    pub fn edges(&self) -> Result<Vec<Edge>> {
        let mut seen: HashSet<Edge> = HashSet::new();
        let mut edges = Vec::new();
        for object in self.segments() {
            for edge in self.open_segment(&object)?.edges()? {
                if seen.insert(edge) {
                    edges.push(edge);
                }
            }
        }
        Ok(edges)
    }
}
