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
use crate::error::{Error, Result};
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
        heads.extend(self.catalog_heads()?);
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

    /// Every file the snapshot needs, as paths under the database root in
    /// sorted order: the manifests that make it up, then the segments, blob
    /// packs and index files it holds. Copying these and
    /// [`Database::adopt`]ing the snapshot's id elsewhere reproduces it.
    pub fn files(&self) -> Result<Vec<String>> {
        let resolved = self.db.resolve(&[self.id])?;
        let mut paths = Vec::new();
        for manifest in &resolved.manifests {
            paths.push(Key::new(Kind::Manifest, &manifest.to_string())?.relative_path());
        }
        for object in &resolved.live {
            paths.push(object.key()?.relative_path());
        }
        paths.sort();
        paths.dedup();
        Ok(paths)
    }

    /// Makes sure the manifest the snapshot is named by exists, so its id can
    /// be recorded and reopened later. A snapshot of one head is named by
    /// that head; any other (several heads, or none) by their merge, which is
    /// written here, deterministically, so doing it twice changes nothing.
    pub fn persist(&self) -> Result<ContentId> {
        if self.heads.len() != 1 {
            self.db
                .put_manifest(&crate::manifest::Manifest::merge(self.heads.clone()))?;
        }
        Ok(self.id)
    }

    /// Keeps the snapshot's files alive under `holder`'s name until
    /// [`Database::unpin`]. A snapshot of several heads is first stored as a
    /// merge manifest, so the pin names something that exists.
    pub fn pin(&self, holder: &str) -> Result<()> {
        self.persist()?;
        let key = Key::new(Kind::Pin, holder)?;
        self.db
            .backend()
            .replace(&key, self.id.to_string().as_bytes())?;
        // The pin protects the snapshot's files from now on, but a collection
        // that began earlier may already have taken some. Look: if anything is
        // gone the pin is withdrawn and the caller told, rather than left
        // naming a snapshot that can no longer be read.
        if let Err(missing) = self.verify_files() {
            self.db.backend().remove(&key)?;
            return Err(missing);
        }
        Ok(())
    }

    /// Checks that every manifest and file of the snapshot still exists.
    fn verify_files(&self) -> Result<()> {
        let resolved = self.db.resolve(&[self.id])?;
        for manifest in &resolved.manifests {
            let key = Key::new(Kind::Manifest, &manifest.to_string())?;
            if !self.db.backend().exists(&key)? {
                return Err(Error::NotFound {
                    what: format!("manifest {manifest} of the snapshot, already collected"),
                });
            }
        }
        for object in &resolved.live {
            if !self.db.backend().exists(&object.key()?)? {
                return Err(Error::NotFound {
                    what: format!("file {} of the snapshot, already collected", object.id),
                });
            }
        }
        Ok(())
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
