// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The blob store: put bytes, get them back by the id of their content.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::chunker::chunks;
use super::pack::{pack_key, read_all, read_index, EntryKind, Location, PackBuilder};
use crate::backend::{Kind, StorageBackend};
use crate::config::Config;
use crate::database::Database;
use crate::error::{Error, Result};
use crate::id::ContentId;

/// A reference to stored bytes: the hash of the whole object and its length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlobRef {
    /// The content id of the whole object.
    pub id: ContentId,
    /// Its length in bytes.
    pub len: u64,
}

/// What a store has written since it was opened.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BlobStats {
    /// Objects put, including ones already stored.
    pub objects_put: u64,
    /// Entries (chunks and chunk lists) newly written.
    pub chunks_written: u64,
    /// Bytes of those entries.
    pub bytes_written: u64,
}

/// A store over the blob packs of a database. Packs it writes are named by
/// the manifest that publishes them; until [`flush`](BlobStore::flush) the
/// newest entries live only in memory.
pub struct BlobStore {
    backend: Arc<dyn StorageBackend>,
    config: Arc<Config>,
    index: HashMap<ContentId, Location>,
    builder: PackBuilder,
    written: Vec<ContentId>,
    stats: BlobStats,
}

impl BlobStore {
    /// A store over every pack the database holds.
    pub fn open(db: &Database) -> Result<Self> {
        let names = db
            .backend()
            .list(Kind::BlobPack)?
            .into_iter()
            .filter_map(|key| {
                key.name()
                    .strip_suffix(".pack")
                    .and_then(|n| ContentId::parse(n).ok())
            })
            .collect::<Vec<_>>();
        Self::with_packs(db, &names)
    }

    /// A store over exactly the named packs, for reading a snapshot.
    pub fn with_packs(db: &Database, packs: &[ContentId]) -> Result<Self> {
        let mut store = Self {
            backend: db.backend_arc(),
            config: Arc::new(db.config().clone()),
            index: HashMap::new(),
            builder: PackBuilder::default(),
            written: Vec::new(),
            stats: BlobStats::default(),
        };
        for pack in packs {
            store.load_pack(pack)?;
        }
        Ok(store)
    }

    fn load_pack(&mut self, pack: &ContentId) -> Result<()> {
        for (id, offset, len, kind) in read_index(self.backend.as_ref(), pack)? {
            self.index.entry(id).or_insert(Location {
                pack: *pack,
                offset,
                len,
                kind,
            });
        }
        Ok(())
    }

    /// Writes one pack holding every entry of `packs`, each once, and returns
    /// the reference a manifest publishes it by. The inputs are left as they
    /// are; removing them is the manifest's business.
    pub fn repack(db: &Database, packs: &[ContentId]) -> Result<crate::manifest::ObjectRef> {
        let mut builder = PackBuilder::default();
        for pack in packs {
            for (id, kind, data) in read_all(db.backend(), pack)? {
                builder.add(id, kind, data);
            }
        }
        let (name, file, _) = builder.finish();
        db.backend().write_once(&pack_key(&name)?, &file)?;
        Ok(crate::manifest::ObjectRef::blob_pack(
            name,
            file.len() as u64,
        ))
    }

    /// What this store has written.
    pub fn stats(&self) -> BlobStats {
        self.stats
    }

    /// Whether the object is stored (flushed or pending).
    pub fn contains(&self, blob: &BlobRef) -> bool {
        self.index.contains_key(&blob.id) || self.builder.contains(&blob.id)
    }

    /// Stores `data` and returns its reference. Identical content already
    /// stored, whole or in part, is not stored again.
    pub fn put(&mut self, data: &[u8]) -> Result<BlobRef> {
        self.stats.objects_put += 1;
        let blob = BlobRef {
            id: ContentId::of(data),
            len: data.len() as u64,
        };
        if self.contains(&blob) {
            return Ok(blob);
        }
        let pieces = chunks(data, &self.config);
        if pieces.len() <= 1 {
            self.add(blob.id, EntryKind::Raw, data.to_vec())?;
            return Ok(blob);
        }
        let mut list = Vec::with_capacity(pieces.len() * 36);
        for (offset, len) in pieces {
            let chunk = &data[offset..offset + len];
            let id = ContentId::of(chunk);
            list.extend_from_slice(id.as_bytes());
            list.extend_from_slice(&(len as u32).to_le_bytes());
            if !self.index.contains_key(&id) {
                self.add(id, EntryKind::Raw, chunk.to_vec())?;
            }
        }
        self.add(blob.id, EntryKind::ChunkList, list)?;
        Ok(blob)
    }

    fn add(&mut self, id: ContentId, kind: EntryKind, data: Vec<u8>) -> Result<()> {
        let len = data.len() as u64;
        if self.builder.add(id, kind, data) {
            self.stats.chunks_written += 1;
            self.stats.bytes_written += len;
        }
        if self.builder.bytes() >= self.config.pack_target_bytes {
            self.seal()?;
        }
        Ok(())
    }

    fn seal(&mut self) -> Result<()> {
        if self.builder.is_empty() {
            return Ok(());
        }
        let (name, file, entries) = std::mem::take(&mut self.builder).finish();
        self.backend.write_once(&pack_key(&name)?, &file)?;
        for (id, offset, len, kind) in entries {
            self.index.insert(
                id,
                Location {
                    pack: name,
                    offset,
                    len,
                    kind,
                },
            );
        }
        self.written.push(name);
        Ok(())
    }

    /// Writes the open pack, and returns the name of every pack this store
    /// has written, for the manifest that publishes them.
    pub fn flush(&mut self) -> Result<Vec<ContentId>> {
        self.seal()?;
        Ok(self.written.clone())
    }

    /// The bytes of an object, verified against their content id.
    pub fn get(&self, blob: &BlobRef) -> Result<Vec<u8>> {
        let (kind, bytes) = self.entry(&blob.id)?;
        let data = match kind {
            EntryKind::Raw => bytes,
            EntryKind::ChunkList => {
                let mut whole = Vec::with_capacity(blob.len as usize);
                for item in bytes.as_chunks::<36>().0.iter() {
                    let mut id = [0u8; 32];
                    id.copy_from_slice(&item[..32]);
                    let (_, chunk) = self.entry(&ContentId::from_bytes(id))?;
                    whole.extend_from_slice(&chunk);
                }
                whole
            }
        };
        if ContentId::of(&data) != blob.id || data.len() as u64 != blob.len {
            return Err(Error::corrupt(
                format!("blob {}", blob.id),
                "bytes do not match their content id",
            ));
        }
        Ok(data)
    }

    /// The bytes of an object known only by id; the length is read from the
    /// stored entry itself.
    pub fn get_by_id(&self, id: &ContentId) -> Result<Vec<u8>> {
        let (kind, bytes) = self.entry(id)?;
        let len = match kind {
            EntryKind::Raw => bytes.len() as u64,
            EntryKind::ChunkList => bytes
                .as_chunks::<36>()
                .0
                .iter()
                .map(|item| {
                    u64::from(u32::from_le_bytes(
                        item[32..36].try_into().unwrap_or([0; 4]),
                    ))
                })
                .sum(),
        };
        self.get(&BlobRef { id: *id, len })
    }

    /// One verified entry: a chunk or a chunk list.
    fn entry(&self, id: &ContentId) -> Result<(EntryKind, Vec<u8>)> {
        if let Some((kind, data)) = self.builder.get(id) {
            return Ok((kind, data.to_vec()));
        }
        let at = self.index.get(id).ok_or_else(|| Error::NotFound {
            what: format!("blob {id}"),
        })?;
        let data = self
            .backend
            .read_range(&pack_key(&at.pack)?, at.offset, at.len as usize)?;
        if at.kind == EntryKind::Raw && ContentId::of(&data) != *id {
            return Err(Error::corrupt(
                format!("blob {id}"),
                format!("pack {} holds different bytes", at.pack),
            ));
        }
        Ok((at.kind, data))
    }
}
