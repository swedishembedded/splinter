// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Pack files: many chunks in one immutable file with a sorted index.
//!
//! ```text
//! magic | entry bytes... | index | footer
//! index  = count x (id[32] offset[8] len[4] kind[1]), sorted by id
//! footer = index_offset[8] count[4] index_crc[4] magic
//! ```

use std::collections::BTreeMap;

use crate::backend::{Key, Kind, StorageBackend};
use crate::error::{Error, Result};
use crate::id::ContentId;

const MAGIC: &[u8; 8] = b"EXPPACK1";
const INDEX_ENTRY: usize = 32 + 8 + 4 + 1;
const FOOTER: usize = 8 + 4 + 4 + 8;

/// What an entry's bytes are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EntryKind {
    /// The bytes of a chunk, or of a whole small object.
    Raw,
    /// A list of `(chunk id, chunk length)` that make up a large object.
    ChunkList,
}

impl EntryKind {
    fn byte(self) -> u8 {
        match self {
            EntryKind::Raw => 0,
            EntryKind::ChunkList => 1,
        }
    }

    fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(EntryKind::Raw),
            1 => Some(EntryKind::ChunkList),
            _ => None,
        }
    }
}

/// One row of a pack's index: id, offset, length and kind.
pub(crate) type IndexRow = (ContentId, u64, u32, EntryKind);

/// Where an entry lives.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Location {
    pub(crate) pack: ContentId,
    pub(crate) offset: u64,
    pub(crate) len: u32,
    pub(crate) kind: EntryKind,
}

/// Entries gathered for the pack being built.
#[derive(Default)]
pub(crate) struct PackBuilder {
    entries: BTreeMap<ContentId, (EntryKind, Vec<u8>)>,
    bytes: usize,
}

impl PackBuilder {
    pub(crate) fn contains(&self, id: &ContentId) -> bool {
        self.entries.contains_key(id)
    }

    pub(crate) fn get(&self, id: &ContentId) -> Option<(EntryKind, &[u8])> {
        self.entries
            .get(id)
            .map(|(kind, data)| (*kind, data.as_slice()))
    }

    /// Adds an entry; false if the id was already there.
    pub(crate) fn add(&mut self, id: ContentId, kind: EntryKind, data: Vec<u8>) -> bool {
        if self.entries.contains_key(&id) {
            return false;
        }
        self.bytes += data.len();
        self.entries.insert(id, (kind, data));
        true
    }

    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The file's name (the hash of its bytes), its bytes and its index.
    pub(crate) fn finish(self) -> (ContentId, Vec<u8>, Vec<IndexRow>) {
        let mut file = MAGIC.to_vec();
        let mut index = Vec::with_capacity(self.entries.len());
        for (id, (kind, data)) in self.entries {
            index.push((id, file.len() as u64, data.len() as u32, kind));
            file.extend_from_slice(&data);
        }
        let index_offset = file.len() as u64;
        let mut table = Vec::with_capacity(index.len() * INDEX_ENTRY);
        for (id, offset, len, kind) in &index {
            table.extend_from_slice(id.as_bytes());
            table.extend_from_slice(&offset.to_le_bytes());
            table.extend_from_slice(&len.to_le_bytes());
            table.push(kind.byte());
        }
        let crc = crc32fast::hash(&table);
        file.extend_from_slice(&table);
        file.extend_from_slice(&index_offset.to_le_bytes());
        file.extend_from_slice(&(index.len() as u32).to_le_bytes());
        file.extend_from_slice(&crc.to_le_bytes());
        file.extend_from_slice(MAGIC);
        (ContentId::of(&file), file, index)
    }
}

/// The key of the pack file named by its content hash.
pub(crate) fn pack_key(name: &ContentId) -> Result<Key> {
    Key::new(Kind::BlobPack, &format!("{name}.pack"))
}

/// Reads a pack's index from its footer without reading its data.
pub(crate) fn read_index(backend: &dyn StorageBackend, name: &ContentId) -> Result<Vec<IndexRow>> {
    let key = pack_key(name)?;
    let what = format!("blob pack {name}");
    let len = backend.len(&key)?;
    if len < (MAGIC.len() + FOOTER) as u64 {
        return Err(Error::corrupt(what, "shorter than its footer"));
    }
    if backend.read_range(&key, 0, MAGIC.len())? != MAGIC {
        return Err(Error::corrupt(what, "leading magic is wrong"));
    }
    let footer = backend.read_range(&key, len - FOOTER as u64, FOOTER)?;
    if &footer[FOOTER - 8..] != MAGIC {
        return Err(Error::corrupt(what, "footer magic is wrong"));
    }
    let index_offset = u64::from_le_bytes(footer[0..8].try_into().unwrap_or([0; 8]));
    let count = u32::from_le_bytes(footer[8..12].try_into().unwrap_or([0; 4])) as usize;
    let crc = u32::from_le_bytes(footer[12..16].try_into().unwrap_or([0; 4]));
    let table_len = count * INDEX_ENTRY;
    if index_offset + table_len as u64 + FOOTER as u64 != len {
        return Err(Error::corrupt(
            what,
            "index does not end where the footer begins",
        ));
    }
    let table = backend.read_range(&key, index_offset, table_len)?;
    if crc32fast::hash(&table) != crc {
        return Err(Error::corrupt(what, "index checksum mismatch"));
    }
    let mut entries = Vec::with_capacity(count);
    for row in table.as_chunks::<INDEX_ENTRY>().0 {
        let mut id = [0u8; 32];
        id.copy_from_slice(&row[..32]);
        let offset = u64::from_le_bytes(row[32..40].try_into().unwrap_or([0; 8]));
        let entry_len = u32::from_le_bytes(row[40..44].try_into().unwrap_or([0; 4]));
        let kind = EntryKind::from_byte(row[44])
            .ok_or_else(|| Error::corrupt(what.clone(), "unknown entry kind"))?;
        entries.push((ContentId::from_bytes(id), offset, entry_len, kind));
    }
    Ok(entries)
}

/// Every entry of a pack, each checked: a raw entry must hash to its id.
pub(crate) fn read_all(
    backend: &dyn StorageBackend,
    name: &ContentId,
) -> Result<Vec<(ContentId, EntryKind, Vec<u8>)>> {
    let rows = read_index(backend, name)?;
    let file = backend.read(&pack_key(name)?)?;
    let mut entries = Vec::with_capacity(rows.len());
    for (id, offset, len, kind) in rows {
        let data = file
            .get(offset as usize..offset as usize + len as usize)
            .ok_or_else(|| {
                Error::corrupt(
                    format!("blob pack {name}"),
                    "an entry runs past the end of the file",
                )
            })?
            .to_vec();
        if kind == EntryKind::Raw && ContentId::of(&data) != id {
            return Err(Error::corrupt(
                format!("blob pack {name}"),
                format!("entry {id} holds different bytes"),
            ));
        }
        entries.push((id, kind, data));
    }
    Ok(entries)
}
