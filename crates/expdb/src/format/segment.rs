// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Writing and reading whole segments.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::columns::{decode_edges, encode_edges, encode_records, Block};
use super::frame::{frame, unframe};
use super::zone::Zone;
use crate::backend::{Key, Kind, StorageBackend};
use crate::config::Config;
use crate::error::{Error, Result};
use crate::id::ContentId;
use crate::model::{Edge, Record};

const MAGIC: &[u8; 8] = b"EXPSEG01";
const TAIL: usize = 8 + 4 + 4 + 8;

/// Where one block is and what it can hold.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockInfo {
    /// Offset of the block's frame in the file.
    pub offset: u64,
    /// Length of the frame.
    pub len: u64,
    /// Records (or edges) in the block.
    pub count: u32,
    /// What the block can hold; empty for edge blocks.
    pub zone: Zone,
}

/// A segment's directory, kept in its footer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentInfo {
    /// Total records.
    pub records: u64,
    /// Total edges.
    pub edges: u64,
    /// Record blocks in order.
    pub blocks: Vec<BlockInfo>,
    /// Edge blocks in order.
    pub edge_blocks: Vec<BlockInfo>,
}

/// The key a segment is stored under.
pub(crate) fn segment_key(kind: Kind, id: &ContentId) -> Result<Key> {
    Key::new(kind, &format!("{id}.seg"))
}

/// Encodes a segment: its content id and its bytes.
pub fn encode_segment(
    records: &[Record],
    edges: &[Edge],
    config: &Config,
) -> Result<(ContentId, Vec<u8>)> {
    if records.is_empty() && edges.is_empty() {
        return Err(Error::invalid(
            "segment",
            "a segment must hold at least one record or edge",
        ));
    }
    let per_block = config.block_records.max(1);
    let mut file = MAGIC.to_vec();
    let place = |file: &mut Vec<u8>, raw: &[u8], count: usize, zone: Zone| -> Result<BlockInfo> {
        let framed = frame(raw, config.compression)?;
        let info = BlockInfo {
            offset: file.len() as u64,
            len: framed.len() as u64,
            count: count as u32,
            zone,
        };
        file.extend_from_slice(&framed);
        Ok(info)
    };
    let mut blocks = Vec::new();
    for chunk in records.chunks(per_block) {
        blocks.push(place(
            &mut file,
            &encode_records(chunk)?,
            chunk.len(),
            Zone::of(chunk),
        )?);
    }
    let mut edge_blocks = Vec::new();
    for chunk in edges.chunks(per_block) {
        edge_blocks.push(place(
            &mut file,
            &encode_edges(chunk),
            chunk.len(),
            Zone::of(&[]),
        )?);
    }
    let info = SegmentInfo {
        records: records.len() as u64,
        edges: edges.len() as u64,
        blocks,
        edge_blocks,
    };
    let footer = serde_json::to_vec(&info).map_err(|source| Error::Encode {
        what: "segment footer",
        source,
    })?;
    let footer = frame(&footer, config.compression)?;
    let footer_offset = file.len() as u64;
    file.extend_from_slice(&footer);
    let mut tail = Vec::with_capacity(TAIL);
    tail.extend_from_slice(&footer_offset.to_le_bytes());
    tail.extend_from_slice(&(footer.len() as u32).to_le_bytes());
    let crc = crc32fast::hash(&tail);
    tail.extend_from_slice(&crc.to_le_bytes());
    tail.extend_from_slice(MAGIC);
    file.extend_from_slice(&tail);
    Ok((ContentId::of(&file), file))
}

/// Encodes a segment and publishes it, returning its id. Sealing the same
/// content again publishes nothing new.
pub fn seal_segment(
    backend: &dyn StorageBackend,
    records: &[Record],
    edges: &[Edge],
    config: &Config,
) -> Result<ContentId> {
    seal_segment_in(backend, Kind::Segment, records, edges, config)
}

/// Like [`seal_segment`], into another kind of storage: a writer's spool
/// holds segments that are not yet part of the database.
pub fn seal_segment_in(
    backend: &dyn StorageBackend,
    kind: Kind,
    records: &[Record],
    edges: &[Edge],
    config: &Config,
) -> Result<ContentId> {
    let (id, bytes) = encode_segment(records, edges, config)?;
    backend.write_once(&segment_key(kind, &id)?, &bytes)?;
    Ok(id)
}

/// An open segment: its directory is read, its blocks on demand.
pub struct Segment {
    backend: Arc<dyn StorageBackend>,
    kind: Kind,
    id: ContentId,
    info: SegmentInfo,
    blocks_read: Option<Arc<std::sync::atomic::AtomicU64>>,
}

impl Segment {
    /// Opens a segment, checking its tail and footer. Blocks are checked as
    /// they are read.
    pub fn open(backend: Arc<dyn StorageBackend>, id: ContentId) -> Result<Self> {
        Self::open_in(backend, Kind::Segment, id)
    }

    /// Opens a segment held in another kind of storage, such as a spool.
    pub fn open_in(backend: Arc<dyn StorageBackend>, kind: Kind, id: ContentId) -> Result<Self> {
        let key = segment_key(kind, &id)?;
        let what = format!("segment {id}");
        let len = backend.len(&key)?;
        if len < (MAGIC.len() + TAIL) as u64 {
            return Err(Error::corrupt(what, "shorter than its tail"));
        }
        let tail = backend.read_range(&key, len - TAIL as u64, TAIL)?;
        if &tail[TAIL - 8..] != MAGIC {
            return Err(Error::corrupt(what, "tail magic is wrong"));
        }
        if crc32fast::hash(&tail[..12])
            != u32::from_le_bytes(tail[12..16].try_into().unwrap_or([0; 4]))
        {
            return Err(Error::corrupt(what, "tail checksum mismatch"));
        }
        let footer_offset = u64::from_le_bytes(tail[0..8].try_into().unwrap_or([0; 8]));
        let footer_len = u64::from(u32::from_le_bytes(tail[8..12].try_into().unwrap_or([0; 4])));
        if footer_offset + footer_len + TAIL as u64 != len {
            return Err(Error::corrupt(
                what,
                "footer does not end where the tail begins",
            ));
        }
        let footer = unframe(
            &what,
            &backend.read_range(&key, footer_offset, footer_len as usize)?,
        )?;
        let info = serde_json::from_slice(&footer).map_err(|source| Error::Decode {
            what: format!("footer of {what}"),
            source,
        })?;
        Ok(Self {
            backend,
            kind,
            id,
            info,
            blocks_read: None,
        })
    }

    /// Counts every block this segment reads into `counter`.
    pub fn count_blocks_in(&mut self, counter: Arc<std::sync::atomic::AtomicU64>) {
        self.blocks_read = Some(counter);
    }

    /// The segment's content id.
    pub fn id(&self) -> ContentId {
        self.id
    }

    /// The reference a manifest uses to add or remove this segment.
    pub fn object_ref(&self) -> Result<crate::manifest::ObjectRef> {
        Ok(crate::manifest::ObjectRef {
            kind: crate::manifest::ObjectKind::Segment,
            id: self.id,
            bytes: self.backend.len(&segment_key(self.kind, &self.id)?)?,
            records: self.info.records,
        })
    }

    /// The segment's directory.
    pub fn info(&self) -> &SegmentInfo {
        &self.info
    }

    fn raw(&self, what: &str, block: &BlockInfo) -> Result<Vec<u8>> {
        let key = segment_key(self.kind, &self.id)?;
        let bytes = self
            .backend
            .read_range(&key, block.offset, block.len as usize)?;
        unframe(&format!("{what} of segment {}", self.id), &bytes)
    }

    /// Reads and decodes record block `index`.
    pub fn read_block(&self, index: usize) -> Result<Block> {
        let info = self.info.blocks.get(index).ok_or_else(|| Error::NotFound {
            what: format!("block {index} of segment {}", self.id),
        })?;
        if let Some(counter) = &self.blocks_read {
            counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Block::decode(&self.raw(&format!("block {index}"), info)?)
    }

    /// Every record, in stored order.
    pub fn records(&self) -> Result<Vec<Record>> {
        let mut records = Vec::with_capacity((self.info.records as usize).min(1 << 20));
        for index in 0..self.info.blocks.len() {
            let block = self.read_block(index)?;
            for i in 0..block.len() {
                records.push(block.record(i)?);
            }
        }
        Ok(records)
    }

    /// Every edge, in stored order.
    pub fn edges(&self) -> Result<Vec<Edge>> {
        let mut edges = Vec::with_capacity((self.info.edges as usize).min(1 << 20));
        for (index, info) in self.info.edge_blocks.iter().enumerate() {
            edges.extend(decode_edges(
                &self.raw(&format!("edge block {index}"), info)?,
            )?);
        }
        Ok(edges)
    }

    /// Reads every byte and checks it against the segment's content id and
    /// every block against its checksum.
    pub fn verify(&self) -> Result<()> {
        let key = segment_key(self.kind, &self.id)?;
        let bytes = self.backend.read(&key)?;
        if ContentId::of(&bytes) != self.id {
            return Err(Error::corrupt(
                format!("segment {}", self.id),
                "bytes do not match the content id",
            ));
        }
        for index in 0..self.info.blocks.len() {
            self.read_block(index)?;
        }
        self.edges()?;
        Ok(())
    }
}
