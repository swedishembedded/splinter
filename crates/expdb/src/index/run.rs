// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A run: the sorted table of where records live, as stored in a file.

use crate::backend::{Key, Kind};
use crate::config::Compression;
use crate::error::{Error, Result};
use crate::format::frame::{frame, unframe};
use crate::format::Segment;
use crate::id::{ContentId, RecordId, WriterId};
use crate::model::{Edge, Rel};

const MAGIC: &[u8; 8] = b"EXPIDX01";
const ROW: usize = 16 + 4 + 4 + 4 + 1 + 1 + 16 + 16 + 32 + 32 + 32 + 8 + 8;
const EDGE: usize = 16 + 16 + 1 + 4;

const HAS_PARENT: u8 = 1;
const HAS_ATTEMPT: u8 = 2;
const HAS_FAMILY: u8 = 4;
const HAS_INSTANCE: u8 = 8;
const HAS_ENTITY: u8 = 16;

/// One record's place and neighbours.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Row {
    pub(crate) id: RecordId,
    /// Index into the run's `covers`.
    pub(crate) seg: u32,
    pub(crate) block: u32,
    pub(crate) row: u32,
    pub(crate) kind: u8,
    /// When the record was stamped.
    pub(crate) timestamp_ns: u64,
    pub(crate) parent: Option<RecordId>,
    pub(crate) attempt: Option<RecordId>,
    pub(crate) family: Option<ContentId>,
    pub(crate) instance: Option<ContentId>,
    /// The entity the record defines and its class hash.
    pub(crate) entity: Option<(ContentId, u64)>,
}

/// An edge and the segment (index into `covers`) that holds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EdgeRow {
    pub(crate) edge: Edge,
    pub(crate) seg: u32,
}

/// A run over some segments.
#[derive(Debug, Clone, Default)]
pub(crate) struct Run {
    pub(crate) covers: Vec<ContentId>,
    pub(crate) rows: Vec<Row>,
    pub(crate) edges: Vec<EdgeRow>,
}

fn put_id(out: &mut Vec<u8>, id: Option<RecordId>) {
    out.extend_from_slice(&id.map_or(0, |i| i.writer().raw()).to_le_bytes());
    out.extend_from_slice(&id.map_or(0, RecordId::sequence).to_le_bytes());
}

fn put_content(out: &mut Vec<u8>, id: Option<ContentId>) {
    out.extend_from_slice(id.unwrap_or(ContentId::from_bytes([0; 32])).as_bytes());
}

struct Cursor<'a>(&'a [u8], usize);

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.1.checked_add(n).filter(|e| *e <= self.0.len());
        let end = end.ok_or_else(|| Error::corrupt("index run", "truncated"))?;
        let slice = &self.0[self.1..end];
        self.1 = end;
        Ok(slice)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().unwrap_or([0; 4]),
        ))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().unwrap_or([0; 8]),
        ))
    }
    fn id(&mut self) -> Result<RecordId> {
        let (w, s) = (self.u64()?, self.u64()?);
        Ok(RecordId::new(WriterId::from_raw(w), s))
    }
    fn content(&mut self) -> Result<ContentId> {
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(self.take(32)?);
        Ok(ContentId::from_bytes(bytes))
    }
}

impl Run {
    /// Sorts rows and edges so equal content encodes to equal bytes.
    pub(crate) fn normalise(&mut self) {
        self.rows.sort_by_key(|r| r.id);
        self.rows.dedup_by_key(|r| r.id);
        self.edges.sort_by_key(|e| (e.edge, e.seg));
        self.edges.dedup_by_key(|e| e.edge);
    }

    pub(crate) fn encode(&self, compression: Compression) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.covers.len() as u32).to_le_bytes());
        for cover in &self.covers {
            out.extend_from_slice(cover.as_bytes());
        }
        out.extend_from_slice(&(self.rows.len() as u32).to_le_bytes());
        for r in &self.rows {
            put_id(&mut out, Some(r.id));
            out.extend_from_slice(&r.seg.to_le_bytes());
            out.extend_from_slice(&r.block.to_le_bytes());
            out.extend_from_slice(&r.row.to_le_bytes());
            out.push(r.kind);
            out.extend_from_slice(&r.timestamp_ns.to_le_bytes());
            let flag = |present: bool, bit: u8| if present { bit } else { 0 };
            out.push(
                flag(r.parent.is_some(), HAS_PARENT)
                    | flag(r.attempt.is_some(), HAS_ATTEMPT)
                    | flag(r.family.is_some(), HAS_FAMILY)
                    | flag(r.instance.is_some(), HAS_INSTANCE)
                    | flag(r.entity.is_some(), HAS_ENTITY),
            );
            put_id(&mut out, r.parent);
            put_id(&mut out, r.attempt);
            put_content(&mut out, r.family);
            put_content(&mut out, r.instance);
            put_content(&mut out, r.entity.map(|(id, _)| id));
            out.extend_from_slice(&r.entity.map_or(0, |(_, class)| class).to_le_bytes());
        }
        out.extend_from_slice(&(self.edges.len() as u32).to_le_bytes());
        for e in &self.edges {
            put_id(&mut out, Some(e.edge.from));
            put_id(&mut out, Some(e.edge.to));
            out.push(e.edge.rel.code());
            out.extend_from_slice(&e.seg.to_le_bytes());
        }
        let mut file = MAGIC.to_vec();
        file.extend_from_slice(&frame(&out, compression)?);
        Ok(file)
    }

    pub(crate) fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < MAGIC.len() || &bytes[..MAGIC.len()] != MAGIC {
            return Err(Error::corrupt("index run", "magic is wrong"));
        }
        let payload = unframe("index run", &bytes[MAGIC.len()..])?;
        let mut c = Cursor(&payload, 0);
        let covers = (0..c.u32()?)
            .map(|_| c.content())
            .collect::<Result<Vec<_>>>()?;
        let n_rows = c.u32()? as usize;
        if n_rows.checked_mul(ROW).is_none_or(|n| n > payload.len()) {
            return Err(Error::corrupt("index run", "row count exceeds the file"));
        }
        let mut rows = Vec::with_capacity(n_rows);
        for _ in 0..n_rows {
            let id = c.id()?;
            let (seg, block, row) = (c.u32()?, c.u32()?, c.u32()?);
            let kind = c.take(1)?[0];
            let timestamp_ns = c.u64()?;
            let flags = c.take(1)?[0];
            let (parent, attempt) = (c.id()?, c.id()?);
            let (family, instance) = (c.content()?, c.content()?);
            let (entity, class) = (c.content()?, c.u64()?);
            rows.push(Row {
                id,
                seg,
                block,
                row,
                kind,
                timestamp_ns,
                parent: (flags & HAS_PARENT != 0).then_some(parent),
                attempt: (flags & HAS_ATTEMPT != 0).then_some(attempt),
                family: (flags & HAS_FAMILY != 0).then_some(family),
                instance: (flags & HAS_INSTANCE != 0).then_some(instance),
                entity: (flags & HAS_ENTITY != 0).then_some((entity, class)),
            });
        }
        let n_edges = c.u32()? as usize;
        if n_edges.checked_mul(EDGE).is_none_or(|n| n > payload.len()) {
            return Err(Error::corrupt("index run", "edge count exceeds the file"));
        }
        let mut edges = Vec::with_capacity(n_edges);
        for _ in 0..n_edges {
            let (from, to) = (c.id()?, c.id()?);
            let rel = Rel::from_code(c.take(1)?[0])
                .ok_or_else(|| Error::corrupt("index run", "unknown relation"))?;
            edges.push(EdgeRow {
                edge: Edge { from, rel, to },
                seg: c.u32()?,
            });
        }
        Ok(Self {
            covers,
            rows,
            edges,
        })
    }

    /// Reads a run stored in the index kind.
    pub(crate) fn load(
        backend: &dyn crate::backend::StorageBackend,
        id: &ContentId,
    ) -> Result<Self> {
        let bytes = backend.read(&Key::new(Kind::Index, &format!("{id}.idx"))?)?;
        if ContentId::of(&bytes) != *id {
            return Err(Error::corrupt(
                format!("index run {id}"),
                "bytes do not match the content id",
            ));
        }
        Self::decode(&bytes)
    }

    /// The run for segments read from storage, scanning envelope columns
    /// only: no record body is parsed.
    pub(crate) fn scan(segments: &[std::sync::Arc<Segment>]) -> Result<Self> {
        let mut run = Run::default();
        for (seg, segment) in segments.iter().enumerate() {
            run.covers.push(segment.id());
            for index in 0..segment.info().blocks.len() {
                let block = segment.read_block(index)?;
                for row in 0..block.len() {
                    run.rows.push(Row {
                        id: block.id(row),
                        seg: seg as u32,
                        block: index as u32,
                        row: row as u32,
                        kind: block.kind(row)?.code(),
                        timestamp_ns: block.timestamp_ns(row),
                        parent: block.parent(row),
                        attempt: block.attempt(row),
                        family: block.family(row),
                        instance: block.task_instance(row),
                        entity: block
                            .entity(row)
                            .map(|id| (id, block.entity_class(row).unwrap_or(0))),
                    });
                }
            }
            run.edges
                .extend(segment.edges()?.into_iter().map(|edge| EdgeRow {
                    edge,
                    seg: seg as u32,
                }));
        }
        run.normalise();
        Ok(run)
    }

    /// One run covering what several runs cover, restricted to the segments
    /// in `live`.
    pub(crate) fn merge(runs: &[Run], live: &std::collections::HashSet<ContentId>) -> Self {
        let mut merged = Run::default();
        let mut slot = std::collections::HashMap::new();
        for run in runs {
            let map: Vec<Option<u32>> = run
                .covers
                .iter()
                .map(|cover| {
                    live.contains(cover).then(|| {
                        *slot.entry(*cover).or_insert_with(|| {
                            merged.covers.push(*cover);
                            (merged.covers.len() - 1) as u32
                        })
                    })
                })
                .collect();
            for row in &run.rows {
                if let Some(Some(seg)) = map.get(row.seg as usize) {
                    merged.rows.push(Row {
                        seg: *seg,
                        ..row.clone()
                    });
                }
            }
            for edge in &run.edges {
                if let Some(Some(seg)) = map.get(edge.seg as usize) {
                    merged.edges.push(EdgeRow { seg: *seg, ..*edge });
                }
            }
        }
        merged.normalise();
        merged
    }
}

/// The segments a stored run covers.
pub(crate) fn run_covers(
    backend: &dyn crate::backend::StorageBackend,
    id: &ContentId,
) -> Result<Vec<ContentId>> {
    Ok(Run::load(backend, id)?.covers)
}
