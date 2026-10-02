// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Column encoding of record blocks and edge blocks.
//!
//! A record block stores each envelope field as a fixed-width column, then
//! the bodies as one offset-indexed byte column, so a reader can filter on
//! the envelope without parsing a single body.

use crate::error::{Error, Result};
use crate::id::{ContentId, RecordId, WriterId};
use crate::model::{Body, Edge, Record, RecordKind, Rel};

const HAS_PARENT: u8 = 1;
const HAS_ATTEMPT: u8 = 2;
const HAS_FAMILY: u8 = 4;
const HAS_INSTANCE: u8 = 8;
const HAS_ENTITY: u8 = 16;

/// A number that stands for an entity class in the envelope, so entities of
/// one class can be listed without reading a body.
pub(crate) fn class_hash(class: &str) -> u64 {
    ContentId::of(class.as_bytes()).prefix_u64()
}

/// The content id and class hash of the entity a body defines, if it
/// defines one.
fn entity_of(body: &Body) -> Result<Option<(ContentId, u64)>> {
    let Some(id) = body.entity_id()? else {
        return Ok(None);
    };
    let class = match body {
        Body::Entity(entity) => class_hash(&entity.class),
        _ => 0,
    };
    Ok(Some((id, class)))
}

struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.at.checked_add(n).filter(|end| *end <= self.buf.len());
        let end = end.ok_or_else(|| {
            Error::corrupt("segment block", "column runs past the end of the block")
        })?;
        let slice = &self.buf[self.at..end];
        self.at = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().unwrap_or([0; 4]),
        ))
    }

    fn u64s(&mut self, n: usize) -> Result<Vec<u64>> {
        Ok(self
            .take(n * 8)?
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| u64::from_le_bytes(*b))
            .collect())
    }

    fn u16s(&mut self, n: usize) -> Result<Vec<u16>> {
        Ok(self
            .take(n * 2)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes(*b))
            .collect())
    }

    fn ids(&mut self, n: usize) -> Result<Vec<RecordId>> {
        let writers = self.u64s(n)?;
        let seqs = self.u64s(n)?;
        Ok(writers
            .into_iter()
            .zip(seqs)
            .map(|(w, s)| RecordId::new(WriterId::from_raw(w), s))
            .collect())
    }

    fn contents_interleaved(&mut self, n: usize) -> Result<Vec<(ContentId, u64)>> {
        let raw = self.take(
            n.checked_mul(40)
                .ok_or_else(|| Error::corrupt("segment block", "entity column is too long"))?,
        )?;
        Ok(raw
            .as_chunks::<40>()
            .0
            .iter()
            .map(|b| {
                let mut id = [0u8; 32];
                id.copy_from_slice(&b[..32]);
                let class = u64::from_le_bytes(b[32..40].try_into().unwrap_or([0; 8]));
                (ContentId::from_bytes(id), class)
            })
            .collect())
    }

    fn contents(&mut self, n: usize) -> Result<Vec<ContentId>> {
        Ok(self
            .take(n * 32)?
            .as_chunks::<32>()
            .0
            .iter()
            .map(|b| ContentId::from_bytes(*b))
            .collect())
    }
}

fn put_u64s(out: &mut Vec<u8>, values: impl Iterator<Item = u64>) {
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn put_ids(out: &mut Vec<u8>, ids: &[Option<RecordId>]) {
    put_u64s(
        out,
        ids.iter().map(|id| id.map_or(0, |id| id.writer().raw())),
    );
    put_u64s(out, ids.iter().map(|id| id.map_or(0, RecordId::sequence)));
}

/// Encodes `records` as one block.
pub(crate) fn encode_records(records: &[Record]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(&(records.len() as u32).to_le_bytes());
    let own: Vec<_> = records.iter().map(|r| Some(r.id)).collect();
    put_ids(&mut out, &own);
    put_u64s(&mut out, records.iter().map(|r| r.timestamp_ns));
    out.extend(records.iter().map(|r| r.kind().code()));
    for record in records {
        out.extend_from_slice(&record.schema.to_le_bytes());
    }
    let entities: Vec<Option<(ContentId, u64)>> = records
        .iter()
        .map(|r| entity_of(&r.body))
        .collect::<Result<_>>()?;
    out.extend(records.iter().zip(&entities).map(|(r, entity)| {
        let flag = |present: bool, bit: u8| if present { bit } else { 0 };
        flag(r.parent.is_some(), HAS_PARENT)
            | flag(r.attempt.is_some(), HAS_ATTEMPT)
            | flag(r.family.is_some(), HAS_FAMILY)
            | flag(r.task_instance.is_some(), HAS_INSTANCE)
            | flag(entity.is_some(), HAS_ENTITY)
    }));
    put_ids(
        &mut out,
        &records.iter().map(|r| r.parent).collect::<Vec<_>>(),
    );
    put_ids(
        &mut out,
        &records.iter().map(|r| r.attempt).collect::<Vec<_>>(),
    );
    for field in [|r: &Record| r.family, |r: &Record| r.task_instance] {
        for record in records {
            out.extend_from_slice(
                field(record)
                    .unwrap_or(ContentId::from_bytes([0; 32]))
                    .as_bytes(),
            );
        }
    }
    let defined: Vec<(ContentId, u64)> = entities.into_iter().flatten().collect();
    out.extend_from_slice(&(defined.len() as u32).to_le_bytes());
    for (id, class) in &defined {
        out.extend_from_slice(id.as_bytes());
        out.extend_from_slice(&class.to_le_bytes());
    }
    let mut bodies = Vec::new();
    let mut offsets = vec![0u32];
    for record in records {
        serde_json::to_writer(&mut bodies, &record.body).map_err(|source| Error::Encode {
            what: "record body",
            source,
        })?;
        offsets.push(bodies.len() as u32);
    }
    for offset in offsets {
        out.extend_from_slice(&offset.to_le_bytes());
    }
    out.extend_from_slice(&bodies);
    Ok(out)
}

/// One decoded record block: envelope columns, with bodies parsed on demand.
#[derive(Debug)]
pub struct Block {
    ids: Vec<RecordId>,
    timestamps: Vec<u64>,
    kinds: Vec<u8>,
    schemas: Vec<u16>,
    parents: Vec<Option<RecordId>>,
    attempts: Vec<Option<RecordId>>,
    families: Vec<Option<ContentId>>,
    instances: Vec<Option<ContentId>>,
    entities: Vec<Option<(ContentId, u64)>>,
    offsets: Vec<u32>,
    bodies: Vec<u8>,
}

impl Block {
    pub(crate) fn decode(raw: &[u8]) -> Result<Self> {
        let mut r = Reader { buf: raw, at: 0 };
        let n = r.u32()? as usize;
        let ids = r.ids(n)?;
        let timestamps = r.u64s(n)?;
        let kinds = r.take(n)?.to_vec();
        let schemas = r.u16s(n)?;
        let flags = r.take(n)?.to_vec();
        let parent_ids = r.ids(n)?;
        let attempt_ids = r.ids(n)?;
        let family_ids = r.contents(n)?;
        let instance_ids = r.contents(n)?;
        let n_entities = r.u32()? as usize;
        let defined = r.contents_interleaved(n_entities)?;
        let offsets: Vec<u32> = (0..=n).map(|_| r.u32()).collect::<Result<_>>()?;
        let bodies = raw[r.at..].to_vec();
        let end = offsets.last().copied().unwrap_or(0) as usize;
        if end != bodies.len() || offsets.windows(2).any(|w| w[0] > w[1]) {
            return Err(Error::corrupt(
                "segment block",
                "body offsets do not match the body column",
            ));
        }
        let pick = |flag: u8, i: usize| flags[i] & flag != 0;
        let mut defined = defined.into_iter();
        let entities: Vec<Option<(ContentId, u64)>> = (0..n)
            .map(|i| {
                if pick(HAS_ENTITY, i) {
                    defined.next()
                } else {
                    None
                }
            })
            .collect();
        if defined.next().is_some() {
            return Err(Error::corrupt(
                "segment block",
                "more entity ids than records that define one",
            ));
        }
        Ok(Self {
            ids,
            timestamps,
            kinds,
            schemas,
            parents: (0..n)
                .map(|i| pick(HAS_PARENT, i).then_some(parent_ids[i]))
                .collect(),
            attempts: (0..n)
                .map(|i| pick(HAS_ATTEMPT, i).then_some(attempt_ids[i]))
                .collect(),
            families: (0..n)
                .map(|i| pick(HAS_FAMILY, i).then_some(family_ids[i]))
                .collect(),
            instances: (0..n)
                .map(|i| pick(HAS_INSTANCE, i).then_some(instance_ids[i]))
                .collect(),
            entities,
            offsets,
            bodies,
        })
    }

    /// How many records the block holds.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the block is empty.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The id of record `i`.
    pub fn id(&self, i: usize) -> RecordId {
        self.ids[i]
    }

    /// The timestamp of record `i`.
    pub fn timestamp_ns(&self, i: usize) -> u64 {
        self.timestamps[i]
    }

    /// The kind of record `i`.
    pub fn kind(&self, i: usize) -> Result<RecordKind> {
        RecordKind::from_code(self.kinds[i])
            .ok_or_else(|| Error::corrupt("segment block", "unknown record kind"))
    }

    /// The parent of record `i`.
    pub fn parent(&self, i: usize) -> Option<RecordId> {
        self.parents[i]
    }

    /// The attempt of record `i`.
    pub fn attempt(&self, i: usize) -> Option<RecordId> {
        self.attempts[i]
    }

    /// The family of record `i`.
    pub fn family(&self, i: usize) -> Option<ContentId> {
        self.families[i]
    }

    /// The task instance of record `i`.
    pub fn task_instance(&self, i: usize) -> Option<ContentId> {
        self.instances[i]
    }

    /// The content id of the entity record `i` defines, without reading its
    /// body.
    pub fn entity(&self, i: usize) -> Option<ContentId> {
        self.entities[i].map(|(id, _)| id)
    }

    /// The class hash of the application entity record `i` defines; zero for
    /// the other kinds that define one.
    pub fn entity_class(&self, i: usize) -> Option<u64> {
        self.entities[i].map(|(_, class)| class)
    }

    /// The whole of record `i`, its body parsed now.
    pub fn record(&self, i: usize) -> Result<Record> {
        let bytes = &self.bodies[self.offsets[i] as usize..self.offsets[i + 1] as usize];
        let body: Body = serde_json::from_slice(bytes).map_err(|source| Error::Decode {
            what: format!("body of record {}", self.ids[i]),
            source,
        })?;
        if body.kind().code() != self.kinds[i] {
            return Err(Error::corrupt(
                format!("record {}", self.ids[i]),
                "body does not match the stored kind",
            ));
        }
        Ok(Record {
            id: self.ids[i],
            timestamp_ns: self.timestamps[i],
            parent: self.parents[i],
            attempt: self.attempts[i],
            family: self.families[i],
            task_instance: self.instances[i],
            schema: self.schemas[i],
            body,
        })
    }
}

/// Encodes `edges` as one block.
pub(crate) fn encode_edges(edges: &[Edge]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(edges.len() as u32).to_le_bytes());
    let from: Vec<_> = edges.iter().map(|e| Some(e.from)).collect();
    let to: Vec<_> = edges.iter().map(|e| Some(e.to)).collect();
    put_ids(&mut out, &from);
    put_ids(&mut out, &to);
    out.extend(edges.iter().map(|e| e.rel.code()));
    out
}

/// Decodes an edge block.
pub(crate) fn decode_edges(raw: &[u8]) -> Result<Vec<Edge>> {
    let mut r = Reader { buf: raw, at: 0 };
    let n = r.u32()? as usize;
    let from = r.ids(n)?;
    let to = r.ids(n)?;
    let rels = r.take(n)?;
    from.into_iter()
        .zip(to)
        .zip(rels)
        .map(|((from, to), code)| {
            let rel = Rel::from_code(*code)
                .ok_or_else(|| Error::corrupt("edge block", "unknown relation"))?;
            Ok(Edge { from, rel, to })
        })
        .collect()
}
