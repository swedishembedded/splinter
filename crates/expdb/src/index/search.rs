// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Semantic and text search over immutable shards.
//!
//! Embeddings are supplied by the caller: computing them is model work that
//! belongs to the model runtime, not to the store. A shard is written once;
//! queries search every live shard and merge, and shards are merged in the
//! background like index runs. Nothing is ever updated in place.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::database::Database;
use crate::error::{Error, Result};
use crate::format::frame::{frame, unframe};
use crate::id::{ContentId, RecordId};
use crate::manifest::{ObjectKind, ObjectRef, Snapshot};

const VECTOR_MAGIC: &[u8; 8] = b"EXPVEC01";
const TEXT_MAGIC: &[u8; 8] = b"EXPTXT01";
const SEARCH_JOB: &str = "search";

/// A shard of unit-length vectors of one model.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VectorShard {
    model: String,
    dims: usize,
    ids: Vec<RecordId>,
    data: Vec<f32>,
}

fn normalised(vector: &[f32]) -> Result<Vec<f32>> {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if vector.is_empty() || !norm.is_finite() || norm == 0.0 {
        return Err(Error::invalid(
            "vector",
            "it must be finite, non-empty and not all zero",
        ));
    }
    Ok(vector.iter().map(|x| x / norm).collect())
}

fn encode<T: Serialize>(
    magic: &[u8; 8],
    what: &'static str,
    value: &T,
    db: &Database,
) -> Result<Vec<u8>> {
    let json = serde_json::to_vec(value).map_err(|source| Error::Encode { what, source })?;
    let mut file = magic.to_vec();
    file.extend_from_slice(&frame(&json, db.config().compression)?);
    Ok(file)
}

fn decode<T: for<'de> Deserialize<'de>>(magic: &[u8; 8], what: &str, bytes: &[u8]) -> Result<T> {
    if bytes.len() < magic.len() || &bytes[..magic.len()] != magic {
        return Err(Error::corrupt(what, "magic is wrong"));
    }
    let json = unframe(what, &bytes[magic.len()..])?;
    serde_json::from_slice(&json).map_err(|source| Error::Decode {
        what: what.to_owned(),
        source,
    })
}

impl Database {
    fn store_shard(&self, kind: ObjectKind, bytes: Vec<u8>, records: u64) -> Result<ObjectRef> {
        let id = ContentId::of(&bytes);
        let object = ObjectRef {
            kind,
            id,
            bytes: bytes.len() as u64,
            records,
        };
        self.backend().write_once(&object.key()?, &bytes)?;
        Ok(object)
    }

    fn read_shard<T: for<'de> Deserialize<'de>>(
        &self,
        object: &ObjectRef,
        magic: &[u8; 8],
    ) -> Result<T> {
        let bytes = self.backend().read(&object.key()?)?;
        if ContentId::of(&bytes) != object.id {
            return Err(Error::corrupt(
                format!("shard {}", object.id),
                "bytes do not match the content id",
            ));
        }
        decode(magic, &format!("shard {}", object.id), &bytes)
    }

    /// Stores a shard of embeddings of one `model` for existing records and
    /// publishes it. Every vector of a model must have the same size.
    pub fn add_vectors(&self, model: &str, items: &[(RecordId, Vec<f32>)]) -> Result<ObjectRef> {
        let Some(first) = items.first() else {
            return Err(Error::invalid(
                "vector shard",
                "it must hold at least one vector",
            ));
        };
        let dims = first.1.len();
        if let Some(existing) = self.snapshot()?.vector_shards(model)?.first() {
            if existing.dims != dims {
                return Err(Error::invalid(
                    "vector",
                    format!(
                        "model `{model}` has {}-dimensional vectors, not {dims}",
                        existing.dims
                    ),
                ));
            }
        }
        let mut shard = VectorShard {
            model: model.to_owned(),
            dims,
            ids: Vec::new(),
            data: Vec::new(),
        };
        for (id, vector) in items {
            if vector.len() != dims {
                return Err(Error::invalid(
                    "vector",
                    format!("{id} has {} dimensions, expected {dims}", vector.len()),
                ));
            }
            shard.ids.push(*id);
            shard.data.extend(normalised(vector)?);
        }
        let object = self.store_shard(
            ObjectKind::Vector,
            encode(VECTOR_MAGIC, "vector shard", &shard, self)?,
            items.len() as u64,
        )?;
        self.publish_once(SEARCH_JOB, vec![object.clone()], Vec::new())?;
        Ok(object)
    }

    /// Merges the snapshot's shards of `model` into one. Returns its id, or
    /// `None` if there were fewer than two.
    pub fn compact_vectors(&self, model: &str) -> Result<Option<ContentId>> {
        let snapshot = self.snapshot()?;
        let old: Vec<ObjectRef> = snapshot
            .objects(ObjectKind::Vector)
            .into_iter()
            .filter(|o| {
                self.read_shard::<VectorShard>(o, VECTOR_MAGIC)
                    .is_ok_and(|s| s.model == model)
            })
            .collect();
        if old.len() < 2 {
            return Ok(None);
        }
        let mut merged: Option<VectorShard> = None;
        for object in &old {
            let shard: VectorShard = self.read_shard(object, VECTOR_MAGIC)?;
            match &mut merged {
                None => merged = Some(shard),
                Some(m) => {
                    m.ids.extend(shard.ids);
                    m.data.extend(shard.data);
                }
            }
        }
        let Some(merged) = merged else {
            return Ok(None);
        };
        let object = self.store_shard(
            ObjectKind::Vector,
            encode(VECTOR_MAGIC, "vector shard", &merged, self)?,
            merged.ids.len() as u64,
        )?;
        let removed = old.into_iter().filter(|o| o.id != object.id).collect();
        self.publish_once(SEARCH_JOB, vec![object.clone()], removed)?;
        Ok(Some(object.id))
    }

    /// Stores a shard of an inverted text index and publishes it.
    pub fn add_text(&self, items: &[(RecordId, String)]) -> Result<ObjectRef> {
        if items.is_empty() {
            return Err(Error::invalid(
                "text shard",
                "it must hold at least one text",
            ));
        }
        let mut postings: BTreeMap<String, Vec<(RecordId, u32)>> = BTreeMap::new();
        for (id, text) in items {
            let mut counts: BTreeMap<String, u32> = BTreeMap::new();
            for term in tokens(text) {
                *counts.entry(term).or_default() += 1;
            }
            for (term, count) in counts {
                postings.entry(term).or_default().push((*id, count));
            }
        }
        let object = self.store_shard(
            ObjectKind::Text,
            encode(TEXT_MAGIC, "text shard", &postings, self)?,
            items.len() as u64,
        )?;
        self.publish_once(SEARCH_JOB, vec![object.clone()], Vec::new())?;
        Ok(object)
    }
}

/// Lowercase alphanumeric words.
fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

impl Snapshot {
    fn vector_shards(&self, model: &str) -> Result<Vec<VectorShard>> {
        let mut shards = Vec::new();
        for object in self.objects(ObjectKind::Vector) {
            let shard: VectorShard = self.database().read_shard(&object, VECTOR_MAGIC)?;
            if shard.model == model {
                shards.push(shard);
            }
        }
        Ok(shards)
    }

    /// The `k` records whose embeddings are most similar to `query` by
    /// cosine, best first, searching every shard of `model` exactly. Vectors
    /// of records that are not in the snapshot are never returned.
    pub fn search_vectors(
        &self,
        model: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(RecordId, f32)>> {
        let query = normalised(query)?;
        let index = self.index()?;
        let mut best: HashMap<RecordId, f32> = HashMap::new();
        for shard in self.vector_shards(model)? {
            if shard.dims != query.len() {
                return Err(Error::invalid(
                    "query vector",
                    format!(
                        "it has {} dimensions, model `{model}` has {}",
                        query.len(),
                        shard.dims
                    ),
                ));
            }
            for (id, vector) in shard.ids.iter().zip(shard.data.chunks_exact(shard.dims)) {
                if index.contains(*id) {
                    let score: f32 = vector.iter().zip(&query).map(|(a, b)| a * b).sum();
                    let entry = best.entry(*id).or_insert(score);
                    *entry = entry.max(score);
                }
            }
        }
        let mut hits: Vec<(RecordId, f32)> = best.into_iter().collect();
        hits.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        hits.truncate(k);
        Ok(hits)
    }

    /// The `k` records whose text best matches the words of `query`: each
    /// matching word counts once per occurrence. Records not in the snapshot
    /// are never returned.
    pub fn search_text(&self, query: &str, k: usize) -> Result<Vec<(RecordId, u32)>> {
        let mut terms = tokens(query);
        terms.sort();
        terms.dedup();
        let index = self.index()?;
        let mut scores: HashMap<RecordId, u32> = HashMap::new();
        for object in self.objects(ObjectKind::Text) {
            let postings: BTreeMap<String, Vec<(RecordId, u32)>> =
                self.database().read_shard(&object, TEXT_MAGIC)?;
            for term in &terms {
                for (id, count) in postings.get(term).into_iter().flatten() {
                    if index.contains(*id) {
                        *scores.entry(*id).or_default() += count;
                    }
                }
            }
        }
        let mut hits: Vec<(RecordId, u32)> = scores.into_iter().collect();
        hits.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        hits.truncate(k);
        Ok(hits)
    }
}
