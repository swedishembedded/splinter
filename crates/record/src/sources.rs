// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable, content-addressed source stores
// that every training example traces back to, for its clients. If your team
// needs expertise in training-data lineage or crash-safe storage, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The source store: sources under their address, and their parts' content
//! stored once per digest.
//!
//! A source is an entity keyed by its [`SourceId`], which addresses its origin
//! and parts but not its capture time, so a read decodes it and recomputes the
//! address. Each part's bytes are stored once per digest in the blob store and
//! the source entity names them; [`SourceStore::put_source`] stores both in one
//! commit, so a source never names content the store lacks.

use splinter_expdb::model::Entity;

use crate::error::StoreError;
use crate::workspace::{content_id, put_spilling, Workspace};
use splinter_core::digest::Digest;
use splinter_core::experience::Span;
use splinter_core::source::{CapturedSource, Source, SourceId};

const SOURCE: &str = "source";

/// The source store: sources, and their parts' content stored once per
/// digest.
#[derive(Clone, Debug)]
pub struct SourceStore {
    workspace: Workspace,
}

impl SourceStore {
    /// The store over `workspace`.
    #[must_use]
    pub fn new(workspace: &Workspace) -> Self {
        Self {
            workspace: workspace.clone(),
        }
    }

    /// Stores `captured` and returns its id. Content already stored is not
    /// written again, and a source already stored keeps its first capture
    /// time. One already stored that no longer matches its address is
    /// reported rather than replaced.
    pub fn put_source(&self, captured: &CapturedSource) -> Result<SourceId, StoreError> {
        let source = captured.source();
        source.validate()?;
        if self.contains(&source.id)? {
            self.get_source(&source.id)?;
            return Ok(source.id.clone());
        }
        let value = serde_json::to_value(source).map_err(|source| StoreError::Serialize {
            what: "source",
            source,
        })?;
        let mut parts = Vec::with_capacity(source.parts.len());
        for part in &source.parts {
            let bytes = captured
                .content(&part.name)
                .ok_or_else(|| StoreError::UnknownPart {
                    source_id: source.id.clone(),
                    part: part.name.clone(),
                })?;
            parts.push((part.content.clone(), bytes));
        }
        let key = content_id(&source.id.0)?;
        self.workspace.write(|s| {
            let mut blobs = Vec::with_capacity(parts.len());
            for (_, bytes) in &parts {
                blobs.push(s.put_blob(bytes)?);
            }
            put_spilling(
                s,
                Entity::keyed(SOURCE, key, value.clone()).with_blobs(blobs),
            )
            .map(|_| ())
        })?;
        Ok(source.id.clone())
    }

    /// Whether the store holds `id` (without verifying it).
    pub fn contains(&self, id: &SourceId) -> Result<bool, StoreError> {
        self.workspace.has(SOURCE, &id.0)
    }

    /// The source stored under `id`, verified against its address.
    pub fn get_source(&self, id: &SourceId) -> Result<Source, StoreError> {
        let entity = self
            .workspace
            .find(SOURCE, &id.0)?
            .ok_or_else(|| StoreError::UnknownSource(id.clone()))?;
        let source: Source = serde_json::from_value(entity.value.clone()).map_err(|e| {
            StoreError::UndecodableObject {
                what: format!("source {id}"),
                reason: e.to_string(),
            }
        })?;
        if source.id != *id {
            return Err(StoreError::Altered {
                what: format!("source {id}"),
                expected: id.0.clone(),
                found: source.id.0,
            });
        }
        source.validate()?;
        // The value handed back must be the record stored, not what
        // survived decoding it.
        let stored = splinter_core::digest::canonical_json(&entity.value).map_err(|source| {
            StoreError::Serialize {
                what: "source",
                source,
            }
        })?;
        if source.canonical()? != stored {
            return Err(StoreError::UndecodableObject {
                what: format!("source {id}"),
                reason: "it does not re-encode to its stored form".into(),
            });
        }
        Ok(source)
    }

    /// The content stored under `digest`, verified.
    pub fn read_blob(&self, digest: &Digest) -> Result<Vec<u8>, StoreError> {
        let Some(id) = crate::address::content_id(digest) else {
            return Err(StoreError::UnknownBlob(digest.clone()));
        };
        match self.workspace.read(|s| s.read_blob(&id)) {
            Ok(bytes) => Ok(bytes),
            Err(StoreError::Database(splinter_expdb::Error::NotFound { .. })) => {
                Err(StoreError::UnknownBlob(digest.clone()))
            }
            Err(other) => Err(other),
        }
    }

    /// The content of part `name` of source `id`, verified.
    pub fn read_part(&self, id: &SourceId, name: &str) -> Result<Vec<u8>, StoreError> {
        let source = self.get_source(id)?;
        let part = source.part(name).ok_or_else(|| StoreError::UnknownPart {
            source_id: id.clone(),
            part: name.to_string(),
        })?;
        self.read_blob(&part.content)
    }

    /// The bytes `span` covers. A span that names a part must name one
    /// whose content is the content the span indexes; a span that names
    /// none resolves by its content digest alone.
    pub fn read_span(&self, span: &Span) -> Result<Vec<u8>, StoreError> {
        if let Some(part_ref) = &span.part {
            let source = self.get_source(&part_ref.source)?;
            let part = source
                .part(&part_ref.name)
                .ok_or_else(|| StoreError::UnknownPart {
                    source_id: part_ref.source.clone(),
                    part: part_ref.name.clone(),
                })?;
            if part.content != span.source {
                return Err(StoreError::SpanPart {
                    source_id: part_ref.source.clone(),
                    part: part_ref.name.clone(),
                    content: span.source.clone(),
                    actual: part.content.clone(),
                });
            }
        }
        let bytes = self.read_blob(&span.source)?;
        let len = bytes.len() as u64;
        if span.start > span.end || span.end > len {
            return Err(StoreError::SpanOutOfRange {
                content: span.source.clone(),
                start: span.start,
                end: span.end,
                len,
            });
        }
        // Both offsets are at most `len`, which came from a `usize`.
        Ok(bytes[span.start as usize..span.end as usize].to_vec())
    }

    /// Every stored source's id, in id order (without verifying them).
    pub fn list(&self) -> Result<Vec<SourceId>, StoreError> {
        Ok(self
            .workspace
            .ids_of(SOURCE)?
            .into_iter()
            .map(SourceId)
            .collect())
    }
}
