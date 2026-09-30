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
//! ```text
//! <root>/sources/
//!   blobs/<hex>            one content, its raw bytes; written once, shared
//!                          by every source part with that digest
//!   objects/<hex>.json     one source, its canonical form; written once
//! ```
//!
//! A blob's name is the digest of its bytes, and every read checks it. A
//! source object's name is its [`SourceId`], which addresses its origin and
//! parts but not its capture time, so a read decodes it and recomputes the
//! address instead of hashing the file. [`SourceStore::put_source`] writes
//! the blobs before the object, so an object never names content the store
//! lacks; a crash between the two leaves unreferenced blobs, which the next
//! capture of the same content reuses.

use std::fs;
use std::path::PathBuf;

use crate::digest::Digest;
use crate::error::StoreError;
use crate::error::{decode, io, read_verified};
use crate::experience::Span;
use crate::source::{CapturedSource, Source, SourceId};
use crate::{write_once, StateRoot};

/// The source store under one state root.
#[derive(Clone, Debug)]
pub struct SourceStore {
    dir: PathBuf,
}

impl SourceStore {
    /// The store under `root`. Nothing is created until something is
    /// written.
    #[must_use]
    pub fn open(root: &StateRoot) -> Self {
        Self {
            dir: root.sources(),
        }
    }

    fn blob(&self, digest: &Digest) -> PathBuf {
        self.dir.join("blobs").join(digest.hex())
    }

    fn object(&self, id: &SourceId) -> PathBuf {
        self.dir.join("objects").join(format!("{}.json", id.hex()))
    }

    /// Stores `captured` and returns its id. Write-once: content already
    /// stored is not written again, and a source already stored keeps its
    /// first capture time. An existing blob or object that no longer
    /// matches its address is reported as corrupt rather than replaced.
    pub fn put_source(&self, captured: &CapturedSource) -> Result<SourceId, StoreError> {
        let source = captured.source();
        source.validate()?;
        for part in &source.parts {
            let bytes = captured
                .content(&part.name)
                .ok_or_else(|| StoreError::UnknownPart {
                    source_id: source.id.clone(),
                    part: part.name.clone(),
                })?;
            let path = self.blob(&part.content);
            if !write_once(&path, bytes).map_err(io(&path))? {
                read_verified(&path, &part.content)?;
            }
        }
        let bytes = source.canonical()?;
        let path = self.object(&source.id);
        if !write_once(&path, &bytes).map_err(io(&path))? {
            self.get_source(&source.id)?;
        }
        Ok(source.id.clone())
    }

    /// Whether the store holds `id` (without verifying it).
    #[must_use]
    pub fn contains(&self, id: &SourceId) -> bool {
        self.object(id).is_file()
    }

    /// The source stored under `id`, verified against its address.
    pub fn get_source(&self, id: &SourceId) -> Result<Source, StoreError> {
        let path = self.object(id);
        if !path.is_file() {
            return Err(StoreError::UnknownSource(id.clone()));
        }
        let bytes = fs::read(&path).map_err(io(&path))?;
        let source: Source = decode(&path, &bytes)?;
        if source.id != *id {
            return Err(StoreError::Corrupt {
                path,
                expected: id.0.clone(),
                found: source.id.0,
            });
        }
        source.validate()?;
        // As for an experience: the value handed back must be the record
        // stored, not what survived decoding it.
        if source.canonical()? != bytes {
            return Err(StoreError::Undecodable {
                path,
                reason: "it does not re-encode to its stored form".into(),
            });
        }
        Ok(source)
    }

    /// The content stored under `digest`, verified.
    pub fn read_blob(&self, digest: &Digest) -> Result<Vec<u8>, StoreError> {
        let path = self.blob(digest);
        if !path.is_file() {
            return Err(StoreError::UnknownBlob(digest.clone()));
        }
        read_verified(&path, digest)
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
        let dir = self.dir.join("objects");
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(io(&dir)(e)),
        };
        let mut ids = Vec::new();
        for entry in entries {
            let entry = entry.map_err(io(&dir))?;
            let name = entry.file_name();
            // Anything but `<64 hex>.json` (a write-once temporary file) is
            // not a stored source.
            let Some(hex) = name.to_str().and_then(|n| n.strip_suffix(".json")) else {
                continue;
            };
            if let Ok(digest) = Digest::parse(&format!("sha256:{hex}")) {
                ids.push(SourceId(digest));
            }
        }
        ids.sort();
        Ok(ids)
    }
}
