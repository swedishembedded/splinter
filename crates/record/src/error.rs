// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The error every content-addressed store in this crate reports, and the
//! verified read they share: an object is only ever handed back after its
//! bytes were checked against the address it is stored under.

use std::path::PathBuf;

use crate::digest::Digest;
use crate::experience::{ExperienceError, ExperienceId};
use crate::experiences::SetId;
use crate::source::{SourceError, SourceId};
use crate::tasks::TaskSetId;

/// Why a store operation failed.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// The experience is not valid.
    #[error("invalid experience: {0}")]
    Invalid(#[from] ExperienceError),
    /// The source is not valid.
    #[error("invalid source: {0}")]
    InvalidSource(#[from] SourceError),
    /// An annotation or set is not valid.
    #[error("invalid {what}: {reason}")]
    Rejected {
        /// What was refused.
        what: &'static str,
        /// Why.
        reason: String,
    },
    /// The store holds no experience with this id.
    #[error("no experience {0} in the store")]
    UnknownExperience(ExperienceId),
    /// The store holds no set with this id.
    #[error("no experience set {0} in the store")]
    UnknownSet(SetId),
    /// The store holds no source with this id.
    #[error("no source {0} in the store")]
    UnknownSource(SourceId),
    /// The store holds no task with this id.
    #[error("no task {0} in the store")]
    UnknownTask(Digest),
    /// The store holds no task set with this id.
    #[error("no task set {0} in the store")]
    UnknownTaskSet(TaskSetId),
    /// No run with this id was recorded.
    #[error("no run {0} is recorded")]
    UnknownRun(String),
    /// A run that is not in progress, asked to stop.
    #[error("run {run} is not in progress (it is {status}); there is nothing to cancel")]
    RunNotInProgress {
        /// The run.
        run: String,
        /// Its recorded status.
        status: String,
    },
    /// The source holds no part of this name.
    #[error("source {source_id} has no part {part:?}")]
    UnknownPart {
        /// The source.
        source_id: SourceId,
        /// The part name asked for.
        part: String,
    },
    /// The database holds no artifact with this digest.
    #[error("no artifact {0} is recorded")]
    UnknownArtifact(Digest),
    /// An artifact is recorded but its file is gone.
    #[error(
        "artifact {digest} ({role}) is recorded but its file {path} is missing; \
         restore it with `splinter state repair --from <archive or state>`"
    )]
    MissingArtifact {
        /// The artifact.
        digest: Digest,
        /// What it is for.
        role: String,
        /// Where its file should be.
        path: PathBuf,
    },
    /// The store holds no content under this digest.
    #[error("no content {0} in the store")]
    UnknownBlob(Digest),
    /// A span names a part whose content is not the content the span
    /// indexes.
    #[error("span into {content} names part {part:?} of {source_id}, whose content is {actual}")]
    SpanPart {
        /// The source the span names.
        source_id: SourceId,
        /// The part the span names.
        part: String,
        /// The content digest the span indexes.
        content: Digest,
        /// The content digest the part actually has.
        actual: Digest,
    },
    /// A span reaches past the end of the content it indexes.
    #[error("span [{start}, {end}) is out of range of {content}, which is {len} bytes")]
    SpanOutOfRange {
        /// The content the span indexes.
        content: Digest,
        /// Its start.
        start: u64,
        /// Its end.
        end: u64,
        /// The content's length in bytes.
        len: u64,
    },
    /// A stored object's bytes do not hash to its address.
    #[error("{path} is corrupt: it should hash to {expected}, it hashes to {found}")]
    Corrupt {
        /// The object file.
        path: PathBuf,
        /// Its address.
        expected: Digest,
        /// What its bytes hash to.
        found: Digest,
    },
    /// A stored object hashes correctly but does not read back as the
    /// record it addresses (written by an incompatible schema).
    #[error("{path} does not decode to the record it addresses: {reason}")]
    Undecodable {
        /// The object file.
        path: PathBuf,
        /// Why.
        reason: String,
    },
    /// A stored object no longer matches the address it is stored under.
    #[error("{what} hashes to {found}, not the {expected} it is stored under")]
    Altered {
        /// The object.
        what: String,
        /// Its address.
        expected: Digest,
        /// What its content hashes to.
        found: Digest,
    },
    /// A stored object does not read back as the record it is stored as.
    #[error("{what} does not decode: {reason}")]
    UndecodableObject {
        /// The object.
        what: String,
        /// Why.
        reason: String,
    },
    /// The experience database failed.
    #[error("experience database: {0}")]
    Database(#[from] splinter_expdb::Error),
    /// A record cannot be serialized.
    #[error("cannot serialize {what}: {source}")]
    Serialize {
        /// What was being serialized.
        what: &'static str,
        /// The serializer's error.
        source: serde_json::Error,
    },
    /// A file operation failed.
    #[error("{path}: {source}")]
    Io {
        /// The file or directory.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
}
