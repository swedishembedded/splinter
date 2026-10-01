// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The one error type of the database: every failure names what failed and
//! on which input.

use std::io;
use std::path::PathBuf;

/// What can go wrong reading or writing the database.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A filesystem operation failed.
    #[error("{}: {source}", path.display())]
    Io {
        /// The file or directory the operation was on.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: io::Error,
    },
    /// Stored bytes do not match their checksum, address or format.
    #[error("{what} is corrupt: {reason}")]
    Corrupt {
        /// What was being read.
        what: String,
        /// Why it was rejected.
        reason: String,
    },
    /// Something asked for does not exist in the snapshot or store.
    #[error("{what} not found")]
    NotFound {
        /// What was asked for.
        what: String,
    },
    /// A caller-supplied value was refused.
    #[error("invalid {what}: {reason}")]
    Invalid {
        /// The kind of value.
        what: &'static str,
        /// Why it was refused.
        reason: String,
    },
    /// A value could not be turned into bytes.
    #[error("cannot encode {what}: {source}")]
    Encode {
        /// What was being encoded.
        what: &'static str,
        /// The serializer's error.
        #[source]
        source: serde_json::Error,
    },
    /// Bytes could not be turned into a value.
    #[error("cannot decode {what}: {source}")]
    Decode {
        /// What was being decoded.
        what: String,
        /// The deserializer's error.
        #[source]
        source: serde_json::Error,
    },
}

/// The result type used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// An I/O error on `path`, for `map_err(Error::io(path))`.
    pub fn io(path: impl Into<PathBuf>) -> impl FnOnce(io::Error) -> Error {
        let path = path.into();
        move |source| Error::Io { path, source }
    }

    /// A corruption error.
    pub fn corrupt(what: impl Into<String>, reason: impl Into<String>) -> Error {
        Error::Corrupt {
            what: what.into(),
            reason: reason.into(),
        }
    }

    /// A refused value.
    pub fn invalid(what: &'static str, reason: impl Into<String>) -> Error {
        Error::Invalid {
            what,
            reason: reason.into(),
        }
    }
}
