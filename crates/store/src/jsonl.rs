// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements resumable batch jobs over language models
// for its clients. If your team needs expertise in long-running generation or
// evaluation runs that survive interruption without repeating work, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The resumable job log: one JSON record per line in a plain file, appended
//! as each unit of work finishes, so a run that stops can be started again and
//! skip what is already done.
//!
//! The file is the truth about what finished. [`read`] returns every record in
//! order; [`read_unique`] keeps the first record for each key, which is what a
//! reader wants when two runs overlapped and both wrote the same item. A
//! [`Log`] writes each record as one `write` under a lock, so records
//! appended from many threads never interleave inside a line.

use std::fs::{File, OpenOptions};
use std::hash::Hash;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use serde::de::DeserializeOwned;
use serde::Serialize;

/// Why a log could not be read or written.
#[derive(Debug, thiserror::Error)]
pub enum JsonlError {
    /// The file could not be opened, read or written.
    #[error("{path}: {source}")]
    Io {
        /// The file.
        path: PathBuf,
        /// What the system reported.
        source: std::io::Error,
    },
    /// A line is not a record of the expected shape.
    #[error("{path} line {line}: {source}")]
    Line {
        /// The file.
        path: PathBuf,
        /// The line, from one.
        line: usize,
        /// What the parser reported.
        source: serde_json::Error,
    },
    /// A record could not be written as JSON.
    #[error("a record of {path} cannot be written: {source}")]
    Encode {
        /// The file.
        path: PathBuf,
        /// What the serializer reported.
        source: serde_json::Error,
    },
}

/// Every record of the log at `path`, in order; a file that is not there
/// holds none.
///
/// # Errors
/// The file cannot be read, or a line is not a `T`.
pub fn read<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>, JsonlError> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(source) => {
            return Err(JsonlError::Io {
                path: path.to_path_buf(),
                source,
            })
        }
    };
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(n, line)| {
            serde_json::from_str(line).map_err(|source| JsonlError::Line {
                path: path.to_path_buf(),
                line: n + 1,
                source,
            })
        })
        .collect()
}

/// The records of the log at `path`, the first for each key.
///
/// # Errors
/// As [`read`].
pub fn read_unique<T, K>(path: &Path, key: impl Fn(&T) -> K) -> Result<Vec<T>, JsonlError>
where
    T: DeserializeOwned,
    K: Hash + Eq,
{
    let mut seen = std::collections::HashSet::new();
    Ok(read::<T>(path)?
        .into_iter()
        .filter(|record| seen.insert(key(record)))
        .collect())
}

/// An open log that records are appended to, from any number of threads.
pub struct Log {
    path: PathBuf,
    file: Mutex<File>,
}

impl Log {
    /// Opens the log at `path` for appending, creating it if it is not there.
    ///
    /// # Errors
    /// The file cannot be opened.
    pub fn open(path: &Path) -> Result<Self, JsonlError> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|source| JsonlError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        Ok(Self {
            path: path.to_path_buf(),
            file: Mutex::new(file),
        })
    }

    /// Appends `record` as one line.
    ///
    /// # Errors
    /// The record cannot be encoded or the file cannot be written.
    pub fn append<T: Serialize>(&self, record: &T) -> Result<(), JsonlError> {
        let mut line = serde_json::to_string(record).map_err(|source| JsonlError::Encode {
            path: self.path.clone(),
            source,
        })?;
        line.push('\n');
        let mut file = self.file.lock().unwrap_or_else(PoisonError::into_inner);
        file.write_all(line.as_bytes())
            .map_err(|source| JsonlError::Io {
                path: self.path.clone(),
                source,
            })
    }
}
