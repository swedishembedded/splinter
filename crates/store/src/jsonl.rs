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
//!
//! A crash or a full disk can leave the last line cut short. That line is a
//! record that never finished, so it is not a record: [`read`] leaves it out
//! and says so, [`read_with_tail`] hands it back as a [`TornTail`], and
//! [`Log::open`] cuts it off before the next append, which would otherwise be
//! written onto its end and merge two records into one unreadable line.

use std::fs::{File, OpenOptions};
use std::hash::Hash;
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use serde::de::{DeserializeOwned, IgnoredAny};
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

/// A last line that stops short of its newline and is not complete JSON: the
/// record being written when the run stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TornTail {
    /// The line, from one.
    pub line: usize,
    /// How many bytes of it reached the file.
    pub bytes: usize,
}

/// What a log holds: its records and, if its last line was cut short, that
/// line's place.
#[derive(Debug)]
pub struct Found<T> {
    /// Every complete record, in order.
    pub records: Vec<T>,
    /// The cut-short last line, if there is one.
    pub torn: Option<TornTail>,
}

fn is_torn(tail: &[u8]) -> bool {
    !tail.iter().all(u8::is_ascii_whitespace) && serde_json::from_slice::<IgnoredAny>(tail).is_err()
}

fn scan<T: DeserializeOwned>(path: &Path, bytes: &[u8]) -> Result<Found<T>, JsonlError> {
    let mut records = Vec::new();
    let mut torn = None;
    let mut lines = bytes.split(|b| *b == b'\n').enumerate().peekable();
    while let Some((n, line)) = lines.next() {
        if lines.peek().is_none() && is_torn(line) {
            torn = Some(TornTail {
                line: n + 1,
                bytes: line.len(),
            });
        } else if !line.iter().all(u8::is_ascii_whitespace) {
            records.push(
                serde_json::from_slice(line).map_err(|source| JsonlError::Line {
                    path: path.to_path_buf(),
                    line: n + 1,
                    source,
                })?,
            );
        }
    }
    Ok(Found { records, torn })
}

fn read_bytes(path: &Path) -> Result<Option<Vec<u8>>, JsonlError> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(JsonlError::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

fn say_torn(path: &Path, torn: TornTail) {
    eprintln!(
        "{}: line {} was cut short ({} bytes) and is not a record",
        path.display(),
        torn.line,
        torn.bytes
    );
}

/// Every complete record of the log at `path`, in order, and the cut-short
/// last line if there is one; a file that is not there holds none.
///
/// # Errors
/// The file cannot be read, or a complete line is not a `T`.
pub fn read_with_tail<T: DeserializeOwned>(path: &Path) -> Result<Found<T>, JsonlError> {
    match read_bytes(path)? {
        Some(bytes) => scan(path, &bytes),
        None => Ok(Found {
            records: Vec::new(),
            torn: None,
        }),
    }
}

/// Every complete record of the log at `path`, in order; a file that is not
/// there holds none. A cut-short last line is left out and reported on
/// standard error; [`read_with_tail`] returns it instead.
///
/// # Errors
/// The file cannot be read, or a complete line is not a `T`.
pub fn read<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>, JsonlError> {
    let found = read_with_tail(path)?;
    if let Some(torn) = found.torn {
        say_torn(path, torn);
    }
    Ok(found.records)
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
    truncated: Option<TornTail>,
}

impl Log {
    /// Opens the log at `path` for appending, creating it if it is not there.
    /// A cut-short last line is cut off, with a note on standard error, and a
    /// complete last record that lost its newline is given one, so that the
    /// next append starts a line of its own.
    ///
    /// # Errors
    /// The file cannot be opened, read or repaired.
    pub fn open(path: &Path) -> Result<Self, JsonlError> {
        let io = |source| JsonlError::Io {
            path: path.to_path_buf(),
            source,
        };
        let mut file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(path)
            .map_err(io)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).map_err(io)?;
        let sound_len = bytes.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1);
        let tail = &bytes[sound_len..];
        let mut truncated = None;
        if is_torn(tail) {
            let torn = TornTail {
                line: bytes[..sound_len].iter().filter(|b| **b == b'\n').count() + 1,
                bytes: tail.len(),
            };
            say_torn(path, torn);
            file.set_len(sound_len as u64).map_err(io)?;
            truncated = Some(torn);
        } else if !tail.is_empty() {
            file.write_all(b"\n").map_err(io)?;
        }
        Ok(Self {
            path: path.to_path_buf(),
            file: Mutex::new(file),
            truncated,
        })
    }

    /// The cut-short last line that opening cut off, if there was one.
    #[must_use]
    pub fn truncated(&self) -> Option<&TornTail> {
        self.truncated.as_ref()
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
