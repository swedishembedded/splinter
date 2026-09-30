// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements source capture that grounds every learned
// fact in exact, traceable bytes, for its clients. If your team needs
// expertise in knowledge acquisition or training-data provenance, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Capture: what Splinter learns from, read into a
//! [`CapturedSource`] that the source store persists.
//!
//! * [`capture_document`] - one UTF-8 text file; anything else is refused.
//! * [`capture_repository`] - every UTF-8 text file of a directory tree,
//!   with the git state when it is a work tree.
//! * [`capture_command`] - one run of a program: its standard output and
//!   error, its exit code, under a timeout and an output cap.
//!
//! Every limit is a parameter; the `DEFAULT_*` constants are the defaults
//! a caller passes when it has no reason to differ. Nothing here reads the
//! process environment: a command's environment is passed in explicitly.
//! Capturing unchanged content again yields the same source id.

mod command;
mod repository;

use std::io::Read;
use std::path::{Path, PathBuf};

use splinter_store::clock::Clock;
use splinter_store::source::{CapturedSource, Origin, PartContent, SourceError};

pub use command::{
    capture_command, default_environment, CommandSpec, DEFAULT_ENV_ALLOWLIST, DEFAULT_OUTPUT_CAP,
    DEFAULT_TERM, DEFAULT_TIMEOUT,
};
pub use repository::{capture_repository, IGNORED_NAMES};
pub use splinter_sandbox::process::ProcessError;

/// The default cap on one captured file's size, in bytes: a document
/// larger than this is refused, a repository file larger than this is
/// skipped (and listed as skipped).
pub const DEFAULT_MAX_FILE_BYTES: u64 = 1024 * 1024;

/// Why a capture failed.
#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    /// A file operation failed.
    #[error("{path}: {source}")]
    Io {
        /// The file or directory.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// A document path that is not a regular file.
    #[error("{path} is not a regular file")]
    NotAFile {
        /// The path.
        path: PathBuf,
    },
    /// A repository path that is not a directory.
    #[error("{path} is not a directory")]
    NotADirectory {
        /// The path.
        path: PathBuf,
    },
    /// A document that is not UTF-8 text (or holds a NUL byte).
    #[error("{path} is not UTF-8 text")]
    NotText {
        /// The file.
        path: PathBuf,
    },
    /// A document larger than the cap.
    #[error("{path} is larger than the {limit}-byte cap")]
    TooLarge {
        /// The file.
        path: PathBuf,
        /// The cap it exceeds.
        limit: u64,
    },
    /// A path that is not UTF-8, so it cannot be recorded as an origin.
    #[error("{path} is not a UTF-8 path")]
    NonUtf8Path {
        /// The path, lossily displayed.
        path: PathBuf,
    },
    /// The command could not be run.
    #[error(transparent)]
    Process(#[from] ProcessError),
    /// A tree with a `.git` entry, and no `git` to read its state with.
    #[error("{path} is a git work tree, and git cannot be run to read its revision: {source}")]
    GitUnavailable {
        /// The tree.
        path: PathBuf,
        /// Why git could not be run.
        source: std::io::Error,
    },
    /// A git command failed on a tree that is a work tree.
    #[error("git {args:?} failed in {path}: {detail}")]
    Git {
        /// The git arguments.
        args: Vec<String>,
        /// The tree.
        path: PathBuf,
        /// Its exit status and standard error.
        detail: String,
    },
    /// The captured source is not valid.
    #[error(transparent)]
    Source(#[from] SourceError),
}

/// Wraps an I/O error on `path`.
fn io(path: &Path) -> impl FnOnce(std::io::Error) -> CaptureError + '_ {
    move |source| CaptureError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Whether `bytes` are text: UTF-8 with no NUL byte.
fn is_text(bytes: &[u8]) -> bool {
    !bytes.contains(&0) && std::str::from_utf8(bytes).is_ok()
}

/// The media type of a text file named `name`: Markdown by extension,
/// plain text otherwise.
fn text_media_type(name: &str) -> &'static str {
    let extension = name.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase());
    match extension.as_deref() {
        Some("md" | "markdown") => crate::sections::MARKDOWN,
        _ => "text/plain",
    }
}

/// `path` as UTF-8.
fn utf8(path: &Path) -> Result<&str, CaptureError> {
    path.to_str().ok_or_else(|| CaptureError::NonUtf8Path {
        path: path.to_path_buf(),
    })
}

/// Up to `max_bytes + 1` bytes of the file at `path`: one more than the
/// cap, so a file over it is detected without reading all of it.
fn read_capped(path: &Path, max_bytes: u64) -> Result<Vec<u8>, CaptureError> {
    let file = std::fs::File::open(path).map_err(io(path))?;
    let mut bytes = Vec::new();
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(io(path))?;
    Ok(bytes)
}

/// Captures the text file at `path` as a document source with one part,
/// named by the file's name. Refused when the file is larger than
/// `max_bytes` or is not UTF-8 text. The origin records the absolute path;
/// `clock` stamps the capture.
pub fn capture_document(
    path: &Path,
    max_bytes: u64,
    clock: &dyn Clock,
) -> Result<CapturedSource, CaptureError> {
    let path = path.canonicalize().map_err(io(path))?;
    if !path.metadata().map_err(io(&path))?.is_file() {
        return Err(CaptureError::NotAFile { path });
    }
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| CaptureError::NonUtf8Path { path: path.clone() })?
        .to_string();
    let bytes = read_capped(&path, max_bytes)?;
    if bytes.len() as u64 > max_bytes {
        return Err(CaptureError::TooLarge {
            path,
            limit: max_bytes,
        });
    }
    if !is_text(&bytes) {
        return Err(CaptureError::NotText { path });
    }
    let origin = Origin::Document {
        path: utf8(&path)?.to_string(),
    };
    let part = PartContent {
        media_type: text_media_type(&name).to_string(),
        name,
        bytes,
    };
    Ok(CapturedSource::new(origin, vec![part], clock)?)
}
