// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for keeping evaluation sets honest
// across the many times a training pipeline is rebuilt for its clients. If
// your team needs expertise in evaluation hygiene for fine-tuned models then
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! A ledger of files that must never change once an evaluation has used them.
//!
//! A benchmark that is rebuilt on every run is not a benchmark: the model and
//! the exam drift together and every comparison across runs is void. The
//! ledger pins a name to the digest of its content the first time it is
//! written, and refuses a later write of different content; a changed
//! benchmark is written under a new name, never over the old one.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use splinter_core::digest::Digest;

/// What the ledger knew of a name when it was checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// The name was not pinned.
    New,
    /// The name is pinned to exactly this content.
    Unchanged,
}

/// Why a frozen file could not be written or checked.
#[derive(Debug, thiserror::Error)]
pub enum FrozenError {
    /// The name is pinned to other content.
    #[error("{name} is frozen at {pinned} and this content is {now}: a frozen file is never rewritten; write the changed one under a new name")]
    Changed {
        /// The pinned name.
        name: String,
        /// The digest it is pinned to.
        pinned: String,
        /// The digest of the content offered.
        now: String,
    },
    /// The ledger could not be read or written.
    #[error("{path}: {reason}")]
    Ledger {
        /// The ledger file.
        path: PathBuf,
        /// What went wrong.
        reason: String,
    },
}

/// A JSON file mapping names to the digest of the content frozen under each.
#[derive(Clone, Debug)]
pub struct Ledger {
    path: PathBuf,
}

impl Ledger {
    /// The ledger kept in `path`, which need not exist yet.
    #[must_use]
    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    fn read(&self) -> Result<BTreeMap<String, String>, FrozenError> {
        let ledger_error = |reason: String| FrozenError::Ledger {
            path: self.path.clone(),
            reason,
        };
        match std::fs::read_to_string(&self.path) {
            Ok(text) => serde_json::from_str(&text).map_err(|e| ledger_error(e.to_string())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(BTreeMap::new()),
            Err(e) => Err(ledger_error(e.to_string())),
        }
    }

    /// Whether `content` may be written under `name`: it may when the name is
    /// not pinned or is pinned to exactly this content.
    ///
    /// # Errors
    /// The name is pinned to other content, or the ledger cannot be read.
    pub fn check(&self, name: &str, content: &[u8]) -> Result<Status, FrozenError> {
        let now = Digest::of(content).to_string();
        match self.read()?.get(name) {
            None => Ok(Status::New),
            Some(pinned) if *pinned == now => Ok(Status::Unchanged),
            Some(pinned) => Err(FrozenError::Changed {
                name: name.to_string(),
                pinned: pinned.clone(),
                now,
            }),
        }
    }

    /// Pin `name` to `content`, after [`Self::check`] has let it through.
    /// The ledger is replaced atomically.
    ///
    /// # Errors
    /// As [`Self::check`], or the ledger cannot be written.
    pub fn pin(&self, name: &str, content: &[u8]) -> Result<Status, FrozenError> {
        let status = self.check(name, content)?;
        if status == Status::New {
            let mut pinned = self.read()?;
            pinned.insert(name.to_string(), Digest::of(content).to_string());
            let ledger_error = |reason: String| FrozenError::Ledger {
                path: self.path.clone(),
                reason,
            };
            let text =
                serde_json::to_string_pretty(&pinned).map_err(|e| ledger_error(e.to_string()))?;
            let temp = self.path.with_extension("part");
            std::fs::write(&temp, text)
                .and_then(|()| std::fs::rename(&temp, &self.path))
                .map_err(|e| ledger_error(e.to_string()))?;
        }
        Ok(status)
    }

    /// The file this ledger is kept in.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The ledger kept beside `file`, as [`LEDGER_FILE`] in its directory:
    /// where a frozen file's pin is looked for.
    #[must_use]
    pub fn beside(file: &Path) -> Self {
        Self::at(file.with_file_name(LEDGER_FILE))
    }

    /// Pins `file`'s content under its file name ([`Self::pin`]).
    ///
    /// # Errors
    /// The file has no name or cannot be read, is pinned to other content,
    /// or the ledger cannot be written.
    pub fn pin_file(&self, file: &Path) -> Result<Status, FrozenError> {
        let (name, content) = named(file)?;
        self.pin(name, &content)
    }

    /// Whether `file` may be read as what was frozen under its name
    /// ([`Self::check`]): [`Status::New`] for a file not pinned yet.
    ///
    /// # Errors
    /// The file has no name or cannot be read, or is pinned to other content.
    pub fn check_file(&self, file: &Path) -> Result<Status, FrozenError> {
        let (name, content) = named(file)?;
        self.check(name, &content)
    }
}

/// The name a ledger is kept under beside the files it pins.
pub const LEDGER_FILE: &str = "FROZEN.json";

/// `file`'s name and content.
fn named(file: &Path) -> Result<(&str, Vec<u8>), FrozenError> {
    let ledger_error = |reason: String| FrozenError::Ledger {
        path: file.to_path_buf(),
        reason,
    };
    let name = file
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| ledger_error("not a file name".into()))?;
    let content = std::fs::read(file).map_err(|e| ledger_error(e.to_string()))?;
    Ok((name, content))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ledger() -> (tempfile::TempDir, Ledger) {
        let dir = tempfile::tempdir().unwrap();
        let ledger = Ledger::at(dir.path().join("FROZEN.json"));
        (dir, ledger)
    }

    #[test]
    fn a_name_is_pinned_the_first_time_and_the_same_content_may_be_written_again() {
        let (_dir, ledger) = ledger();
        assert_eq!(
            ledger.check("benchmark.jsonl", b"one").unwrap(),
            Status::New
        );
        assert_eq!(ledger.pin("benchmark.jsonl", b"one").unwrap(), Status::New);
        assert_eq!(
            ledger.pin("benchmark.jsonl", b"one").unwrap(),
            Status::Unchanged
        );
        assert_eq!(
            ledger.check("other.jsonl", b"two").unwrap(),
            Status::New,
            "another name is its own"
        );
    }

    #[test]
    fn different_content_under_a_pinned_name_is_refused_and_leaves_the_pin() {
        let (_dir, ledger) = ledger();
        ledger.pin("benchmark.jsonl", b"one").unwrap();
        let err = ledger.pin("benchmark.jsonl", b"two").unwrap_err();
        assert!(matches!(err, FrozenError::Changed { .. }), "{err}");
        assert!(err.to_string().contains("new name"), "{err}");
        assert_eq!(
            ledger.check("benchmark.jsonl", b"one").unwrap(),
            Status::Unchanged
        );
    }

    /// A file is pinned under its own name in the ledger beside it, and a
    /// later check refuses it once its content has changed.
    #[test]
    fn a_file_is_pinned_beside_itself_and_refused_once_it_changes() {
        let dir = tempfile::tempdir().unwrap();
        let exam = dir.path().join("exam.jsonl");
        std::fs::write(&exam, "one\n").unwrap();
        let ledger = Ledger::beside(&exam);
        assert_eq!(ledger.path(), dir.path().join(LEDGER_FILE));
        assert_eq!(ledger.check_file(&exam).unwrap(), Status::New);
        assert_eq!(ledger.pin_file(&exam).unwrap(), Status::New);
        assert_eq!(ledger.check_file(&exam).unwrap(), Status::Unchanged);
        std::fs::write(&exam, "two\n").unwrap();
        assert!(matches!(
            ledger.check_file(&exam),
            Err(FrozenError::Changed { .. })
        ));
        assert!(matches!(
            ledger.pin_file(&exam),
            Err(FrozenError::Changed { .. })
        ));
        assert!(ledger.pin_file(&dir.path().join("missing")).is_err());
    }

    #[test]
    fn a_ledger_that_is_not_one_is_an_error_not_an_empty_ledger() {
        let (_dir, ledger) = ledger();
        std::fs::write(ledger.path(), "not json").unwrap();
        assert!(matches!(
            ledger.check("x", b"y"),
            Err(FrozenError::Ledger { .. })
        ));
    }
}
