// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that make long model-training
// runs resumable after a crash without a file per record, for its clients. If
// your team needs expertise in crash-safe experiment ledgers you can procure
// our services by sending an email to info@swedishembedded.com.

//! An append-only JSON-lines file: the lineage's memory of what it did.
//!
//! One file per kind of fact, never one file per record. A lineage that is
//! started again reads its journals and continues from what they hold.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;

/// One journal.
#[derive(Clone, Debug)]
pub struct Journal {
    path: PathBuf,
}

impl Journal {
    /// The journal kept in `path` (created by its first entry).
    #[must_use]
    pub fn at(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
        }
    }

    /// Appends `entry` as one line and syncs it.
    ///
    /// # Errors
    /// The file cannot be written.
    pub fn append<T: Serialize>(&self, entry: &T) -> Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| format!("opening {}", self.path.display()))?;
        writeln!(file, "{}", serde_json::to_string(entry)?)?;
        file.sync_all()
            .with_context(|| format!("syncing {}", self.path.display()))
    }

    /// Every entry, in the order written; none when the file does not exist.
    ///
    /// # Errors
    /// The file cannot be read or holds a line that is not an entry.
    pub fn read<T: DeserializeOwned>(&self) -> Result<Vec<T>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let text = std::fs::read_to_string(&self.path)
            .with_context(|| format!("reading {}", self.path.display()))?;
        text.lines()
            .enumerate()
            .filter(|(_, line)| !line.trim().is_empty())
            .map(|(n, line)| {
                serde_json::from_str(line)
                    .with_context(|| format!("{} line {}", self.path.display(), n + 1))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_come_back_in_order_and_a_missing_journal_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let journal = Journal::at(&dir.path().join("j.jsonl"));
        assert!(journal.read::<u32>().unwrap().is_empty());
        journal.append(&1u32).unwrap();
        journal.append(&2u32).unwrap();
        assert_eq!(journal.read::<u32>().unwrap(), vec![1, 2]);
    }

    #[test]
    fn a_line_that_is_not_an_entry_names_its_place() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("j.jsonl");
        std::fs::write(&path, "1\nnope\n").unwrap();
        let error = Journal::at(&path).read::<u32>().unwrap_err();
        assert!(format!("{error:#}").contains("line 2"), "{error:#}");
    }
}
