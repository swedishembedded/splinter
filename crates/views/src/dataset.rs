// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Records written as a chat dataset brain trains on.
//!
//! The format is `generic-messages-v2` JSONL: one conversation per line,
//! `{"messages":[{"role","content","train"}...],"metadata":{...}}`, where
//! `train` on every message marks what is supervised and `metadata` is
//! carried but never rendered. The file is written beside its destination,
//! parsed with brain's own dataset parser (through `splinter-policy`), and
//! only then moved into place - so a dataset the trainer would refuse is
//! never reported written.

use std::path::{Path, PathBuf};

use splinter_store::experience::Digest;

use crate::{Record, ViewError};

/// A dataset file, as written and validated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dataset {
    /// Where it is.
    pub path: PathBuf,
    /// The digest of its bytes.
    pub digest: Digest,
    /// Conversations in it.
    pub records: usize,
    /// Supervised messages across them.
    pub trained_messages: usize,
}

/// Writes `records` to `path` as a `generic-messages-v2` dataset and
/// returns where it is and its digest. Refuses an empty set of records, and
/// a dataset brain's parser rejects or that supervises nothing; in either
/// case nothing is left at `path`.
pub fn write_dataset(path: &Path, records: &[Record]) -> Result<Dataset, ViewError> {
    if records.is_empty() {
        return Err(ViewError::Empty);
    }
    let mut text = String::new();
    for record in records {
        text.push_str(&serde_json::to_string(record)?);
        text.push('\n');
    }
    let io = |path: &Path| {
        let path = path.to_path_buf();
        move |source| ViewError::Io { path, source }
    };
    let pending = path.with_extension("pending");
    splinter_store::write_atomic(&pending, &text).map_err(io(&pending))?;
    let checked = splinter_policy::train::validate_dataset(&pending)
        .map_err(|e| ViewError::Invalid {
            path: path.to_path_buf(),
            reason: format!("{e:#}"),
        })
        .and_then(|summary| {
            if summary.trained_messages == 0 {
                return Err(ViewError::Invalid {
                    path: path.to_path_buf(),
                    reason: "no message is supervised".into(),
                });
            }
            Ok(summary)
        });
    let summary = match checked {
        Ok(summary) => summary,
        Err(e) => {
            // The refusal is the error worth reporting; a leftover pending
            // file is harmless and replaced by the next write.
            let _ = std::fs::remove_file(&pending);
            return Err(e);
        }
    };
    std::fs::rename(&pending, path).map_err(io(path))?;
    Ok(Dataset {
        path: path.to_path_buf(),
        digest: Digest::of(text.as_bytes()),
        records: summary.records,
        trained_messages: summary.trained_messages,
    })
}
