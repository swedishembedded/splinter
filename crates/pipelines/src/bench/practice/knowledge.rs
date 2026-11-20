// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for measuring what an agent does
// with the reference material it is handed for its clients. If your team
// needs expertise in knowledge-grounded agent practice you can procure our
// services by sending an email to info@swedishembedded.com.

//! A reference-knowledge collection a practice run mounts (`bench task
//! practice --knowledge DIR`, read-only at `/knowledge` in every sandbox),
//! identified by content.
//!
//! The digest is the one the examiner records for the copy it mounts:
//! `sha256:` over one line per regular file, `<relative POSIX path>\0<sha256 of the file>\n`, in
//! path order, without any entry named `.git`. A run is keyed by it, so the
//! same material with another collection (or none) is another job, and the
//! examiner's record of what it mounted is checked against it.

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::bench::artifact::{sha256_file, sha256_hex};
use crate::episodes::EpisodeError;

/// Entries the examiner leaves out of a collection.
const EXCLUDED: &str = ".git";

fn refused(dir: &Path, reason: impl std::fmt::Display) -> EpisodeError {
    EpisodeError::Settings(format!(
        "the knowledge collection {}: {reason}",
        dir.display()
    ))
}

/// Every regular file below `root` as (relative path, path); refuses links
/// and special files, which the examiner refuses too.
fn files(dir: &Path) -> Result<Vec<(String, PathBuf)>, EpisodeError> {
    let mut found = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(current) = pending.pop() {
        for entry in std::fs::read_dir(&current).map_err(|e| refused(dir, e))? {
            let entry = entry.map_err(|e| refused(dir, e))?;
            if entry.file_name() == EXCLUDED {
                continue;
            }
            let path = entry.path();
            let kind = entry.file_type().map_err(|e| refused(dir, e))?;
            if kind.is_dir() {
                pending.push(path);
            } else if kind.is_file() {
                let relative = path
                    .strip_prefix(dir)
                    .map_err(|e| refused(dir, e))?
                    .to_str()
                    .ok_or_else(|| refused(dir, format!("{} is not UTF-8", path.display())))?
                    .replace(std::path::MAIN_SEPARATOR, "/");
                found.push((relative, path));
            } else {
                return Err(refused(
                    dir,
                    format!(
                        "{} is a link or special file; a collection holds regular files and \
                         directories only",
                        path.display()
                    ),
                ));
            }
        }
    }
    found.sort();
    Ok(found)
}

/// The content digest of the collection at `dir` (module docs).
///
/// # Errors
/// [`EpisodeError::Settings`] when `dir` is not a readable directory of
/// regular files.
pub fn knowledge_digest(dir: &Path) -> Result<String, EpisodeError> {
    if !dir.is_dir() {
        return Err(refused(dir, "not a directory"));
    }
    let mut lines = String::new();
    for (relative, path) in files(dir)? {
        let sha = sha256_file(&path).map_err(|e| refused(dir, e))?;
        lines.push_str(&relative);
        lines.push('\0');
        lines.push_str(&sha);
        lines.push('\n');
    }
    Ok(format!("sha256:{}", sha256_hex(lines.as_bytes())))
}

/// Why the examiner's record of the staging (`practice.json`'s `knowledge`)
/// is not the collection the run asked for, `None` when it is: the digest
/// mounted when one was given, nothing mounted when none was.
#[must_use]
pub fn mount_differs(recorded: Option<&Value>, wanted: Option<&str>) -> Option<String> {
    let mounted = recorded
        .filter(|r| r.get("mounted").and_then(Value::as_bool) == Some(true))
        .map(|r| {
            r.get("sha256")
                .and_then(Value::as_str)
                .unwrap_or("no digest")
        });
    match (mounted, wanted) {
        (None, None) => None,
        (Some(m), Some(w)) if m == w => None,
        (Some(m), None) => Some(format!(
            "it mounted the knowledge collection {m}; the run asked for none"
        )),
        (None, Some(w)) => Some(format!(
            "it mounted no knowledge collection; the run asked for {w}"
        )),
        (Some(m), Some(w)) => Some(format!(
            "it mounted the knowledge collection {m}; the run asked for {w}"
        )),
    }
}
