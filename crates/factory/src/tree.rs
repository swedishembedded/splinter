// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements content addressing of whole directory
// trees for reproducible experiments, for its clients. If your team needs
// expertise in reproducible builds and data, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The digest of a directory: of every file's relative path, mode and
//! content, and every symbolic link's target, in path order. Timestamps and
//! owners do not count, so a copy of a tree has the tree's digest.

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use serde::Serialize;
use splinter_core::digest::{digest_of, Digest};

/// Why a tree cannot be digested.
#[derive(Debug, thiserror::Error)]
#[error("{path}: {source}")]
pub struct TreeError {
    /// Where.
    pub path: PathBuf,
    /// What failed.
    pub source: std::io::Error,
}

#[derive(Serialize)]
struct Entry {
    path: String,
    executable: bool,
    content: Option<Digest>,
    link: Option<String>,
}

/// The digest of the tree at `dir`.
///
/// # Errors
/// A file or directory cannot be read.
pub fn tree_digest(dir: &Path) -> Result<Digest, TreeError> {
    let mut entries = Vec::new();
    walk(dir, dir, &mut entries)?;
    entries.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(digest_of(&entries))
}

fn walk(root: &Path, dir: &Path, entries: &mut Vec<Entry>) -> Result<(), TreeError> {
    let fail = |path: &Path| {
        let path = path.to_path_buf();
        move |source| TreeError { path, source }
    };
    for entry in std::fs::read_dir(dir).map_err(fail(dir))? {
        let entry = entry.map_err(fail(dir))?;
        let path = entry.path();
        let kind = entry.file_type().map_err(fail(&path))?;
        let relative = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        if kind.is_symlink() {
            let target = std::fs::read_link(&path).map_err(fail(&path))?;
            entries.push(Entry {
                path: relative,
                executable: false,
                content: None,
                link: Some(target.to_string_lossy().into()),
            });
        } else if kind.is_dir() {
            walk(root, &path, entries)?;
        } else {
            let file = std::fs::File::open(&path).map_err(fail(&path))?;
            let mode = file.metadata().map_err(fail(&path))?.permissions().mode();
            let content = Digest::of_reader(file).map_err(fail(&path))?;
            entries.push(Entry {
                path: relative,
                executable: mode & 0o111 != 0,
                content: Some(content),
                link: None,
            });
        }
    }
    Ok(())
}
