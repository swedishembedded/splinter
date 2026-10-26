// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements interchange of agent benchmark tasks between
// task formats for its clients. If your team needs expertise in importing
// task directories into reproducible, verifiable task sets, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Reading and copying the task directories that interchange formats share:
//! the files of a directory with a bound on its size, a copy that leaves
//! chosen entries out, and the refusal that names the directory and why.

use std::collections::BTreeMap;
use std::path::Path;

use super::WorkspaceError;

/// The most bytes of one task directory read.
const MAX_TASK_BYTES: u64 = 64 * 1024 * 1024;

/// A refusal of the task directory `dir`, saying why.
pub(super) fn refuse<T>(dir: &Path, reason: impl Into<String>) -> Result<T, WorkspaceError> {
    Err(WorkspaceError::Fixture {
        path: dir.to_path_buf(),
        reason: reason.into(),
    })
}

/// The error constructor for a failed file operation on `path`.
pub(super) fn io(path: &Path) -> impl FnOnce(std::io::Error) -> WorkspaceError + '_ {
    move |source| WorkspaceError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Every regular file under `root`, relative, refusing symlinks and a tree
/// over [`MAX_TASK_BYTES`].
pub(super) fn files_under(root: &Path) -> Result<Vec<String>, WorkspaceError> {
    fn walk(
        dir: &Path,
        prefix: &str,
        out: &mut Vec<String>,
        bytes: &mut u64,
    ) -> Result<(), WorkspaceError> {
        for entry in std::fs::read_dir(dir).map_err(io(dir))? {
            let entry = entry.map_err(io(dir))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let kind = entry.file_type().map_err(io(dir))?;
            let path = entry.path();
            if kind.is_symlink() {
                return refuse(&path, "it is a symbolic link");
            }
            if kind.is_dir() {
                walk(&path, &format!("{prefix}{name}/"), out, bytes)?;
            } else {
                *bytes += entry.metadata().map_err(io(&path))?.len();
                if *bytes > MAX_TASK_BYTES {
                    return refuse(dir, "the task is over 64 MiB");
                }
                out.push(format!("{prefix}{name}"));
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    let mut bytes = 0;
    if root.is_dir() {
        walk(root, "", &mut out, &mut bytes)?;
    }
    out.sort();
    Ok(out)
}

/// Copies one file, making the directories above its destination.
pub(super) fn copy_file(from: &Path, to: &Path) -> Result<(), WorkspaceError> {
    if let Some(parent) = to.parent() {
        std::fs::create_dir_all(parent).map_err(io(parent))?;
    }
    std::fs::copy(from, to).map_err(io(from))?;
    Ok(())
}

/// Copies the files of `from` into `to`, leaving out the entries of `from`
/// itself that `skip` names (not those of the directories below it).
pub(super) fn copy_dir(
    from: &Path,
    to: &Path,
    skip: fn(&str) -> bool,
) -> Result<(), WorkspaceError> {
    std::fs::create_dir_all(to).map_err(io(to))?;
    for entry in std::fs::read_dir(from).map_err(io(from))? {
        let entry = entry.map_err(io(from))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if skip(&name) {
            continue;
        }
        let kind = entry.file_type().map_err(io(from))?;
        if kind.is_dir() {
            copy_dir(&entry.path(), &to.join(&name), skip_nothing)?;
        } else {
            copy_file(&entry.path(), &to.join(&name))?;
        }
    }
    Ok(())
}

/// A `skip` that leaves nothing out.
pub(super) fn skip_nothing(_: &str) -> bool {
    false
}

/// Reads the text files of `dir` as `(path, bytes)`.
pub(super) fn read_files(dir: &Path) -> Result<BTreeMap<String, Vec<u8>>, WorkspaceError> {
    let mut files = BTreeMap::new();
    for file in files_under(dir)? {
        let path = dir.join(&file);
        files.insert(file, std::fs::read(&path).map_err(io(&path))?);
    }
    Ok(files)
}
