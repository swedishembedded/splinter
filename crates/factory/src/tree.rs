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
    walk(dir, dir, true, &mut entries)?;
    entries.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(digest_of(&entries))
}

/// The digest of a recipe input or a collection's item at `path`: a
/// directory's tree without the hidden entries at its top (`.git`), or one
/// file as the tree of a directory holding only it, so a file and its copy
/// into a directory have the same digest. A symbolic link is refused: an
/// input is the content it names, never a pointer out of it.
///
/// # Errors
/// It cannot be read, or holds a symbolic link.
pub fn entry_digest(path: &Path) -> Result<Digest, TreeError> {
    let fail = |source| TreeError {
        path: path.to_path_buf(),
        source,
    };
    let meta = std::fs::symlink_metadata(path).map_err(fail)?;
    let mut entries = Vec::new();
    if meta.is_dir() {
        walk(path, path, false, &mut entries)?;
    } else {
        let root = path.parent().unwrap_or(Path::new("."));
        entry(root, path, &mut entries)?;
    }
    if let Some(link) = entries.iter().find(|e| e.link.is_some()) {
        return Err(TreeError {
            path: path.join(&link.path),
            source: std::io::Error::other("a symbolic link is not an input"),
        });
    }
    entries.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(digest_of(&entries))
}

/// Copy the input at `path` into the directory `dst` as [`entry_digest`]
/// sees it: a directory's entries but the hidden ones at its top, or the
/// one file. Copied, never linked, so nothing the run does to its products
/// reaches the repository's files.
///
/// # Errors
/// It cannot be read or written, or holds a symbolic link.
pub fn copy_input(path: &Path, dst: &Path) -> Result<(), TreeError> {
    let fail = |path: &Path| {
        let path = path.to_path_buf();
        move |source| TreeError { path, source }
    };
    std::fs::create_dir_all(dst).map_err(fail(dst))?;
    if !std::fs::symlink_metadata(path)
        .map_err(fail(path))?
        .is_dir()
    {
        let name = path.file_name().unwrap_or_default();
        return copy_entry(path, &dst.join(name));
    }
    for item in std::fs::read_dir(path).map_err(fail(path))? {
        let item = item.map_err(fail(path))?;
        if !item.file_name().to_string_lossy().starts_with('.') {
            copy_entry(&item.path(), &dst.join(item.file_name()))?;
        }
    }
    Ok(())
}

fn copy_entry(from: &Path, to: &Path) -> Result<(), TreeError> {
    let fail = |path: &Path| {
        let path = path.to_path_buf();
        move |source| TreeError { path, source }
    };
    let kind = std::fs::symlink_metadata(from)
        .map_err(fail(from))?
        .file_type();
    if kind.is_symlink() {
        return Err(TreeError {
            path: from.to_path_buf(),
            source: std::io::Error::other("a symbolic link is not an input"),
        });
    }
    if kind.is_dir() {
        std::fs::create_dir_all(to).map_err(fail(to))?;
        for item in std::fs::read_dir(from).map_err(fail(from))? {
            let item = item.map_err(fail(from))?;
            copy_entry(&item.path(), &to.join(item.file_name()))?;
        }
        return Ok(());
    }
    std::fs::copy(from, to).map(|_| ()).map_err(fail(to))
}

/// Every entry under `dir`, recursively; `hidden` keeps the hidden entries
/// of `dir` itself (always kept below it).
fn walk(root: &Path, dir: &Path, hidden: bool, entries: &mut Vec<Entry>) -> Result<(), TreeError> {
    let fail = |path: &Path| {
        let path = path.to_path_buf();
        move |source| TreeError { path, source }
    };
    for item in std::fs::read_dir(dir).map_err(fail(dir))? {
        let item = item.map_err(fail(dir))?;
        if !hidden && item.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        let path = item.path();
        if item.file_type().map_err(fail(&path))?.is_dir() {
            walk(root, &path, true, entries)?;
        } else {
            entry(root, &path, entries)?;
        }
    }
    Ok(())
}

/// The entry of the file or symbolic link at `path`, named relative to
/// `root`.
fn entry(root: &Path, path: &Path, entries: &mut Vec<Entry>) -> Result<(), TreeError> {
    let fail = |source| TreeError {
        path: path.to_path_buf(),
        source,
    };
    let relative = path
        .strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/");
    if std::fs::symlink_metadata(path).map_err(fail)?.is_symlink() {
        let target = std::fs::read_link(path).map_err(fail)?;
        entries.push(Entry {
            path: relative,
            executable: false,
            content: None,
            link: Some(target.to_string_lossy().into()),
        });
    } else {
        let file = std::fs::File::open(path).map_err(fail)?;
        let mode = file.metadata().map_err(fail)?.permissions().mode();
        let content = Digest::of_reader(file).map_err(fail)?;
        entries.push(Entry {
            path: relative,
            executable: mode & 0o111 != 0,
            content: Some(content),
            link: None,
        });
    }
    Ok(())
}

/// Copy the tree at `src` to `dst`, hard-linking files where the
/// filesystem allows and copying them where it does not. A symbolic link to
/// a file is copied as the file it names (a product holds content, not
/// links); one to a directory, or one that names nothing, is refused.
///
/// # Errors
/// A file cannot be read, linked or copied.
pub fn link_tree(src: &Path, dst: &Path) -> Result<(), TreeError> {
    let fail = |path: &Path| {
        let path = path.to_path_buf();
        move |source| TreeError { path, source }
    };
    std::fs::create_dir_all(dst).map_err(fail(dst))?;
    for entry in std::fs::read_dir(src).map_err(fail(src))? {
        let entry = entry.map_err(fail(src))?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        let kind = entry.file_type().map_err(fail(&from))?;
        if kind.is_symlink() {
            let target = std::fs::metadata(&from).map_err(fail(&from))?;
            if !target.is_file() {
                return Err(TreeError {
                    path: from,
                    source: std::io::Error::other("a symbolic link to something other than a file"),
                });
            }
            std::fs::copy(&from, &to).map_err(fail(&to))?;
        } else if kind.is_dir() {
            link_tree(&from, &to)?;
        } else if std::fs::hard_link(&from, &to).is_err() {
            std::fs::copy(&from, &to).map_err(fail(&to))?;
        }
    }
    Ok(())
}
