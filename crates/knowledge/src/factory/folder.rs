// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reproducible, content-addressed training
// environments for its clients. If your team needs expertise in packaging
// agent tasks so that every instance can be rebuilt and compared byte for
// byte, you can procure our services by sending an email to
// info@swedishembedded.com.

//! A task folder held in memory: paths and bytes, with a content address.
//!
//! Generators build a [`Folder`], admission runs it, and only then is it
//! written to disk, so a generator never leaves a half-written environment
//! behind and two builds of the same instance can be compared by
//! [`Folder::digest`] before anything is stored.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Serialize;
use splinter_core::digest::{digest_of, Digest};

/// The most files a folder read from disk may hold.
pub const MAX_FILES: usize = 4096;

/// The largest file a folder read from disk may hold, in bytes.
pub const MAX_FILE_BYTES: u64 = 16 * 1024 * 1024;

/// Why a folder could not be built, written or read.
#[derive(Debug, thiserror::Error)]
pub enum FolderError {
    /// A path that is empty, absolute or climbs out of the folder.
    #[error("{0:?} is not a path inside the folder (relative, no `..`, no empty parts)")]
    BadPath(String),
    /// The target directory already holds files.
    #[error("{0} already holds files; an environment is never written over another")]
    NotEmpty(String),
    /// A symbolic link, device or other entry that is not a plain file.
    #[error("{0} is not a regular file or directory; links are refused so a folder cannot point outside itself")]
    NotRegular(String),
    /// More files or bytes than a folder may hold.
    #[error("{what} exceeds the limit of {limit}")]
    TooLarge {
        /// What was too large.
        what: String,
        /// The limit it passed.
        limit: u64,
    },
    /// The disk refused.
    #[error("{path}: {source}")]
    Io {
        /// What was being touched.
        path: String,
        /// The error.
        source: std::io::Error,
    },
}

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> FolderError + '_ {
    move |source| FolderError::Io {
        path: path.display().to_string(),
        source,
    }
}

/// One file of a folder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// The file's bytes.
    pub bytes: Vec<u8>,
    /// Whether it is executable.
    pub executable: bool,
}

/// A set of files by relative path.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Folder {
    files: BTreeMap<String, Entry>,
}

fn check_path(path: &str) -> Result<(), FolderError> {
    let bad = path.is_empty()
        || path.starts_with('/')
        || path.contains('\\')
        || path.contains('\0')
        || path
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..");
    if bad {
        return Err(FolderError::BadPath(path.to_string()));
    }
    Ok(())
}

impl Folder {
    /// An empty folder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds (or replaces) the file at `path`.
    pub fn put(
        &mut self,
        path: &str,
        bytes: impl Into<Vec<u8>>,
        executable: bool,
    ) -> Result<(), FolderError> {
        check_path(path)?;
        self.files.insert(
            path.to_string(),
            Entry {
                bytes: bytes.into(),
                executable,
            },
        );
        Ok(())
    }

    /// Adds a text file.
    pub fn put_text(&mut self, path: &str, text: &str) -> Result<(), FolderError> {
        self.put(path, text.as_bytes(), false)
    }

    /// Adds an executable script.
    pub fn put_script(&mut self, path: &str, text: &str) -> Result<(), FolderError> {
        self.put(path, text.as_bytes(), true)
    }

    /// The file at `path`.
    #[must_use]
    pub fn get(&self, path: &str) -> Option<&Entry> {
        self.files.get(path)
    }

    /// The text of the file at `path`, when it is UTF-8.
    #[must_use]
    pub fn text(&self, path: &str) -> Option<&str> {
        self.files
            .get(path)
            .and_then(|e| std::str::from_utf8(&e.bytes).ok())
    }

    /// Removes the file at `path`; whether it was there.
    pub fn remove(&mut self, path: &str) -> bool {
        self.files.remove(path).is_some()
    }

    /// Every path, in order.
    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.files.keys().map(String::as_str)
    }

    /// Every file, in path order.
    pub fn entries(&self) -> impl Iterator<Item = (&str, &Entry)> {
        self.files.iter().map(|(p, e)| (p.as_str(), e))
    }

    /// The files under `prefix/` (the prefix included in each path).
    pub fn under<'a>(&'a self, prefix: &'a str) -> impl Iterator<Item = (&'a str, &'a Entry)> {
        self.entries().filter(move |(p, _)| {
            p.strip_prefix(prefix)
                .is_some_and(|rest| rest.starts_with('/'))
        })
    }

    /// How many files the folder holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.files.len()
    }

    /// Whether the folder holds no file.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// The content address of the folder: every path, its executable bit and
    /// the digest of its bytes, in path order.
    #[must_use]
    pub fn digest(&self) -> Digest {
        #[derive(Serialize)]
        struct Line<'a>(&'a str, bool, Digest);
        let lines: Vec<Line<'_>> = self
            .files
            .iter()
            .map(|(p, e)| Line(p, e.executable, Digest::of(&e.bytes)))
            .collect();
        digest_of(&lines)
    }

    /// This folder without the files whose path is `except`.
    #[must_use]
    pub fn without(&self, except: &[&str]) -> Self {
        Self {
            files: self
                .files
                .iter()
                .filter(|(p, _)| !except.contains(&p.as_str()))
                .map(|(p, e)| (p.clone(), e.clone()))
                .collect(),
        }
    }

    /// Writes the folder under `dir`, which must not hold files already.
    pub fn write_to(&self, dir: &Path) -> Result<(), FolderError> {
        if dir.is_dir() && std::fs::read_dir(dir).map_err(io(dir))?.next().is_some() {
            return Err(FolderError::NotEmpty(dir.display().to_string()));
        }
        std::fs::create_dir_all(dir).map_err(io(dir))?;
        for (path, entry) in &self.files {
            let target = dir.join(path);
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent).map_err(io(parent))?;
            }
            std::fs::write(&target, &entry.bytes).map_err(io(&target))?;
            if entry.executable {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o755))
                    .map_err(io(&target))?;
            }
        }
        Ok(())
    }

    /// Reads every file under `dir`; a link, a device, more than
    /// [`MAX_FILES`] files or a file over [`MAX_FILE_BYTES`] is refused.
    pub fn read_from(dir: &Path) -> Result<Self, FolderError> {
        fn walk(root: &Path, at: &Path, into: &mut Folder) -> Result<(), FolderError> {
            let mut names: Vec<_> = std::fs::read_dir(at)
                .map_err(io(at))?
                .collect::<Result<_, _>>()
                .map_err(io(at))?;
            names.sort_by_key(std::fs::DirEntry::file_name);
            for item in names {
                let path = item.path();
                let kind = item.file_type().map_err(io(&path))?;
                if kind.is_dir() {
                    walk(root, &path, into)?;
                } else if kind.is_file() {
                    let meta = item.metadata().map_err(io(&path))?;
                    if meta.len() > MAX_FILE_BYTES {
                        return Err(FolderError::TooLarge {
                            what: path.display().to_string(),
                            limit: MAX_FILE_BYTES,
                        });
                    }
                    if into.len() >= MAX_FILES {
                        return Err(FolderError::TooLarge {
                            what: format!("the file count of {}", root.display()),
                            limit: MAX_FILES as u64,
                        });
                    }
                    use std::os::unix::fs::PermissionsExt;
                    let relative = path
                        .strip_prefix(root)
                        .map_err(|_| FolderError::BadPath(path.display().to_string()))?
                        .to_string_lossy()
                        .into_owned();
                    let bytes = std::fs::read(&path).map_err(io(&path))?;
                    into.put(&relative, bytes, meta.permissions().mode() & 0o111 != 0)?;
                } else {
                    return Err(FolderError::NotRegular(path.display().to_string()));
                }
            }
            Ok(())
        }
        let mut folder = Self::new();
        walk(dir, dir, &mut folder)?;
        Ok(folder)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)] // a panic is the failure report
mod tests {
    use super::*;

    #[test]
    fn a_path_must_stay_inside_the_folder() {
        let mut folder = Folder::new();
        for bad in ["", "/etc/passwd", "../x", "a/../b", "a//b", "a\\b", "./x"] {
            assert!(folder.put_text(bad, "x").is_err(), "{bad:?}");
        }
        assert!(folder.put_text("tests/test.sh", "x").is_ok());
    }

    #[test]
    fn the_digest_follows_content_path_and_executable_bit_and_nothing_else() {
        let mut a = Folder::new();
        a.put_text("a", "1").unwrap();
        a.put_script("b", "2").unwrap();
        let mut b = Folder::new();
        b.put_script("b", "2").unwrap(); // inserted in another order
        b.put_text("a", "1").unwrap();
        assert_eq!(a.digest(), b.digest());
        b.put("b", "2", false).unwrap();
        assert_ne!(a.digest(), b.digest());
        assert_ne!(a.digest(), a.without(&["a"]).digest());
    }

    #[test]
    fn a_folder_round_trips_through_disk_and_is_never_written_over_another() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("task");
        let mut folder = Folder::new();
        folder.put_text("instruction.md", "do it").unwrap();
        folder.put_script("tests/test.sh", "#!/bin/sh\n").unwrap();
        folder.write_to(&target).unwrap();
        let back = Folder::read_from(&target).unwrap();
        assert_eq!(back, folder);
        assert!(matches!(
            folder.write_to(&target),
            Err(FolderError::NotEmpty(_))
        ));
    }

    #[test]
    fn a_link_in_a_folder_on_disk_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink("/etc/hostname", dir.path().join("link")).unwrap();
        assert!(matches!(
            Folder::read_from(dir.path()),
            Err(FolderError::NotRegular(_))
        ));
    }

    #[test]
    fn under_lists_only_a_directory_not_a_name_that_starts_the_same() {
        let mut folder = Folder::new();
        folder.put_text("tests/a", "").unwrap();
        folder.put_text("tests2/b", "").unwrap();
        let listed: Vec<_> = folder.under("tests").map(|(p, _)| p).collect();
        assert_eq!(listed, ["tests/a"]);
    }
}
