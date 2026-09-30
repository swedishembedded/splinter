// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements source capture that grounds every learned
// fact in exact, traceable bytes, for its clients. If your team needs
// expertise in knowledge acquisition or training-data provenance, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A directory tree captured as a repository source.

use std::fs;
use std::path::Path;
use std::process::{Command, Output, Stdio};

use splinter_store::clock::Clock;
use splinter_store::source::{CapturedSource, Origin, PartContent, Revision, SkipReason, Skipped};

use super::{io, is_text, read_capped, text_media_type, utf8, CaptureError};

/// Entries of a tree a repository capture never descends into or reads,
/// at any depth: version control state, build output and installed
/// dependencies. They are not listed as skipped.
pub const IGNORED_NAMES: &[&str] = &[".git", "target", "node_modules"];

/// Variables that point git at a repository other than the one in the
/// working directory (a hook, for one, sets them); removed so git reads the
/// captured tree's own state.
const GIT_LOCATION_VARIABLES: &[&str] = &[
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
];

/// Captures the directory tree at `path`: every regular UTF-8 text file
/// under it is a part named by its `/`-separated path relative to `path`.
/// [`IGNORED_NAMES`] are not entered; symbolic links are not followed.
/// A file over `max_file_bytes`, a file that is not text, and an entry that
/// is neither a regular file nor a directory are recorded as skipped. When
/// `path` is in a git work tree, `HEAD` and whether the tree under `path`
/// is dirty are recorded, read with read-only git commands. Parts and
/// skipped entries are in path order, whatever order the filesystem lists
/// them in.
pub fn capture_repository(
    path: &Path,
    max_file_bytes: u64,
    clock: &dyn Clock,
) -> Result<CapturedSource, CaptureError> {
    let root = path.canonicalize().map_err(io(path))?;
    if !root.metadata().map_err(io(&root))?.is_dir() {
        return Err(CaptureError::NotADirectory { path: root });
    }
    let mut walk = Walk {
        max_file_bytes,
        parts: Vec::new(),
        skipped: Vec::new(),
    };
    walk.dir(&root, "")?;
    walk.skipped.sort_by(|a, b| a.path.cmp(&b.path));
    let origin = Origin::Repository {
        path: utf8(&root)?.to_string(),
        revision: git_revision(&root)?,
        skipped: walk.skipped,
    };
    Ok(CapturedSource::new(origin, walk.parts, clock)?)
}

struct Walk {
    max_file_bytes: u64,
    parts: Vec<PartContent>,
    skipped: Vec<Skipped>,
}

impl Walk {
    /// Walks `dir`, whose path relative to the tree's root is `prefix`
    /// (empty, or ending in `/`).
    fn dir(&mut self, dir: &Path, prefix: &str) -> Result<(), CaptureError> {
        let mut entries = Vec::new();
        for entry in fs::read_dir(dir).map_err(io(dir))? {
            entries.push(entry.map_err(io(dir))?);
        }
        entries.sort_by_key(fs::DirEntry::file_name);
        for entry in entries {
            let file_name = entry.file_name();
            let Some(name) = file_name.to_str() else {
                self.skip(
                    format!("{prefix}{}", file_name.to_string_lossy()),
                    SkipReason::NonUtf8Name,
                );
                continue;
            };
            if IGNORED_NAMES.contains(&name) {
                continue;
            }
            let rel = format!("{prefix}{name}");
            let path = entry.path();
            // The entry itself, not what a symbolic link points to: a link
            // may leave the tree or loop back into it.
            let meta = fs::symlink_metadata(&path).map_err(io(&path))?;
            if meta.is_dir() {
                self.dir(&path, &format!("{rel}/"))?;
            } else if !meta.is_file() {
                self.skip(rel, SkipReason::NotRegular);
            } else if meta.len() > self.max_file_bytes {
                self.skip(rel, SkipReason::TooLarge { bytes: meta.len() });
            } else {
                self.file(&path, rel)?;
            }
        }
        Ok(())
    }

    fn file(&mut self, path: &Path, rel: String) -> Result<(), CaptureError> {
        let bytes = read_capped(path, self.max_file_bytes)?;
        let len = bytes.len() as u64;
        if len > self.max_file_bytes {
            // It grew between the listing and the read.
            self.skip(rel, SkipReason::TooLarge { bytes: len });
        } else if !is_text(&bytes) {
            self.skip(rel, SkipReason::NotText);
        } else {
            self.parts.push(PartContent {
                media_type: text_media_type(&rel).to_string(),
                name: rel,
                bytes,
            });
        }
        Ok(())
    }

    fn skip(&mut self, path: String, reason: SkipReason) {
        self.skipped.push(Skipped { path, reason });
    }
}

/// Runs `git args` in `dir`, isolated from any repository the caller's
/// environment points at.
fn git(dir: &Path, args: &[&str]) -> std::io::Result<Output> {
    let mut command = Command::new("git");
    command
        .current_dir(dir)
        .args(args)
        .stdin(Stdio::null())
        // Read-only: never take the index lock to refresh it.
        .env("GIT_OPTIONAL_LOCKS", "0");
    for variable in GIT_LOCATION_VARIABLES {
        command.env_remove(variable);
    }
    command.output()
}

fn git_failure(dir: &Path, args: &[&str], output: &Output) -> CaptureError {
    CaptureError::Git {
        args: args.iter().map(|a| a.to_string()).collect(),
        path: dir.to_path_buf(),
        detail: format!(
            "{}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ),
    }
}

/// The git state of the tree at `dir`: `None` when it is not in a work
/// tree. A tree with its own `.git` must be readable by git; any other
/// tree is taken as outside version control when git cannot tell.
fn git_revision(dir: &Path) -> Result<Option<Revision>, CaptureError> {
    let owns_repository = dir.join(".git").exists();
    let probe = ["rev-parse", "--is-inside-work-tree"];
    let inside = match git(dir, &probe) {
        Ok(output) => output,
        Err(source) if owns_repository => {
            return Err(CaptureError::GitUnavailable {
                path: dir.to_path_buf(),
                source,
            })
        }
        Err(_) => return Ok(None),
    };
    if !inside.status.success() {
        return if owns_repository {
            Err(git_failure(dir, &probe, &inside))
        } else {
            Ok(None)
        };
    }
    if String::from_utf8_lossy(&inside.stdout).trim() != "true" {
        return Ok(None);
    }
    // No HEAD commit (a repository with nothing committed yet) is a
    // failure of `--verify`, and a revision without a commit.
    let head = git(dir, &["rev-parse", "--verify", "--quiet", "HEAD"]).map_err(io(dir))?;
    let commit = head
        .status
        .success()
        .then(|| String::from_utf8_lossy(&head.stdout).trim().to_string());
    let status_args = ["status", "--porcelain", "--untracked-files=all", "--", "."];
    let status = git(dir, &status_args).map_err(io(dir))?;
    if !status.status.success() {
        return Err(git_failure(dir, &status_args, &status));
    }
    Ok(Some(Revision {
        commit,
        dirty: !status.stdout.is_empty(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Git state is only claimed for a work tree: a plain directory has
    /// none, and a `.git` that git cannot read is an error, not a clean
    /// tree.
    #[test]
    fn a_tree_outside_git_has_no_revision_and_a_broken_one_is_an_error() {
        if git(&std::env::temp_dir(), &["--version"]).is_err() {
            eprintln!("SKIPPED: git is not installed");
            return;
        }
        let dir = std::env::temp_dir().join(format!("splinter-repo-plain-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        assert!(git_revision(&dir).unwrap().is_none());
        fs::write(dir.join(".git"), "gitdir: /nonexistent\n").unwrap();
        assert!(matches!(git_revision(&dir), Err(CaptureError::Git { .. })));
        fs::remove_dir_all(&dir).unwrap();
    }
}
