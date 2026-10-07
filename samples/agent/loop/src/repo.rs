// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements self-improving coding agents whose every
// step is auditable, for its clients. If your team needs expertise in agent
// evaluation or locally operated coding agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The target repository, and the isolated checkout the worker edits.
//!
//! The worker never touches the repository the task names: it works in a
//! detached git worktree of the baseline revision, kept in the run's own
//! directory. What it changed is read back from git (a patch, and for each
//! file its status and the content hashes before and after), not from what
//! the worker says it did. A worktree is isolation from accidents, not a
//! sandbox: the shell tool still runs as the user.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use splinter_sdk::vocabulary::digest::Digest;

/// Excludes sven's own working directory (its audit log and logs, written
/// into the project root) and Python bytecode caches from every view of what
/// the worker changed.
const EXCLUDE_SVEN: [&str; 4] = [
    "--",
    ".",
    ":(exclude).sven",
    ":(exclude,glob)**/__pycache__/**",
];

/// What the loop did to one file, with the content hash before and after
/// (absent where the file did not exist).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileChange {
    /// Path relative to the repository root.
    pub path: String,
    /// `added`, `modified`, `deleted` or `renamed`.
    pub status: String,
    /// SHA-256 of the baseline content.
    pub before: Option<String>,
    /// SHA-256 of the final content.
    pub after: Option<String>,
}

/// Runs `git` in `dir` and returns its standard output.
pub fn git(dir: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .with_context(|| format!("running git {args:?} in {}", dir.display()))?;
    if !output.status.success() {
        bail!(
            "git {args:?} in {} failed: {}",
            dir.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// The revision `dir` is at, which must be a git working tree.
pub fn head_rev(dir: &Path) -> Result<String> {
    Ok(git(dir, &["rev-parse", "HEAD"])?.trim().to_string())
}

/// Whether `dir` has no uncommitted change (untracked files count).
pub fn is_clean(dir: &Path) -> Result<bool> {
    Ok(git(dir, &["status", "--porcelain", "-uall"])?
        .trim()
        .is_empty())
}

/// A detached checkout of `rev` of `repo` at `dest`.
pub fn add_worktree(repo: &Path, dest: &Path, rev: &str) -> Result<()> {
    let dest_text = dest.to_string_lossy();
    git(repo, &["worktree", "add", "--detach", &dest_text, rev]).map(|_| ())
}

/// The checkout at `work` put back to exactly `rev`: no edit, no untracked
/// file survives (ignored build products are kept: they are caches), so a retry starts from a known baseline.
pub fn reset_to(work: &Path, rev: &str) -> Result<()> {
    git(work, &["reset", "--hard", rev])?;
    git(work, &[&["clean", "-fd"][..], &EXCLUDE_SVEN].concat()).map(|_| ())
}

/// Removes the checkout at `work` from `repo`.
pub fn remove_worktree(repo: &Path, work: &Path) -> Result<()> {
    let work_text = work.to_string_lossy();
    git(repo, &["worktree", "remove", "--force", &work_text]).map(|_| ())
}

/// SHA-256 of the file at `path`, or `None` when it does not exist.
fn hash_file(path: &Path) -> Result<Option<String>> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(Some(Digest::sha256_of(&bytes).to_string())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

/// SHA-256 of `path` as it is in `rev`, or `None` when it is not there.
pub(crate) fn hash_in_rev(work: &Path, rev: &str, path: &str) -> Result<Option<String>> {
    let spec = format!("{rev}:{path}");
    let output = Command::new("git")
        .arg("-C")
        .arg(work)
        .args(["show", &spec])
        .output()
        .context("running git show")?;
    Ok(output
        .status
        .success()
        .then(|| Digest::sha256_of(&output.stdout).to_string()))
}

/// Everything the checkout at `work` differs from `rev` by, as a patch and
/// as the list of changed files. Staging is done in the worktree's own
/// index; the repository the task names is never touched.
pub fn changes_since(work: &Path, rev: &str) -> Result<(String, Vec<FileChange>)> {
    git(work, &[&["add", "-A"][..], &EXCLUDE_SVEN].concat())?;
    let patch = git(
        work,
        &[&["diff", "--cached", "--binary", rev][..], &EXCLUDE_SVEN].concat(),
    )?;
    let listing = git(
        work,
        &[
            &["diff", "--cached", "--name-status", "-z", rev][..],
            &EXCLUDE_SVEN,
        ]
        .concat(),
    )?;
    let mut parts = listing.split('\0').filter(|p| !p.is_empty());
    let mut changes = Vec::new();
    while let Some(code) = parts.next() {
        let status = match code.chars().next() {
            Some('A') => "added",
            Some('D') => "deleted",
            Some('R') => "renamed",
            _ => "modified",
        };
        let first = parts.next().context("a name-status record has a path")?;
        let path = if status == "renamed" {
            parts.next().context("a rename has a target")?
        } else {
            first
        };
        let before_path = if status == "renamed" { first } else { path };
        changes.push(FileChange {
            path: path.to_string(),
            status: status.to_string(),
            before: hash_in_rev(work, rev, before_path)?,
            after: hash_file(&work.join(path))?,
        });
    }
    Ok((patch, changes))
}

/// The content hash of every file that differs from the last commit (new
/// files included), by path; the loop diffs two of these to name what a
/// tool call changed.
pub fn dirty_snapshot(work: &Path) -> Result<BTreeMap<String, Option<String>>> {
    let listing = git(
        work,
        &[&["status", "--porcelain", "-uall", "-z"][..], &EXCLUDE_SVEN].concat(),
    )?;
    let mut snapshot = BTreeMap::new();
    let mut parts = listing.split('\0').filter(|p| p.len() > 3);
    while let Some(entry) = parts.next() {
        let status = &entry[..2];
        let path = &entry[3..];
        if status.starts_with('R') || status.starts_with('C') {
            // The next record is the rename's source.
            let _ = parts.next();
        }
        snapshot.insert(path.to_string(), hash_file(&work.join(path))?);
    }
    Ok(snapshot)
}

/// The paths among `changes` that fall under any of `protected` (each a
/// file or a directory prefix, relative to the repository root).
#[must_use]
pub fn touched_protected(changes: &[FileChange], protected: &[String]) -> Vec<String> {
    changes
        .iter()
        .filter(|c| {
            protected.iter().any(|p| {
                let p = p.trim_end_matches('/');
                c.path == p || c.path.starts_with(&format!("{p}/"))
            })
        })
        .map(|c| c.path.clone())
        .collect()
}

/// The directory that holds a run's checkout.
#[must_use]
pub fn work_dir(run_dir: &Path) -> PathBuf {
    run_dir.join(crate::store::WORK_DIR)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init(dir: &Path) {
        git(dir, &["init", "-q", "-b", "main"]).unwrap();
        git(dir, &["config", "user.email", "t@example.com"]).unwrap();
        git(dir, &["config", "user.name", "t"]).unwrap();
        std::fs::write(dir.join("a.txt"), "one\n").unwrap();
        std::fs::write(dir.join("b.txt"), "keep\n").unwrap();
        git(dir, &["add", "-A"]).unwrap();
        git(dir, &["commit", "-q", "-m", "base"]).unwrap();
    }

    #[test]
    fn changes_are_read_from_git_with_hashes_and_a_reset_restores_the_baseline() {
        let repo = tempfile::tempdir().unwrap();
        init(repo.path());
        let rev = head_rev(repo.path()).unwrap();
        let outer = tempfile::tempdir().unwrap();
        let work = outer.path().join("work");
        add_worktree(repo.path(), &work, &rev).unwrap();

        std::fs::write(work.join("a.txt"), "two\n").unwrap();
        std::fs::write(work.join("new.txt"), "n\n").unwrap();
        std::fs::remove_file(work.join("b.txt")).unwrap();
        let (patch, changes) = changes_since(&work, &rev).unwrap();

        assert!(patch.contains("+two"), "{patch}");
        let by_path: BTreeMap<_, _> = changes.iter().map(|c| (c.path.as_str(), c)).collect();
        assert_eq!(by_path["a.txt"].status, "modified");
        assert!(by_path["a.txt"].before.is_some() && by_path["a.txt"].after.is_some());
        assert_ne!(by_path["a.txt"].before, by_path["a.txt"].after);
        assert_eq!(by_path["new.txt"].status, "added");
        assert!(by_path["new.txt"].before.is_none());
        assert_eq!(by_path["b.txt"].status, "deleted");
        assert!(by_path["b.txt"].after.is_none());
        assert!(
            is_clean(repo.path()).unwrap(),
            "the target repository is untouched"
        );

        reset_to(&work, &rev).unwrap();
        assert!(is_clean(&work).unwrap());
        assert_eq!(
            std::fs::read_to_string(work.join("a.txt")).unwrap(),
            "one\n"
        );
    }

    #[test]
    fn a_dirty_snapshot_names_what_changed_between_two_points() {
        let repo = tempfile::tempdir().unwrap();
        init(repo.path());
        let before = dirty_snapshot(repo.path()).unwrap();
        std::fs::write(repo.path().join("a.txt"), "two\n").unwrap();
        let after = dirty_snapshot(repo.path()).unwrap();
        assert!(before.is_empty());
        assert_eq!(after.keys().collect::<Vec<_>>(), ["a.txt"]);
    }

    #[test]
    fn protected_paths_match_files_and_directory_prefixes_only() {
        let change = |p: &str| FileChange {
            path: p.into(),
            status: "modified".into(),
            before: None,
            after: None,
        };
        let changes = [change("tests/t.py"), change("src/a.py"), change("tests2/x")];
        let hit = touched_protected(&changes, &["tests/".into(), "src/a.py".into()]);
        assert_eq!(hit, ["tests/t.py", "src/a.py"]);
    }
}
