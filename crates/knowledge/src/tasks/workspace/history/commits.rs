// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements discovery of training tasks in a
// repository's commit history, for its clients. If your team needs expertise
// in turning a codebase's history into executable benchmarks, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Reading a revision range: its commits, what each changed, which commits
//! cannot be tasks (merges, formatting-only, generated, vendored, test-less)
//! and the request and solution a commit that can be becomes.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::Serialize;
use splinter_core::digest::{canonical_json, Digest};

use super::super::profile::is_test_path;
use super::{
    Halt, HistoryReason, HistoryRequest, WorkspaceError, WorkspaceGenerator, WorkspaceTask,
};
use crate::tasks::dedup::Seen;

/// The shortest added line, in characters, whose appearance in a request is
/// a leak of the solution.
const MIN_LEAKED_LINE_CHARS: usize = 6;

/// How much of a changed file is read to look for a generated-content marker.
const GENERATED_MARKER_BYTES: usize = 4096;

/// Directory names whose files are third-party vendored code.
const VENDORED_DIRS: [&str; 6] = [
    "vendor",
    "vendored",
    "third_party",
    "third-party",
    "external",
    "deps",
];

/// Directory names whose files are build output or generated.
const GENERATED_DIRS: [&str; 8] = [
    "dist",
    "build",
    "target",
    "node_modules",
    "__pycache__",
    ".venv",
    "gen",
    "generated",
];

/// File suffixes of generated files.
const GENERATED_SUFFIXES: [&str; 8] = [
    ".min.js",
    ".min.css",
    ".pb.go",
    "_pb2.py",
    ".g.dart",
    ".lock",
    ".snap",
    ".generated",
];

/// File names of generated or lock files.
const GENERATED_NAMES: [&str; 5] = [
    "Cargo.lock",
    "package-lock.json",
    "yarn.lock",
    "poetry.lock",
    "uv.lock",
];

/// Markers a generated file carries in its first bytes.
const GENERATED_MARKERS: [&str; 4] = [
    "code generated",
    "do not edit",
    "@generated",
    "automatically generated",
];

/// A commit considered, before qualification.
pub(super) struct Commit {
    pub(super) sha: String,
    pub(super) parents: Vec<String>,
    pub(super) message: String,
    pub(super) changes: Vec<Change>,
}

impl Commit {
    pub(super) fn parent(&self) -> Option<&str> {
        self.parents.first().map(String::as_str)
    }
}

/// One changed path of a commit.
pub(super) struct Change {
    /// `A`, `M`, `D`, or a rename/copy status starting with `R` or `C`. The
    /// path is always the new one.
    pub(super) status: char,
    pub(super) path: String,
}

impl WorkspaceGenerator {
    /// The commits of `request.range`, newest first, up to the bound.
    pub(super) fn discover(
        &self,
        request: &HistoryRequest<'_>,
    ) -> Result<Vec<Commit>, WorkspaceError> {
        let listing = git(
            request.repository,
            &["rev-list", "--parents", request.range],
        )?;
        let mut commits = Vec::new();
        for line in listing.lines() {
            if commits.len() >= request.max_commits {
                break;
            }
            let mut words = line.split_whitespace();
            let Some(sha) = words.next() else {
                continue;
            };
            let parents: Vec<String> = words.map(str::to_string).collect();
            let message = git(request.repository, &["show", "-s", "--format=%B", sha])?;
            let changes =
                self.changes(request.repository, sha, parents.first().map(String::as_str))?;
            commits.push(Commit {
                sha: sha.to_string(),
                parents,
                message,
                changes,
            });
        }
        Ok(commits)
    }

    /// The changed paths of `sha` against `parent` (against the empty tree
    /// for a root commit), with rename detection.
    pub(super) fn changes(
        &self,
        repository: &Path,
        sha: &str,
        parent: Option<&str>,
    ) -> Result<Vec<Change>, WorkspaceError> {
        let listing = match parent {
            Some(parent) => git(
                repository,
                &[
                    "diff-tree",
                    "--no-commit-id",
                    "--name-status",
                    "-r",
                    "-M",
                    parent,
                    sha,
                ],
            )?,
            None => git(
                repository,
                &[
                    "diff-tree",
                    "--no-commit-id",
                    "--name-status",
                    "-r",
                    "-M",
                    "--root",
                    sha,
                ],
            )?,
        };
        Ok(listing
            .lines()
            .filter_map(|line| {
                let mut parts = line.split('\t');
                let status = parts.next()?.chars().next()?;
                // A rename or copy names the old path first and the new one
                // second; the new path is what the commit changed.
                let path = parts.next_back()?.trim();
                (!path.is_empty()).then(|| Change {
                    status,
                    path: path.to_string(),
                })
            })
            .collect())
    }

    /// Why the commit cannot be a task, if it cannot.
    pub(super) fn excluded(&self, repository: &Path, commit: &Commit) -> Option<HistoryReason> {
        if commit.parents.is_empty() {
            return Some(HistoryReason::NoParent);
        }
        if commit.parents.len() > 1 {
            return Some(HistoryReason::Merge);
        }
        if commit.changes.is_empty() {
            return Some(HistoryReason::NoSourceChange);
        }
        let paths: Vec<&str> = commit.changes.iter().map(|c| c.path.as_str()).collect();
        if paths.iter().all(|p| is_vendored_path(p)) {
            return Some(HistoryReason::Vendored);
        }
        if paths
            .iter()
            .all(|p| is_generated_path(p) || generated_content(repository, &commit.sha, p))
        {
            return Some(HistoryReason::Generated);
        }
        if let Some(parent) = commit.parent() {
            if formatting_only(repository, parent, &commit.sha, &paths) {
                return Some(HistoryReason::FormattingOnly);
            }
        }
        if !paths.iter().any(|p| is_test_path(p)) {
            return Some(HistoryReason::NoTestChange);
        }
        if !paths.iter().any(|p| is_source_path(p)) {
            return Some(HistoryReason::NoSourceChange);
        }
        None
    }
}

/// The module a unittest test file is imported as: `tests/test_x.py` is
/// `tests.test_x`.
pub(super) fn module_id(file: &str) -> String {
    file.trim_end_matches(".py").replace('/', ".")
}

/// What identifies a history task: the commit, its parent and the family.
pub(super) fn history_key(commit: &Commit, family: &str) -> Result<Digest, WorkspaceError> {
    #[derive(Serialize)]
    struct Key<'a> {
        commit: &'a str,
        parent: &'a str,
        family: &'a str,
    }
    Ok(Digest::of(&canonical_json(&Key {
        commit: &commit.sha,
        parent: commit.parent().unwrap_or_default(),
        family,
    })?))
}

pub(super) fn remember(seen: &mut BTreeMap<String, Seen>, task: &WorkspaceTask) {
    if let Some(seen) = seen.get_mut(&task.family) {
        seen.admit(&task.task.instruction);
    }
}

/// The change from `parent` to `commit` restricted to `sources`, as a git
/// diff the workspace verifier can apply.
pub(super) fn diff_between(
    repository: &Path,
    parent: &str,
    commit: &str,
    sources: &[String],
) -> Result<String, Halt> {
    let mut args = vec!["diff", "--no-color", "--no-ext-diff", parent, commit, "--"];
    args.extend(sources.iter().map(String::as_str));
    Ok(git(repository, &args)?)
}

/// The request: the commit message and the changed source files, never the
/// diff and never a hidden test's path.
pub(super) fn describe(
    commit: &Commit,
    sources: &[String],
    summaries: &[String],
    tests: &[String],
    solution: &str,
) -> Result<String, HistoryReason> {
    let mut kept: Vec<String> = Vec::new();
    for line in commit.message.lines() {
        let line = line.trim();
        if line.is_empty() {
            if kept.last().is_some_and(|last| !last.is_empty()) {
                kept.push(String::new());
            }
            continue;
        }
        if line.starts_with("diff --git")
            || line.starts_with("@@")
            || line.starts_with("+++")
            || line.starts_with("---")
            || line.starts_with("Signed-off-by:")
            || line.starts_with("Co-authored-by:")
        {
            continue;
        }
        kept.push(line.to_string());
    }
    while kept.last().is_some_and(String::is_empty) {
        kept.pop();
    }
    let body = kept.join("\n");
    let changed = if summaries.is_empty() {
        sources.join(", ")
    } else {
        summaries.join(", ")
    };
    let text = if body.is_empty() {
        format!("Files changed: {changed}")
    } else {
        format!("{body}\n\nFiles changed: {changed}")
    };
    let lowered = text.to_lowercase();
    if added_lines(solution)
        .iter()
        .any(|line| lowered.contains(&line.to_lowercase()))
    {
        return Err(HistoryReason::LeaksChange);
    }
    for path in tests {
        let module = path.trim_end_matches(".py").replace('/', ".");
        let name = path.rsplit('/').next().unwrap_or(path);
        if text.contains(path)
            || (!module.is_empty() && text.contains(&module))
            || (name.len() > 3 && text.contains(name))
        {
            return Err(HistoryReason::LeaksTest);
        }
    }
    Ok(text)
}

/// The lines a diff adds, trimmed, leaving out those too short to say
/// anything.
fn added_lines(diff: &str) -> BTreeSet<String> {
    diff.lines()
        .filter(|line| line.starts_with('+') && !line.starts_with("+++"))
        .map(|line| line[1..].trim().to_string())
        .filter(|line| line.chars().count() >= MIN_LEAKED_LINE_CHARS)
        .collect()
}

/// Runs `git args` in `repository` and returns its standard output, refused
/// when the command fails.
fn git(repository: &Path, args: &[&str]) -> Result<String, WorkspaceError> {
    let failed = |detail: String| WorkspaceError::Git {
        args: args.iter().map(|a| (*a).to_string()).collect(),
        repository: repository.to_path_buf(),
        detail,
    };
    let output = crate::capture::git::read_only(repository)
        .args(args)
        .output()
        .map_err(|source| WorkspaceError::Io {
            path: repository.to_path_buf(),
            source,
        })?;
    if !output.status.success() {
        return Err(failed(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Whether the change between `parent` and `sha`, restricted to `paths`,
/// makes no difference beyond whitespace and blank lines.
fn formatting_only(repository: &Path, parent: &str, sha: &str, paths: &[&str]) -> bool {
    let mut args = vec![
        "diff",
        "-w",
        "--ignore-blank-lines",
        "--quiet",
        parent,
        sha,
        "--",
    ];
    args.extend(paths);
    crate::capture::git::read_only(repository)
        .args(&args)
        .output()
        .is_ok_and(|output| output.status.success())
}

/// Whether `path` is beneath a directory of third-party code.
fn is_vendored_path(path: &str) -> bool {
    path.split('/').any(|part| VENDORED_DIRS.contains(&part))
}

/// Whether `path` is a generated file by its location, suffix or name.
fn is_generated_path(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    path.split('/').any(|part| GENERATED_DIRS.contains(&part))
        || GENERATED_SUFFIXES
            .iter()
            .any(|suffix| path.ends_with(suffix))
        || GENERATED_NAMES.contains(&name)
}

/// Whether the file `path` at `sha` carries a generated-content marker.
fn generated_content(repository: &Path, sha: &str, path: &str) -> bool {
    let Ok(text) = git(repository, &["show", &format!("{sha}:{path}")]) else {
        return false;
    };
    let head = &text[..text.len().min(GENERATED_MARKER_BYTES)];
    let lowered = head.to_lowercase();
    GENERATED_MARKERS
        .iter()
        .any(|marker| lowered.contains(marker))
}

/// Whether `path` is a source file of the project, not a test.
pub(super) fn is_source_path(path: &str) -> bool {
    !is_test_path(path)
        && (path.ends_with(".py") || (path.starts_with("src/") && path.ends_with(".rs")))
        && !is_vendored_path(path)
        && !is_generated_path(path)
}
