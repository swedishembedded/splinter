// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Reading a repository's history without touching it.
//!
//! splinter depends on no Rust git library, and its other crates (the
//! repository capture, the harbor adapter) shell out to the `git` binary, so
//! this one does too. Everything here reads *objects* (`rev-list`, `log`,
//! `diff-tree`, `archive`): no command refreshes an index or writes a ref, and
//! the source is first cloned bare into the state directory ([`mirror`]) so
//! the rest of the run never touches it at all.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{bail, Context, Result};

/// A git repository (usually a bare mirror) read through the `git` binary.
#[derive(Clone, Debug)]
pub struct Git {
    dir: PathBuf,
}

/// How a commit changed one path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The path did not exist in the parent.
    Added,
    /// The path exists in both with other content or mode.
    Modified,
    /// The path does not exist in the commit.
    Deleted,
    /// The content moved from `old_path` (possibly edited).
    Renamed,
}

/// One path a commit changes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Change {
    /// What happened to the path.
    pub kind: Kind,
    /// The path in the commit (in the parent, for a deletion).
    pub path: String,
    /// The path in the parent, for a rename.
    pub old_path: Option<String>,
    /// Lines added, `None` for a binary file.
    pub added: Option<usize>,
    /// Lines removed, `None` for a binary file.
    pub removed: Option<usize>,
    /// A submodule (gitlink) or symlink entry: not reproducible with file tools.
    pub special: bool,
}

impl Change {
    /// Whether git treats the file as binary.
    #[must_use]
    pub fn binary(&self) -> bool {
        self.added.is_none()
    }

    /// Lines touched (added plus removed); zero for a binary file.
    #[must_use]
    pub fn lines(&self) -> usize {
        self.added.unwrap_or(0) + self.removed.unwrap_or(0)
    }
}

/// A commit on the first-parent chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitMeta {
    /// Full object name.
    pub sha: String,
    /// Parents, in order; a commit with exactly one is replayable.
    pub parents: Vec<String>,
    /// Author name, as scrubbing needs it.
    pub author_name: String,
    /// Author address, as scrubbing needs it.
    pub author_email: String,
    /// The whole message.
    pub message: String,
}

fn command(dir: &Path) -> Command {
    let mut c = Command::new("git");
    // The run is configured by flags and the state it was given, not by the
    // reader's `~/.gitconfig` (a signing key, a pager, an `autocrlf`).
    c.env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_SYSTEM", "/dev/null")
        .env("GIT_TERMINAL_PROMPT", "0")
        .arg("-C")
        .arg(dir);
    c
}

impl Git {
    /// The repository at `dir`.
    #[must_use]
    pub fn at(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// Where it is.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// `git <args>` and its standard output, or the error with git's own words.
    pub fn bytes(&self, args: &[&str]) -> Result<Vec<u8>> {
        let out = command(&self.dir)
            .args(args)
            .stdin(Stdio::null())
            .output()
            .with_context(|| format!("running git {}", args.join(" ")))?;
        if !out.status.success() {
            bail!(
                "git {} failed: {}",
                args.join(" "),
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        Ok(out.stdout)
    }

    /// [`Git::bytes`] as text.
    pub fn text(&self, args: &[&str]) -> Result<String> {
        Ok(String::from_utf8_lossy(&self.bytes(args)?).into_owned())
    }

    /// The first-parent chain of `to` (default `HEAD`), oldest first, after
    /// `from` when it is given (`from` itself is not included).
    pub fn first_parent_chain(
        &self,
        from: Option<&str>,
        to: Option<&str>,
    ) -> Result<Vec<CommitMeta>> {
        let to = to.unwrap_or("HEAD");
        let range = match from {
            Some(from) => format!("{from}..{to}"),
            None => to.to_string(),
        };
        let raw = self.text(&[
            "log",
            "--first-parent",
            "--reverse",
            "--format=%H%x1f%P%x1f%an%x1f%ae%x1f%B%x1e",
            &range,
        ])?;
        let mut commits = Vec::new();
        for record in raw.split('\u{1e}') {
            let record = record.trim_start_matches('\n');
            if record.trim().is_empty() {
                continue;
            }
            let fields: Vec<&str> = record.splitn(5, '\u{1f}').collect();
            let [sha, parents, name, email, message] = fields[..] else {
                bail!("unreadable git log record {record:?}");
            };
            commits.push(CommitMeta {
                sha: sha.to_string(),
                parents: parents.split_whitespace().map(String::from).collect(),
                author_name: name.to_string(),
                author_email: email.to_string(),
                message: message.trim_end().to_string(),
            });
        }
        Ok(commits)
    }

    /// The paths `commit` changes against `parent`, renames detected.
    pub fn changes(&self, parent: &str, commit: &str) -> Result<Vec<Change>> {
        let raw = self.bytes(&[
            "diff-tree",
            "-r",
            "-M",
            "-z",
            "--raw",
            "--no-commit-id",
            parent,
            commit,
        ])?;
        let numstat = self.bytes(&["diff", "--numstat", "-M", "-z", parent, commit])?;
        let stats = parse_numstat(&numstat);
        let mut fields = raw
            .split(|b| *b == 0)
            .map(|f| String::from_utf8_lossy(f).into_owned());
        let mut changes = Vec::new();
        while let Some(head) = fields.next() {
            if head.is_empty() {
                continue;
            }
            // `:<old mode> <new mode> <old sha> <new sha> <status>[score]`
            let parts: Vec<&str> = head.trim_start_matches(':').split(' ').collect();
            let [old_mode, new_mode, _, _, status] = parts[..] else {
                bail!("unreadable diff-tree entry {head:?}");
            };
            let first = fields.next().context("diff-tree entry without a path")?;
            let (kind, path, old_path) = match status.chars().next() {
                Some('A') => (Kind::Added, first, None),
                Some('D') => (Kind::Deleted, first, None),
                Some('R') => {
                    let second = fields.next().context("rename without a target")?;
                    (Kind::Renamed, second, Some(first))
                }
                Some('M' | 'T') => (Kind::Modified, first, None),
                other => bail!("unsupported diff-tree status {other:?} for {first}"),
            };
            let special = [old_mode, new_mode]
                .iter()
                .any(|m| m.starts_with("160") || m.starts_with("120"));
            let (added, removed) = stats.get(path.as_str()).copied().unwrap_or((None, None));
            changes.push(Change {
                kind,
                path,
                old_path,
                added,
                removed,
                special,
            });
        }
        Ok(changes)
    }

    /// The unified diff of `paths` between `parent` and `commit`, as a person
    /// reads it (renames detected, no colour).
    pub fn diff(&self, parent: &str, commit: &str, paths: &[String]) -> Result<String> {
        let mut args = vec![
            "diff",
            "--no-color",
            "-M",
            "--src-prefix=a/",
            "--dst-prefix=b/",
            parent,
            commit,
            "--",
        ];
        args.extend(paths.iter().map(String::as_str));
        self.text(&args)
    }

    /// The first `limit` bytes of the file `path` at `rev`, empty when it is absent.
    pub fn head_of_blob(&self, rev: &str, path: &str, limit: usize) -> String {
        let spec = format!("{rev}:{path}");
        self.bytes(&["cat-file", "blob", &spec])
            .map(|b| String::from_utf8_lossy(&b[..b.len().min(limit)]).into_owned())
            .unwrap_or_default()
    }

    /// Lays out the tree of `rev` in `dest` (which must exist), tracked files
    /// only, as `git archive` writes it.
    pub fn export_tree(&self, rev: &str, dest: &Path) -> Result<()> {
        let mut archive = command(&self.dir)
            .args(["archive", "--format=tar", rev])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("starting git archive")?;
        let stdout = archive.stdout.take().context("git archive has no output")?;
        let extracted = Command::new("tar")
            .arg("-x")
            .arg("-C")
            .arg(dest)
            .stdin(Stdio::from(stdout))
            .output()
            .context("starting tar")?;
        let done = archive
            .wait_with_output()
            .context("waiting for git archive")?;
        if !done.status.success() {
            bail!(
                "git archive {rev} failed: {}",
                String::from_utf8_lossy(&done.stderr).trim()
            );
        }
        if !extracted.status.success() {
            bail!(
                "tar failed: {}",
                String::from_utf8_lossy(&extracted.stderr).trim()
            );
        }
        Ok(())
    }
}

/// `path -> (added, removed)` from `git diff --numstat -z`; binary files are
/// `(None, None)`. A rename is keyed by its new path.
fn parse_numstat(raw: &[u8]) -> std::collections::HashMap<String, (Option<usize>, Option<usize>)> {
    let mut stats = std::collections::HashMap::new();
    let mut fields = raw
        .split(|b| *b == 0)
        .map(|f| String::from_utf8_lossy(f).into_owned());
    while let Some(head) = fields.next() {
        let parts: Vec<&str> = head.splitn(3, '\t').collect();
        let [added, removed, path] = parts[..] else {
            continue;
        };
        let path = if path.is_empty() {
            // `-z` renames: `added\tremoved\t\0old\0new\0`.
            let _old = fields.next();
            fields.next().unwrap_or_default()
        } else {
            path.to_string()
        };
        stats.insert(path, (added.parse().ok(), removed.parse().ok()));
    }
    stats
}

/// Clones `source` bare into `dest`: the copy every later read goes through,
/// so the source is read once and never written (hooks do not run on a clone's
/// object copy and nothing is checked out).
pub fn mirror(source: &Path, dest: &Path) -> Result<Git> {
    if dest.join("HEAD").exists() {
        return Ok(Git::at(dest));
    }
    let out = Command::new("git")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_SYSTEM", "/dev/null")
        .args(["clone", "--bare", "--no-hardlinks", "--quiet"])
        .arg(source)
        .arg(dest)
        .stdin(Stdio::null())
        .output()
        .context("starting git clone")?;
    if !out.status.success() {
        bail!(
            "cloning {} failed: {}",
            source.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(Git::at(dest))
}

/// A fresh repository in `dir` holding the files already there as one commit:
/// the attempt tree. It has no other history, no refs but its branch, no
/// remote, and none of the objects of any later commit, so an agent that
/// reaches for `git log`, `git checkout <rev>` or a remote finds nothing.
pub fn single_commit_repo(dir: &Path) -> Result<()> {
    let git = Git::at(dir);
    git.bytes(&["init", "--quiet", "--initial-branch=main"])?;
    git.bytes(&["add", "-A", "-f"])?;
    git.bytes(&[
        "-c",
        "user.name=snapshot",
        "-c",
        "user.email=snapshot@localhost",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "--no-verify",
        "--allow-empty",
        "-m",
        "snapshot",
    ])?;
    Ok(())
}
