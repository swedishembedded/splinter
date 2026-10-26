// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent-driven training-task generation that
// is judged by what the tasks do when run, for its clients. If your team
// needs expertise in synthetic task generation then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The invent harness: the workspace it runs in, and what a run left in it.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sven_sdk::CancelToken;

use super::{copy_tree, io, tail, Call, FlywheelError, HarnessRunRecord, Runner, SvenLaunch};

/// What an invent run logged, by outcome.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Yield {
    /// Ideas that became tasks.
    pub admitted: usize,
    /// Ideas whose task the reviewer or the gate refused.
    pub rejected: usize,
    /// Ideas the novelty decision found to repeat a task that exists.
    pub duplicate: usize,
    /// Angles that failed for good.
    pub failed: usize,
    /// Log lines that are not an outcome of the four kinds.
    pub unreadable: usize,
}

impl Yield {
    fn count(&mut self, line: &str) {
        let outcome = serde_json::from_str::<Value>(line)
            .ok()
            .and_then(|v| v.get("outcome").and_then(Value::as_str).map(str::to_string));
        match outcome.as_deref() {
            Some("admitted") => self.admitted += 1,
            Some("rejected") => self.rejected += 1,
            Some("duplicate") => self.duplicate += 1,
            Some("failed") => self.failed += 1,
            _ => self.unreadable += 1,
        }
    }

    /// Adds `other`'s counts to these.
    pub fn add(&mut self, other: Self) {
        self.admitted += other.admitted;
        self.rejected += other.rejected;
        self.duplicate += other.duplicate;
        self.failed += other.failed;
        self.unreadable += other.unreadable;
    }
}

/// A file of the raw material one invent run works from.
#[derive(Clone, Debug)]
pub struct DataFile {
    /// The file's name under `data/`.
    pub name: String,
    /// Its text.
    pub text: String,
}

/// What one invent run is asked.
#[derive(Clone, Copy, Debug)]
pub struct InventRequest<'a> {
    /// The raw material, laid out as `data/`.
    pub data: &'a [DataFile],
    /// Who the tasks are for and the skills they should come away with,
    /// laid out as `brief.md`.
    pub brief: &'a str,
    /// The most angles, and so the most new tasks, the run works on.
    pub count: usize,
    /// How long the run may take.
    pub deadline: Duration,
}

/// What an invent run produced.
#[derive(Clone, Debug)]
pub struct Invented {
    /// The task directories of the commits the run added, exported as
    /// committed, in commit order.
    pub tasks: Vec<PathBuf>,
    /// The outcome of every idea the run logged.
    pub yielded: Yield,
    /// How the run ended.
    pub run: HarnessRunRecord,
}

/// The workspace invent runs in, kept from run to run: `tasks/` is a git
/// repository that holds every task made so far.
#[derive(Clone, Debug)]
pub struct InventWorkspace {
    root: PathBuf,
}

impl InventWorkspace {
    /// A workspace at `root`, made if it is not there.
    ///
    /// # Errors
    ///
    /// [`FlywheelError::Io`] when the directory cannot be made.
    pub fn create(root: &Path) -> Result<Self, FlywheelError> {
        let tasks = root.join("tasks");
        std::fs::create_dir_all(&tasks).map_err(io(format!("making {}", tasks.display())))?;
        Ok(Self {
            root: root.to_path_buf(),
        })
    }

    fn tasks(&self) -> PathBuf {
        self.root.join("tasks")
    }

    /// Replaces `data/`, `brief.md` and `templates/solve/` with this run's.
    fn lay_out(&self, request: &InventRequest<'_>, solve: &Path) -> Result<(), FlywheelError> {
        let data = self.root.join("data");
        let _ = std::fs::remove_dir_all(&data);
        std::fs::create_dir_all(&data).map_err(io(format!("making {}", data.display())))?;
        for file in request.data {
            let path = data.join(&file.name);
            std::fs::write(&path, &file.text).map_err(io(format!("writing {}", path.display())))?;
        }
        let brief = self.root.join("brief.md");
        std::fs::write(&brief, request.brief)
            .map_err(io(format!("writing {}", brief.display())))?;
        let template = self.root.join("templates/solve");
        let _ = std::fs::remove_dir_all(&template);
        copy_tree(solve, &template, &[])
    }

    /// The repository's head, `None` while it holds no commit.
    fn head(&self) -> Result<Option<String>, FlywheelError> {
        if !self.tasks().join(".git").exists() {
            return Ok(None);
        }
        Ok(git(&self.tasks(), &["rev-parse", "--verify", "-q", "HEAD"]).ok())
    }

    /// The number of lines `tasks/log.jsonl` holds.
    fn log_lines(&self) -> Result<usize, FlywheelError> {
        match std::fs::read_to_string(self.tasks().join("log.jsonl")) {
            Ok(text) => Ok(text.lines().count()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(0),
            Err(e) => Err(io("reading tasks/log.jsonl")(e)),
        }
    }
}

/// The standard output of `git args` in `repository`.
fn git(repository: &Path, args: &[&str]) -> Result<String, FlywheelError> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(args)
        .output()
        .map_err(io("running git"))?;
    if !output.status.success() {
        return Err(FlywheelError::Git {
            args: args.iter().map(ToString::to_string).collect(),
            repository: repository.to_path_buf(),
            detail: tail(&output.stderr),
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// The invent harness, run on workspaces.
#[derive(Clone, Debug)]
pub struct Invent {
    runner: Runner,
    harness: PathBuf,
    solve: PathBuf,
}

impl Invent {
    /// The invent harness `harness`, whose runs adapt the generic solve
    /// harness `solve` (a directory) to each task.
    #[must_use]
    pub fn new(launch: SvenLaunch, harness: PathBuf, solve: PathBuf) -> Self {
        Self {
            runner: Runner::new(launch),
            harness,
            solve,
        }
    }

    /// The directories of the runtimes the tasks run code in, first on the
    /// child's `PATH`.
    #[must_use]
    pub fn with_tool_dirs(mut self, dirs: Vec<PathBuf>) -> Self {
        self.runner.tool_dirs = dirs;
        self
    }

    /// Stops the run in progress, and every later one, once `cancel` is
    /// cancelled.
    #[must_use]
    pub fn with_cancel(mut self, cancel: CancelToken) -> Self {
        self.runner.cancel = Some(cancel);
        self
    }

    /// The model the harness's agents run on.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.runner.launch.model
    }

    /// One invent run on `workspace`.
    ///
    /// # Errors
    ///
    /// [`FlywheelError`] when the workspace cannot be laid out, sven cannot
    /// run or refuses the document, or the tasks repository cannot be read.
    /// A run that failed, was stopped or ended without a task is not an
    /// error: it yields what it committed, and says how it ended.
    pub async fn run(
        &self,
        workspace: &InventWorkspace,
        request: &InventRequest<'_>,
    ) -> Result<Invented, FlywheelError> {
        workspace.lay_out(request, &self.solve)?;
        let before = (workspace.head()?, workspace.log_lines()?);
        let label = unique_label("invent");
        let finished = self
            .runner
            .run(&Call {
                harness: &self.harness,
                workspace: &workspace.root,
                inputs: &json!({ "count": request.count }),
                deadline: request.deadline,
                label: &label,
            })
            .await?;
        let tasks = self.added_tasks(workspace, before.0.as_deref(), &label)?;
        let mut yielded = Yield::default();
        if let Ok(log) = std::fs::read_to_string(workspace.tasks().join("log.jsonl")) {
            for line in log.lines().skip(before.1) {
                yielded.count(line);
            }
        }
        Ok(Invented {
            tasks,
            yielded,
            run: finished.record,
        })
    }

    /// The task directories the commits after `before` added, exported as
    /// committed under the run's kept directory, in commit order.
    fn added_tasks(
        &self,
        workspace: &InventWorkspace,
        before: Option<&str>,
        label: &str,
    ) -> Result<Vec<PathBuf>, FlywheelError> {
        let tasks = workspace.tasks();
        if workspace.head()?.is_none() {
            return Ok(Vec::new());
        }
        let range = before.map_or_else(|| "HEAD".to_string(), |b| format!("{b}..HEAD"));
        let commits = git(&tasks, &["rev-list", "--reverse", &range])?;
        let mut slugs: Vec<String> = Vec::new();
        for commit in commits.lines() {
            let added = git(
                &tasks,
                &[
                    "diff-tree",
                    "--root",
                    "--no-commit-id",
                    "--name-only",
                    "-r",
                    commit,
                ],
            )?;
            for path in added.lines() {
                if let Some((slug, _)) = path.split_once('/') {
                    if !slugs.iter().any(|s| s == slug) {
                        slugs.push(slug.to_string());
                    }
                }
            }
        }
        let out = self.runner.launch.runs.join(label).join("tasks");
        let mut exported = Vec::new();
        for slug in slugs {
            // A directory a later commit removed is no task.
            if git(&tasks, &["cat-file", "-e", &format!("HEAD:{slug}")]).is_err() {
                continue;
            }
            export(&tasks, &slug, &out)?;
            exported.push(out.join(&slug));
        }
        Ok(exported)
    }
}

/// A name for a kept run that no other run of this process or an earlier
/// one has.
fn unique_label(what: &str) -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!("{what}-{nanos}-{}", NEXT.fetch_add(1, Ordering::Relaxed))
}

/// Exports `slug` as committed at the head of `repository` into `out`.
fn export(repository: &Path, slug: &str, out: &Path) -> Result<(), FlywheelError> {
    std::fs::create_dir_all(out).map_err(io(format!("making {}", out.display())))?;
    let archive = std::process::Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(["archive", "--format=tar", "HEAD", "--", slug])
        .output()
        .map_err(io("running git archive"))?;
    if !archive.status.success() {
        return Err(FlywheelError::Git {
            args: vec!["archive".into(), slug.into()],
            repository: repository.to_path_buf(),
            detail: tail(&archive.stderr),
        });
    }
    let mut unpack = std::process::Command::new("tar")
        .args(["-x", "-C"])
        .arg(out)
        .stdin(Stdio::piped())
        .spawn()
        .map_err(io("running tar"))?;
    {
        use std::io::Write;
        let mut stdin = unpack.stdin.take().ok_or_else(|| FlywheelError::Io {
            what: "running tar".into(),
            source: std::io::Error::other("no standard input"),
        })?;
        stdin
            .write_all(&archive.stdout)
            .map_err(io("writing to tar"))?;
    }
    let status = unpack.wait().map_err(io("waiting for tar"))?;
    if !status.success() {
        return Err(FlywheelError::Io {
            what: format!("unpacking {slug}"),
            source: std::io::Error::other(format!("tar ended with {status}")),
        });
    }
    Ok(())
}
