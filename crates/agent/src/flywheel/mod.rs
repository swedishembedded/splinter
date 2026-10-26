// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent-driven training-task generation that
// is judged by what the tasks do when run, for its clients. If your team
// needs expertise in synthetic task generation then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The flywheel harnesses run as `sven harness run` subprocesses: sven owns
//! the act of running a harness, Splinter lays out the workspace, reads back
//! what the run left on disk and decides what it means.
//!
//! * [`Invent`] runs the invent harness on an [`InventWorkspace`]: `data/`
//!   (the raw material of one window of the sources), `brief.md`,
//!   `templates/solve/` (the generic solve harness the run adapts per task)
//!   and `tasks/`, a git repository that persists from run to run so a later
//!   run sees the tasks the earlier ones made. What a run produced is read
//!   from git: the task directories of the commits it added, exported as
//!   committed, and the outcome of every idea it logged ([`Yield`]). A task
//!   directory lying uncommitted in `tasks/` is not a task. An invent run
//!   takes tens of minutes; its deadline is the run's own, never a request's.
//!   A run that is stopped at its deadline or cancelled still yields what it
//!   committed before.
//! * [`Validate`] runs the validate harness on one finished attempt and
//!   focus and returns its typed decision. A run that gives no decision is
//!   an error, never a discard: the caller reports it apart.
//!
//! Both pass `--api-key-file` and the model as `provider/name` (a brain
//! server is addressed as `brain/<name>`), and keep the trajectory of each
//! run and of each of its conversations under [`SvenLaunch::runs`]. The
//! child's `PATH` holds the task runtime's directories, sven's own directory
//! (the harness compiles per-task harnesses with `sven harness validate`)
//! and the default search path, where `git`, `python3` and `bash`, which the
//! harness's gates need, must be.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use serde_json::Value;
use splinter_sandbox::host::DEFAULT_SEARCH_PATH;
use sven_sdk::CancelToken;
use tokio::process::Command;

mod invent;
mod validate;

pub use invent::{DataFile, Invent, InventRequest, InventWorkspace, Invented, Yield};
pub use validate::{conversations_text, Attempt, Decision, Validate, Verdict, FOCI};

/// How long past a run's deadline sven is given to stop and write its files
/// before the process is killed.
const KILL_GRACE: Duration = Duration::from_secs(60);

/// How often a run in progress looks at its cancel token.
const CANCEL_POLL: Duration = Duration::from_millis(250);

/// The longest stretch of a failed run's stderr a message quotes.
const STDERR_TAIL_BYTES: usize = 2000;

/// The tools the harnesses' gates run, which must be found on the child's
/// `PATH`.
const REQUIRED_TOOLS: [&str; 3] = ["git", "python3", "bash"];

/// sven's exit code for a document or input it refused, and for its own
/// infrastructure failing: the run never produced anything.
const SVEN_REFUSED: i32 = 2;
const SVEN_INFRASTRUCTURE: i32 = 5;

/// Why a harness run could not be made or read.
#[derive(Debug, thiserror::Error)]
pub enum FlywheelError {
    /// A file or directory could not be made, read or written.
    #[error("{what}: {source}")]
    Io {
        /// What was being done.
        what: String,
        /// The underlying error.
        source: std::io::Error,
    },
    /// A tool the harness's gates need is not on the child's `PATH`.
    #[error("{0} is not on the PATH; the flywheel harnesses' gates need it")]
    MissingTool(&'static str),
    /// sven could not be started.
    #[error("cannot run {path}: {source}")]
    Spawn {
        /// The executable.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// sven refused the document or its inputs, or failed itself.
    #[error("sven harness run failed (exit {code:?}): {stderr}")]
    Failed {
        /// The exit code, `None` when a signal ended it.
        code: Option<i32>,
        /// The end of its standard error.
        stderr: String,
    },
    /// A git command failed in the tasks repository.
    #[error("git {args:?} failed in {repository}: {detail}")]
    Git {
        /// The arguments.
        args: Vec<String>,
        /// The repository.
        repository: PathBuf,
        /// What went wrong.
        detail: String,
    },
    /// A validator run ended without a decision.
    #[error("the validator gave no decision ({0})")]
    NoDecision(String),
}

fn io(what: impl Into<String>) -> impl FnOnce(std::io::Error) -> FlywheelError {
    let what = what.into();
    move |source| FlywheelError::Io { what, source }
}

/// Where and how the harnesses are run.
#[derive(Clone, Debug)]
pub struct SvenLaunch {
    /// The `sven` executable.
    pub sven: PathBuf,
    /// The model the harnesses' agents run on, as `provider/name`.
    pub model: String,
    /// The OpenAI-compatible server that serves it.
    pub endpoint: String,
    /// The file holding the server's API key, when it needs one.
    pub key_file: Option<PathBuf>,
    /// How long sven waits for a server's first bytes; sven's default when
    /// `None`.
    pub first_byte_timeout: Option<Duration>,
    /// The directory the trajectories of every run are kept under.
    pub runs: PathBuf,
}

/// What one harness run left that outlives it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HarnessRunRecord {
    /// The run's ATIF trajectory.
    pub trajectory: PathBuf,
    /// The directory of the ATIF trajectory of each conversation.
    pub conversations: PathBuf,
    /// The status sven reported (`completed`, `failed`, ...).
    pub status: Option<String>,
    /// The conclusion sven reported (`success`, `timeout`, ...).
    pub conclusion: Option<String>,
    /// Why the process was killed instead of ending, when it was.
    pub killed: Option<String>,
}

/// One `sven harness run`, before it starts.
struct Call<'a> {
    harness: &'a Path,
    workspace: &'a Path,
    inputs: &'a Value,
    deadline: Duration,
    /// The kept trajectories' directory name under [`SvenLaunch::runs`].
    label: &'a str,
}

/// A `sven harness run` that has ended.
struct Finished {
    record: HarnessRunRecord,
    /// The JSON document sven printed.
    report: Option<Value>,
}

/// A launch, the directories its child finds its tools in, and the token
/// that stops it.
#[derive(Clone, Debug)]
struct Runner {
    launch: SvenLaunch,
    tool_dirs: Vec<PathBuf>,
    cancel: Option<CancelToken>,
}

impl Runner {
    fn new(launch: SvenLaunch) -> Self {
        Self {
            launch,
            tool_dirs: Vec::new(),
            cancel: None,
        }
    }

    /// The child's `PATH`: the runtime's directories, sven's own, then the
    /// default search path, which must hold the tools the gates need. The
    /// child does not inherit the caller's, so the same tools are found
    /// whatever the caller's environment is.
    fn path(&self) -> Result<String, FlywheelError> {
        let mut dirs: Vec<PathBuf> = self.tool_dirs.clone();
        dirs.extend(self.launch.sven.parent().map(Path::to_path_buf));
        dirs.extend(std::env::split_paths(DEFAULT_SEARCH_PATH));
        let mut seen: Vec<PathBuf> = Vec::new();
        for dir in dirs {
            if dir.is_absolute() && !seen.contains(&dir) {
                seen.push(dir);
            }
        }
        for tool in REQUIRED_TOOLS {
            if !seen.iter().any(|dir| dir.join(tool).is_file()) {
                return Err(FlywheelError::MissingTool(tool));
            }
        }
        let path: Vec<String> = seen.iter().map(|d| d.display().to_string()).collect();
        Ok(path.join(":"))
    }

    async fn run(&self, call: &Call<'_>) -> Result<Finished, FlywheelError> {
        let path = self.path()?;
        let kept = self.launch.runs.join(call.label);
        let (trajectory, conversations) =
            (kept.join("trajectory.json"), kept.join("conversations"));
        let scratchpad = kept.join("scratchpad");
        let inputs = kept.join("inputs.json");
        std::fs::create_dir_all(&kept).map_err(io(format!("making {}", kept.display())))?;
        std::fs::write(&inputs, call.inputs.to_string())
            .map_err(io(format!("writing {}", inputs.display())))?;

        let mut command = Command::new(&self.launch.sven);
        command
            .args(["harness", "run"])
            .arg(call.harness)
            .arg("--input-file")
            .arg(&inputs)
            .arg("--workspace")
            .arg(call.workspace)
            .arg("--scratchpad")
            .arg(&scratchpad)
            .args(["--model", &self.launch.model])
            .args(["--model-endpoint", &self.launch.endpoint])
            .args(["--approvals", "auto", "--json", "--quiet"])
            .arg("--trajectory")
            .arg(&trajectory)
            .arg("--conversations")
            .arg(&conversations)
            .args(["--timeout", &call.deadline.as_secs().max(1).to_string()]);
        if let Some(key) = &self.launch.key_file {
            command.arg("--api-key-file").arg(key);
        }
        if let Some(wait) = self.launch.first_byte_timeout {
            command.args(["--first-byte-timeout", &wait.as_secs().max(1).to_string()]);
        }
        command
            .env("PATH", path)
            .current_dir(call.workspace)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let child = command.spawn().map_err(|source| FlywheelError::Spawn {
            path: self.launch.sven.clone(),
            source,
        })?;

        let finished = tokio::select! {
            out = child.wait_with_output() => Some(out),
            () = tokio::time::sleep(call.deadline + KILL_GRACE) => None,
            () = self.until_cancelled() => None,
        };
        let mut record = HarnessRunRecord {
            trajectory,
            conversations,
            status: None,
            conclusion: None,
            killed: None,
        };
        // A child dropped here is killed (`kill_on_drop`).
        let Some(output) = finished else {
            record.killed = Some(
                if self.cancel.as_ref().is_some_and(CancelToken::is_cancelled) {
                    "cancelled"
                } else {
                    "outlived its deadline"
                }
                .to_string(),
            );
            return Ok(Finished {
                record,
                report: None,
            });
        };
        let output = output.map_err(io("sven did not finish"))?;
        let report: Option<Value> = serde_json::from_slice(&output.stdout).ok();
        let code = output.status.code();
        if report.is_none() || matches!(code, Some(SVEN_REFUSED | SVEN_INFRASTRUCTURE) | None) {
            return Err(FlywheelError::Failed {
                code,
                stderr: tail(&output.stderr),
            });
        }
        let field = |name: &str| {
            report
                .as_ref()
                .and_then(|r| r.get(name))
                .and_then(Value::as_str)
                .map(str::to_string)
        };
        record.status = field("status");
        record.conclusion = field("conclusion");
        Ok(Finished { record, report })
    }

    async fn until_cancelled(&self) {
        let Some(cancel) = &self.cancel else {
            return std::future::pending().await;
        };
        while !cancel.is_cancelled() {
            tokio::time::sleep(CANCEL_POLL).await;
        }
    }
}

/// The last bytes of a stream, as text.
fn tail(bytes: &[u8]) -> String {
    let start = bytes.len().saturating_sub(STDERR_TAIL_BYTES);
    String::from_utf8_lossy(&bytes[start..]).trim().to_string()
}

/// Copies the directory `from` to `to`, recursively, leaving out the entries
/// named `skip` at any depth.
fn copy_tree(from: &Path, to: &Path, skip: &[&str]) -> Result<(), FlywheelError> {
    std::fs::create_dir_all(to).map_err(io(format!("making {}", to.display())))?;
    for entry in std::fs::read_dir(from).map_err(io(format!("reading {}", from.display())))? {
        let entry = entry.map_err(io(format!("reading {}", from.display())))?;
        if skip.iter().any(|name| entry.file_name() == *name) {
            continue;
        }
        let target = to.join(entry.file_name());
        if entry
            .file_type()
            .map_err(io(format!("reading {}", entry.path().display())))?
            .is_dir()
        {
            copy_tree(&entry.path(), &target, skip)?;
        } else {
            std::fs::copy(entry.path(), &target)
                .map_err(io(format!("copying {}", entry.path().display())))?;
        }
    }
    Ok(())
}
