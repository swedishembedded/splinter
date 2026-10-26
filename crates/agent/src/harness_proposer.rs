// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent-driven training-task generation that
// is judged by what the tasks do when run, for its clients. If your team
// needs expertise in synthetic task generation then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Task proposals written by a sven harness run as a subprocess.
//!
//! [`HarnessProposer`] answers a request for a kind's tasks by running
//! `sven harness run` on a scratch tree it lays out, the way a benchmark
//! harness runs an agent: sven owns the act of running the harness, Splinter
//! owns the orchestration and checks what sven left on disk. The tree holds
//!
//! * `sections/NN.md`, one file per section in the order the model is shown,
//!   so a citation's section number is the file's number;
//! * `brief.md`, the role, a blank line, then the brief;
//! * `request.json`, the whole request input.
//!
//! The harness is given `{count, kind, runtime}` as its inputs and leaves
//! `tasks.json` at the tree's root: `{"tasks": [...]}` in the one shape
//! [`Reply`] parses. Admission is untouched; the file is only a proposal. A
//! missing file is no reply, an unparsable one is malformed, and a run that
//! was stopped but left a valid file still replied. The runtime's directory is
//! first on the child's `PATH`, so the harness's shell runs its code.
//!
//! The ATIF trajectory of each run and of each of its conversations is kept
//! under [`HarnessLaunch::runs`] and listed by [`HarnessProposer::runs`].

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use splinter_knowledge::tasks::propose::{ProposalRequest, ProposeError, Proposed, TaskProposer};
use splinter_knowledge::tasks::reply::Reply;
use splinter_knowledge::tasks::{Rejection, GENERATION_METHOD};
use splinter_sandbox::host::DEFAULT_SEARCH_PATH;
use splinter_sandbox::RuntimeEnvironment;
use sven_sdk::CancelToken;
use tokio::process::Command;

/// How long past the request's deadline sven is given to stop and write its
/// files before the process is killed.
const KILL_GRACE: Duration = Duration::from_secs(60);

/// How often a run in progress looks at its cancel token.
const CANCEL_POLL: Duration = Duration::from_millis(250);

/// The longest stretch of a failed run's stderr a message quotes.
const STDERR_TAIL_BYTES: usize = 2000;

/// Where and how the harness is run.
#[derive(Clone, Debug)]
pub struct HarnessLaunch {
    /// The `sven` executable.
    pub sven: PathBuf,
    /// The harness document or directory.
    pub harness: PathBuf,
    /// The model the harness's agents run on, as `provider/name`.
    pub model: String,
    /// The OpenAI-compatible server that serves it.
    pub endpoint: String,
    /// The directory the trajectories of every run are kept under.
    pub runs: PathBuf,
    /// How long sven waits for a server's first bytes; sven's default when
    /// `None`.
    pub first_byte_timeout: Option<Duration>,
}

/// What one harness run left that outlives it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HarnessRunRecord {
    /// The kind of tasks the run was for.
    pub kind: Option<String>,
    /// The run's ATIF trajectory.
    pub trajectory: PathBuf,
    /// The directory of the ATIF trajectory of each conversation.
    pub conversations: PathBuf,
    /// How sven reported the run ended (`success`, `timeout`, ...), when it
    /// reported.
    pub conclusion: Option<String>,
}

/// A [`TaskProposer`] that has a sven harness write the tasks.
pub struct HarnessProposer {
    launch: HarnessLaunch,
    runtimes: Vec<RuntimeEnvironment>,
    /// Answers the requests that are not for a kind's tasks.
    fallback: Arc<dyn TaskProposer>,
    cancel: Option<CancelToken>,
    recorded: Mutex<Vec<HarnessRunRecord>>,
}

impl HarnessProposer {
    /// A proposer running `launch`, whose harnesses may use `runtimes`;
    /// `fallback` answers every request that is not for a kind's tasks.
    #[must_use]
    pub fn new(
        launch: HarnessLaunch,
        runtimes: Vec<RuntimeEnvironment>,
        fallback: Arc<dyn TaskProposer>,
    ) -> Self {
        Self {
            launch,
            runtimes,
            fallback,
            cancel: None,
            recorded: Mutex::new(Vec::new()),
        }
    }

    /// Stops the runs in progress, and every later one, once `cancel` is
    /// cancelled: a stopped request has no reply.
    #[must_use]
    pub fn with_cancel(mut self, cancel: CancelToken) -> Self {
        self.cancel = Some(cancel);
        self
    }

    /// The runs made so far, in the order they began.
    #[must_use]
    pub fn runs(&self) -> Vec<HarnessRunRecord> {
        self.recorded.lock().map(|r| r.clone()).unwrap_or_default()
    }

    fn cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(CancelToken::is_cancelled)
    }

    /// The directory of the runtime `name`'s executable, which the harness's
    /// shell finds the runtime by.
    fn runtime_dir(&self, name: &str) -> Option<PathBuf> {
        let runtime = self
            .runtimes
            .iter()
            .find(|env| env.runtime().spec.name == name)?;
        Path::new(&runtime.runtime().executable)
            .parent()
            .filter(|dir| dir.is_absolute())
            .map(Path::to_path_buf)
    }

    async fn run(&self, request: &ProposalRequest<'_>) -> Result<Proposed, ProposeError> {
        let work = tempfile::Builder::new()
            .prefix("invent-")
            .tempdir()
            .map_err(|e| ProposeError(format!("cannot make the run's directory: {e}")))?;
        let tree = work.path().join("tree");
        let scratchpad = work.path().join("scratchpad");
        let id = work
            .path()
            .file_name()
            .map_or_else(|| "run".into(), |n| n.to_string_lossy().into_owned());
        let kept = self.launch.runs.join(id);
        let (trajectory, conversations) =
            (kept.join("trajectory.json"), kept.join("conversations"));
        let inputs = work.path().join("inputs.json");
        lay_out(&tree, &inputs, request)
            .map_err(|e| ProposeError(format!("cannot lay out the run: {e}")))?;
        std::fs::create_dir_all(&kept)
            .map_err(|e| ProposeError(format!("cannot make {}: {e}", kept.display())))?;

        let mut command = Command::new(&self.launch.sven);
        command
            .args(["harness", "run"])
            .arg(&self.launch.harness)
            .arg("--input-file")
            .arg(&inputs)
            .arg("--workspace")
            .arg(&tree)
            .arg("--scratchpad")
            .arg(&scratchpad)
            .args(["--model", &self.launch.model])
            .args(["--model-endpoint", &self.launch.endpoint])
            .args(["--approvals", "auto", "--json", "--quiet"])
            .arg("--trajectory")
            .arg(&trajectory)
            .arg("--conversations")
            .arg(&conversations)
            .args(["--timeout", &request.deadline.as_secs().max(1).to_string()]);
        if let Some(wait) = self.launch.first_byte_timeout {
            command.args(["--first-byte-timeout", &wait.as_secs().max(1).to_string()]);
        }
        if let Some(dir) = request.runtime.and_then(|name| self.runtime_dir(name)) {
            command.env("PATH", path_with_first(&dir));
        }
        command
            .current_dir(&tree)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let child = command
            .spawn()
            .map_err(|e| ProposeError(format!("cannot run {}: {e}", self.launch.sven.display())))?;

        let finished = tokio::select! {
            out = child.wait_with_output() => Some(out),
            () = tokio::time::sleep(request.deadline + KILL_GRACE) => None,
            () = self.until_cancelled() => None,
        };
        // A child dropped here is killed (`kill_on_drop`).
        let Some(output) = finished else {
            return Ok(declined(
                Rejection::NoReply,
                "the run was cancelled or outlived its deadline".into(),
            ));
        };
        let output = output.map_err(|e| ProposeError(format!("sven did not finish: {e}")))?;

        let report: Option<serde_json::Value> = serde_json::from_slice(&output.stdout).ok();
        let field = |name: &str| {
            report
                .as_ref()
                .and_then(|r| r.get(name))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        };
        let (status, conclusion) = (field("status"), field("conclusion"));
        // 2 is a document or input sven refused, 5 its own infrastructure: the
        // run never produced anything, so this is not a declined request.
        let code = output.status.code();
        if report.is_none() || matches!(code, Some(2 | 5) | None) {
            return Err(ProposeError(format!(
                "sven harness run failed (exit {code:?}): {}",
                tail(&output.stderr)
            )));
        }
        if let Ok(mut recorded) = self.recorded.lock() {
            recorded.push(HarnessRunRecord {
                kind: request.kind.map(str::to_string),
                trajectory,
                conversations,
                conclusion: conclusion.clone(),
            });
        }
        let how = format!(
            "status {}, conclusion {}",
            status.as_deref().unwrap_or("unknown"),
            conclusion.as_deref().unwrap_or("unknown")
        );
        let tasks = tree.join("tasks.json");
        let bytes = match std::fs::read(&tasks) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(declined(
                    Rejection::NoReply,
                    format!("the run left no tasks.json ({how})"),
                ));
            }
            Err(e) => {
                return Err(ProposeError(format!(
                    "cannot read {}: {e}",
                    tasks.display()
                )))
            }
        };
        Ok(match serde_json::from_slice::<Reply>(&bytes) {
            Ok(reply) => Proposed::Replied(reply),
            Err(e) => declined(
                Rejection::Malformed,
                format!("tasks.json is not a reply: {e} ({how})"),
            ),
        })
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

#[async_trait]
impl TaskProposer for HarnessProposer {
    fn identity(&self) -> &str {
        &self.launch.model
    }

    async fn propose(&self, request: &ProposalRequest<'_>) -> Result<Proposed, ProposeError> {
        if request.method != GENERATION_METHOD {
            return self.fallback.propose(request).await;
        }
        if self.cancelled() {
            return Ok(declined(
                Rejection::NoReply,
                "cancelled before the run began".into(),
            ));
        }
        self.run(request).await
    }
}

fn declined(rejection: Rejection, detail: String) -> Proposed {
    Proposed::Declined { rejection, detail }
}

/// `dir` followed by the sandbox's default search path: the child does not
/// inherit the caller's, so the same runtime and tools are found whatever
/// the caller's environment is.
fn path_with_first(dir: &Path) -> String {
    format!("{}:{DEFAULT_SEARCH_PATH}", dir.display())
}

/// The last bytes of a stream, as text.
fn tail(bytes: &[u8]) -> String {
    let start = bytes.len().saturating_sub(STDERR_TAIL_BYTES);
    String::from_utf8_lossy(&bytes[start..]).trim().to_string()
}

/// The tree a run works in and the file of its inputs.
fn lay_out(tree: &Path, inputs: &Path, request: &ProposalRequest<'_>) -> std::io::Result<()> {
    let sections = tree.join("sections");
    std::fs::create_dir_all(&sections)?;
    let shown: Vec<(usize, &str)> = request
        .input
        .get("sections")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|s| {
            Some((
                usize::try_from(s.get("position")?.as_u64()?).ok()?,
                s.get("text")?.as_str()?,
            ))
        })
        .collect();
    let width = shown
        .iter()
        .map(|(position, _)| position.to_string().len())
        .max()
        .unwrap_or(0)
        .max(2);
    for (position, text) in shown {
        std::fs::write(sections.join(format!("{position:0width$}.md")), text)?;
    }
    std::fs::write(
        tree.join("brief.md"),
        format!("{}\n\n{}", request.role, request.brief),
    )?;
    std::fs::write(
        tree.join("request.json"),
        serde_json::to_vec_pretty(&request.input)?,
    )?;
    let inputs_json = serde_json::json!({
        "count": request.count,
        "kind": request.kind.unwrap_or(""),
        "runtime": request.runtime.unwrap_or(""),
    });
    std::fs::write(inputs, serde_json::to_vec(&inputs_json)?)
}
