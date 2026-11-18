// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// turning agent failures into training data you can procure our services by
// sending an email to info@swedishembedded.com.

//! The examiner's sandbox session a repair runs in (`bench task session`):
//! the practice task's trial sandbox with its pristine workspace and its
//! service, commands run in it as a trial's agent runs them, and the grade of
//! the state they leave by the task's own verifier, outside any agent.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Stdio};

use serde::{Deserialize, Serialize};

use crate::bench::practice::BenchCheckout;

/// The grade record the examiner prints.
pub const GRADE_SCHEMA: &str = "bench.grade/1";

/// How the task's verifier graded a workspace.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Grade {
    /// Always [`GRADE_SCHEMA`].
    pub schema: String,
    /// The task, as its `task.toml` names it.
    pub task: String,
    /// The staged task's digest, as the practice record names it.
    pub digest: String,
    /// `pass`, `fail` or `environment_error`.
    pub verdict: String,
    /// The verifier's reward in `[0, 1]`.
    #[serde(default)]
    pub reward: Option<f64>,
    /// The verifier's metrics.
    #[serde(default)]
    pub metrics: serde_json::Value,
    /// The verifier's reason.
    #[serde(default)]
    pub reason: Option<String>,
}

impl Grade {
    /// Whether the verifier passed the work.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.verdict == "pass"
    }
}

/// The session announcement the examiner prints first.
pub const SESSION_SCHEMA: &str = "bench.session/1";

/// One practice task's trial sandbox, held open by the examiner
/// (`bench task session`): its workspace is a host directory bound where a
/// trial's agent works, commands run in it as a trial's agent runs them
/// (`bench task session-exec`), the task's service runs beside it, and a
/// grade is the task's own verifier over the state the commands left. The
/// session ends when this value is dropped: its input closes and the
/// examiner stops the sandbox and the service.
pub struct Session {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    /// The socket commands are run through.
    pub socket: PathBuf,
}

#[derive(Deserialize)]
struct Announcement {
    schema: String,
    socket: PathBuf,
}

impl Session {
    /// Opens a session on the task `task` of the practice material at
    /// `practice`, laying out its pristine workspace at `workspace` (new or
    /// empty); the examiner's own temporary files (its socket) go under
    /// `tmp`, which must be a short path.
    ///
    /// # Errors
    /// The examiner's refusal or failure, with its words.
    pub fn open(
        checkout: &BenchCheckout,
        practice: &Path,
        task: &str,
        workspace: &Path,
        tmp: &Path,
    ) -> Result<Self, String> {
        std::fs::create_dir_all(tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
        // The sandbox's own directories (the parent of its workspace among
        // them) are made under `tmp`, and a listing the student recorded
        // shows their modes: a trial's have no set-group-id bit, so neither
        // may what this directory hands down.
        without_setgid(tmp)?;
        let mut child = checkout
            .bench()
            .args(["task", "session"])
            .arg(practice)
            .args(["--task", task, "--workspace"])
            .arg(workspace)
            .arg("--json")
            .env("TMPDIR", tmp)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("starting the examiner's session: {e}"))?;
        let stdin = child.stdin.take();
        let mut stdout = BufReader::new(child.stdout.take().ok_or("the session has no output")?);
        let mut line = String::new();
        let read = stdout
            .read_line(&mut line)
            .map_err(|e| format!("reading the session's announcement: {e}"))?;
        if read == 0 {
            let mut stderr = String::new();
            if let Some(mut err) = child.stderr.take() {
                let _ = err.read_to_string(&mut stderr);
            }
            let _ = child.wait();
            return Err(format!(
                "the examiner's session did not start: {}",
                last_lines(&stderr)
            ));
        }
        let hello: Announcement = serde_json::from_str(&line)
            .map_err(|e| format!("the session's announcement is not one this reader knows: {e}"))?;
        if hello.schema != SESSION_SCHEMA {
            return Err(format!(
                "the session is of schema {:?}, not {SESSION_SCHEMA:?}",
                hello.schema
            ));
        }
        Ok(Self {
            child,
            stdin,
            stdout,
            socket: hello.socket,
        })
    }

    /// The program and arguments that run a command in the session's
    /// sandbox: the command follows them.
    #[must_use]
    pub fn launcher(&self, checkout: &BenchCheckout) -> Vec<String> {
        vec![
            checkout.dir.join("bench").to_string_lossy().into_owned(),
            "task".into(),
            "session-exec".into(),
            self.socket.to_string_lossy().into_owned(),
            "--".into(),
        ]
    }

    /// Grades the session's current state with the task's own verifier, the
    /// trajectory at `trajectory` in the agent's logs.
    ///
    /// # Errors
    /// The examiner's refusal, or a verifier that wrote no reward.
    pub fn grade(&mut self, trajectory: &Path) -> Result<Grade, String> {
        let stdin = self.stdin.as_mut().ok_or("the session is closed")?;
        let request = serde_json::json!({ "grade": { "trajectory": trajectory } });
        writeln!(stdin, "{request}").map_err(|e| format!("asking the session to grade: {e}"))?;
        stdin
            .flush()
            .map_err(|e| format!("asking the session to grade: {e}"))?;
        let mut line = String::new();
        self.stdout
            .read_line(&mut line)
            .map_err(|e| format!("reading the session's grade: {e}"))?;
        let value: serde_json::Value = serde_json::from_str(&line)
            .map_err(|e| format!("the session's grade is not JSON ({e}): {line}"))?;
        if let Some(error) = value.get("error").and_then(serde_json::Value::as_str) {
            return Err(format!("the examiner could not grade: {error}"));
        }
        let grade: Grade = serde_json::from_value(value)
            .map_err(|e| format!("the examiner's grade is not one this reader knows: {e}"))?;
        if grade.schema != GRADE_SCHEMA {
            return Err(format!(
                "the examiner's grade is of schema {:?}, not {GRADE_SCHEMA:?}",
                grade.schema
            ));
        }
        Ok(grade)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // Closing its input stops the session; waiting reaps it, so nothing
        // the examiner started outlives the repair.
        drop(self.stdin.take());
        let _ = self.child.wait();
    }
}

/// Clears the set-group-id bit of `dir`, so the directories made in it do
/// not inherit it.
fn without_setgid(dir: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    const SETGID: u32 = 0o2000;
    let mode = std::fs::metadata(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .permissions()
        .mode();
    if mode & SETGID != 0 {
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(mode & !SETGID))
            .map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    Ok(())
}

fn last_lines(text: &str) -> String {
    let lines: Vec<&str> = text.lines().rev().take(4).collect();
    lines.into_iter().rev().collect::<Vec<_>>().join(" | ")
}
