// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements delegated-task coding agents whose runs,
// traces and learned state survive process restart. If your team needs
// expertise in agent infrastructure or durable operating state, you can
// procure our services by sending an email to info@swedishembedded.com.

//! A run's durable record: its manifest, its limits, and the cross-process
//! cancel request. Everything lives in the run's directory under the state
//! root, so `show`, `resume` and `cancel` work from any process.
//!
//! ```text
//! runs/<run_id>/
//!   run.json               manifest + latest status (atomic)
//!   events.jsonl           the append-only trace
//!   transcript.json        what the agent exchanged with the model
//!   checkpoint/state.json  a suspended AgentState (serde)
//!   captured-requests.json model input captured at the wire, when recorded
//!   artifacts/             large tool outputs, referenced by trace events
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use crate::{write_atomic, StateRoot};

/// A run's manifest: what was configured, and where the attempt stands.
/// Written atomically at every transition, so `show` never depends on the
/// process that wrote it still being alive.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RunManifest {
    /// Schema version, so an old state directory stays readable.
    pub schema: u32,
    pub run_id: String,
    pub workspace: String,
    pub task: String,
    /// `pending` while the attempt runs, then the attempt's final status.
    pub status: String,
    /// A run may be resumed into further attempts; this is the latest.
    pub attempts: u32,
    pub started_ts: String,
    pub updated_ts: String,
    pub model: String,
    pub base_url: Option<String>,
    /// The local adapter this run serves from, when one was configured -
    /// the path, not the display name, so a resume can restore the exact
    /// configuration instead of a base-model run wearing the same label.
    pub local_adapter: Option<std::path::PathBuf>,
    /// The configured limits, as configured - recorded so "why did it stop"
    /// has an answer that does not require guessing what was set.
    pub limits: Limits,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Limits {
    pub timeout_secs: u64,
    pub max_tool_rounds: Option<u32>,
    /// Attempts the run may make in total, the first included; each
    /// `resume` is one more. `None` is unlimited.
    pub max_attempts: Option<u32>,
    #[serde(flatten)]
    pub budget: Budget,
}

/// The configured usage limits of one attempt. `None` is "no limit".
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Budget {
    /// Model output tokens, summed over the attempt's usage reports.
    pub max_output_tokens: Option<u64>,
    /// Provider-billed USD. Only meaningful for a remote model; local
    /// inference is not billed per token.
    pub max_cost_usd: Option<f64>,
}

/// Refuses a further attempt once the run has made as many as its limit
/// allows. Retries spend the same recorded task budget as the first try.
pub fn ensure_attempt_allowed(manifest: &RunManifest) -> anyhow::Result<()> {
    if let Some(max) = manifest.limits.max_attempts {
        anyhow::ensure!(
            manifest.attempts < max,
            "run {} has made {} of {max} allowed attempt(s); the retry budget is spent",
            manifest.run_id,
            manifest.attempts
        );
    }
    Ok(())
}

const CANCEL_REQUEST: &str = "cancel.request";

/// Asks the process running `run_id` to stop its attempt. The request is a
/// file in the run's directory, so it works from any process; the runner
/// polls for it and ends the attempt as `cancelled`. Refused for a run that
/// is not in progress, which has nothing to stop.
pub fn request_cancel(root: &StateRoot, run_id: &str) -> anyhow::Result<PathBuf> {
    let manifest = read_manifest(root, run_id)?;
    anyhow::ensure!(
        manifest.status == "pending",
        "run {run_id} is not in progress (status {:?}); nothing to cancel",
        manifest.status
    );
    let path = root.run_dir(run_id).join(CANCEL_REQUEST);
    write_atomic(&path, &crate::clock::utc_now())?;
    Ok(path)
}

/// Whether a cancel has been requested for the run in `dir`.
#[must_use]
pub fn cancel_requested(dir: &Path) -> bool {
    dir.join(CANCEL_REQUEST).is_file()
}

/// Consumes a cancel request, so the attempt that starts next is not
/// stopped by one aimed at its predecessor.
pub fn clear_cancel(dir: &Path) -> std::io::Result<()> {
    match fs::remove_file(dir.join(CANCEL_REQUEST)) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    }
}

/// Reads a run's manifest, refusing to invent one.
pub fn read_manifest(root: &StateRoot, run_id: &str) -> anyhow::Result<RunManifest> {
    let path = root.run_dir(run_id).join("run.json");
    let text = fs::read_to_string(&path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
}

/// Every run directory, oldest first, by manifest status.
pub fn list_runs(root: &StateRoot) -> anyhow::Result<Vec<RunManifest>> {
    let runs = root.runs();
    if !runs.is_dir() {
        return Ok(Vec::new());
    }
    let mut found = Vec::new();
    for entry in fs::read_dir(&runs)? {
        let path = entry?.path().join("run.json");
        if let Ok(text) = fs::read_to_string(&path) {
            if let Ok(manifest) = serde_json::from_str::<RunManifest>(&text) {
                found.push(manifest);
            }
        }
    }
    found.sort_by(|a, b| a.run_id.cmp(&b.run_id));
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> StateRoot {
        StateRoot::new(std::env::temp_dir().join(format!("splinter-{name}-{}", std::process::id())))
    }

    /// A cancel is requested from another process through the run's
    /// directory, only for a run still in progress, and an attempt that
    /// starts afterwards is not cancelled by a stale request.
    #[test]
    fn a_cancel_request_reaches_only_the_run_in_progress() {
        let root = scratch("cancel");
        let manifest = RunManifest {
            run_id: "loop-test-cancel".into(),
            status: "pending".into(),
            ..Default::default()
        };
        let dir = root.run_dir(&manifest.run_id);
        write_atomic(
            &dir.join("run.json"),
            &serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();

        assert!(!cancel_requested(&dir));
        request_cancel(&root, "loop-test-cancel").unwrap();
        assert!(cancel_requested(&dir));
        clear_cancel(&dir).unwrap();
        assert!(!cancel_requested(&dir), "a new attempt starts uncancelled");

        let finished = RunManifest {
            status: "completed".into(),
            ..manifest
        };
        write_atomic(
            &dir.join("run.json"),
            &serde_json::to_string(&finished).unwrap(),
        )
        .unwrap();
        assert!(
            request_cancel(&root, "loop-test-cancel").is_err(),
            "a finished run has nothing to cancel"
        );
        assert!(request_cancel(&root, "loop-test-missing").is_err());
        let _ = fs::remove_dir_all(root.path());
    }

    /// A resume is a retry, and retries are budgeted: past the recorded
    /// attempt limit it is refused instead of started.
    #[test]
    fn a_resume_past_the_attempt_limit_is_refused() {
        let mut manifest = RunManifest {
            attempts: 2,
            limits: Limits {
                max_attempts: Some(3),
                ..Limits::default()
            },
            ..Default::default()
        };
        assert!(ensure_attempt_allowed(&manifest).is_ok());
        manifest.attempts = 3;
        let refused = ensure_attempt_allowed(&manifest).unwrap_err().to_string();
        assert!(refused.contains("3 of 3"), "{refused}");
        manifest.limits.max_attempts = None;
        assert!(ensure_attempt_allowed(&manifest).is_ok(), "unlimited");
    }
}
