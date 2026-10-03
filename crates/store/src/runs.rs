// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements durable, content-addressed experience
// stores for learning agents, for its clients. If your team needs expertise
// in training-data lineage or crash-safe storage, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Run records: what a command was asked to do, the stages it finished and
//! how it ended, kept for every command that writes pipeline state.
//!
//! A run is a sequence of events in the experience database, never an
//! updated record: it starts, finishes stages, and ends, and what a reader
//! sees is those events folded in order. A cancel request is a signal, which
//! any process can raise and the running one polls without reading anything
//! else. A run's id is claimed with a signal too, so two processes starting in
//! the same instant never share one, and its end raises a third, so asking
//! whether a run is still in progress never reads a record.

use serde::{Deserialize, Serialize};
use serde_json::json;
use splinter_expdb::model::Entity;

use crate::error::StoreError;
use crate::new_id_with_prefix;
use crate::workspace::Workspace;
use splinter_core::clock::Clock;

const RUN_EVENT: &str = "run_event";

/// How many times a run asks for an id before giving up.
const ID_ATTEMPTS: usize = 16;

/// Where a run stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    /// Started and not yet finished (or its process died).
    Running,
    /// Finished with everything it set out to do.
    Completed,
    /// Stopped by an error, recorded in [`Run::error`].
    Failed,
    /// Stopped by a cancel request.
    Cancelled,
}

impl RunStatus {
    /// The status as it is serialized.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

/// One finished stage of a run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunStage {
    /// The stage's name.
    pub stage: String,
    /// When it finished.
    pub finished_at: String,
    /// What it did, as the command reports it.
    pub summary: serde_json::Value,
}

/// A run's record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Run {
    /// The run's id.
    pub id: String,
    /// The command that started it.
    pub command: String,
    /// What the command was asked to do.
    pub arguments: serde_json::Value,
    /// Where it stands.
    pub status: RunStatus,
    /// When it started.
    pub started_at: String,
    /// When its record last changed.
    pub updated_at: String,
    /// Its finished stages, in order.
    pub stages: Vec<RunStage>,
    /// What it produced, as the command reports it; `null` until it ends.
    pub outputs: serde_json::Value,
    /// Why it failed, when it did.
    pub error: Option<String>,
}

/// The writing side of one run. See the module documentation.
#[derive(Debug)]
pub struct RunLog {
    workspace: Workspace,
    run: Run,
    events: u64,
}

fn event(run: &str, seq: u64, kind: &str, at: &str, body: serde_json::Value) -> Entity {
    Entity::new(
        RUN_EVENT,
        json!({ "run": run, "seq": seq, "kind": kind, "at": at, "body": body }),
    )
}

impl RunLog {
    /// Records a new run of `command` with `arguments`, running, stamped by
    /// `clock`.
    pub fn start(
        workspace: &Workspace,
        command: &str,
        arguments: serde_json::Value,
        clock: &dyn Clock,
    ) -> Result<Self, StoreError> {
        // Exclusive: an id is never reused, even by a racing process.
        let mut claimed = None;
        for _ in 0..ID_ATTEMPTS {
            let id = new_id_with_prefix("run");
            if workspace.signal(&claim_signal(&id), command)? {
                claimed = Some(id);
                break;
            }
            std::thread::yield_now();
        }
        let id = claimed.ok_or_else(|| StoreError::Rejected {
            what: "run",
            reason: "no unused run id could be claimed".into(),
        })?;
        let now = clock.utc_now();
        let log = Self {
            workspace: workspace.clone(),
            run: Run {
                id: id.clone(),
                command: command.to_string(),
                arguments: arguments.clone(),
                status: RunStatus::Running,
                started_at: now.clone(),
                updated_at: now.clone(),
                stages: Vec::new(),
                outputs: serde_json::Value::Null,
                error: None,
            },
            events: 1,
        };
        log.append(&event(
            &id,
            0,
            "start",
            &now,
            json!({ "command": command, "arguments": arguments }),
        ))?;
        Ok(log)
    }

    fn append(&self, entity: &Entity) -> Result<(), StoreError> {
        self.workspace.write(|s| s.put_entity(entity).map(|_| ()))
    }

    /// The run's id.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.run.id
    }

    /// Appends the finished stage `stage` with `summary`.
    pub fn stage(
        &mut self,
        stage: &str,
        summary: serde_json::Value,
        clock: &dyn Clock,
    ) -> Result<(), StoreError> {
        let now = clock.utc_now();
        self.append(&event(
            &self.run.id,
            self.events,
            "stage",
            &now,
            json!({ "stage": stage, "summary": summary }),
        ))?;
        self.events += 1;
        self.run.stages.push(RunStage {
            stage: stage.to_string(),
            finished_at: now.clone(),
            summary,
        });
        self.run.updated_at = now;
        Ok(())
    }

    /// Records how the run ended - `status`, what it produced, and why it
    /// failed when it did - and consumes the log.
    pub fn finish(
        mut self,
        status: RunStatus,
        outputs: serde_json::Value,
        error: Option<String>,
        clock: &dyn Clock,
    ) -> Result<Run, StoreError> {
        let now = clock.utc_now();
        self.append(&event(
            &self.run.id,
            self.events,
            "finish",
            &now,
            json!({ "status": status, "outputs": outputs, "error": error }),
        ))?;
        self.workspace
            .signal(&done_signal(&self.run.id), status.as_str())?;
        self.run.status = status;
        self.run.outputs = outputs;
        self.run.error = error;
        self.run.updated_at = now;
        Ok(self.run)
    }
}

/// The signal that says run `run_id` has ended.
fn done_signal(run_id: &str) -> String {
    format!("done/{run_id}")
}

/// The signal that claims the id `run_id`.
fn claim_signal(run_id: &str) -> String {
    format!("run/{run_id}")
}

/// The signal that asks run `run_id` to stop.
fn cancel_signal(run_id: &str) -> String {
    format!("cancel/{run_id}")
}

/// Whether a cancel was requested for `run_id`. A signal that cannot be read
/// is no request: a run is never stopped by a failure to look.
#[must_use]
pub fn cancel_requested(workspace: &Workspace, run_id: &str) -> bool {
    workspace.signalled(&cancel_signal(run_id)).unwrap_or(false)
}

/// Asks the process running `run_id` to stop, by raising a signal any process
/// can see; the run polls for it. Returns the signal's name. Refused for a run
/// that is not recorded or not in progress.
pub fn request_cancel(workspace: &Workspace, run_id: &str) -> Result<String, StoreError> {
    // Three signals say all that is needed, so a cancel reads no record:
    // the run exists if its id was claimed, and it has ended if it said so.
    if !workspace.signalled(&claim_signal(run_id))? {
        return Err(StoreError::UnknownRun(run_id.to_string()));
    }
    if let Some(status) = workspace.signal_note(&done_signal(run_id))? {
        return Err(StoreError::RunNotInProgress {
            run: run_id.to_string(),
            status,
        });
    }
    let signal = cancel_signal(run_id);
    workspace.signal(&signal, &splinter_core::clock::utc_now())?;
    Ok(signal)
}

/// The events of every run, each run's in order.
fn events_by_run(
    workspace: &Workspace,
) -> Result<std::collections::BTreeMap<String, Vec<serde_json::Value>>, StoreError> {
    workspace.refresh()?;
    let stored = workspace.read_or_default(|s| s.entities(RUN_EVENT))?;
    let mut runs: std::collections::BTreeMap<String, Vec<serde_json::Value>> = Default::default();
    for found in stored {
        if let Some(run) = found.entity.value.get("run").and_then(|r| r.as_str()) {
            runs.entry(run.to_string())
                .or_default()
                .push(found.entity.value);
        }
    }
    for events in runs.values_mut() {
        events.sort_by_key(|e| e.get("seq").and_then(serde_json::Value::as_u64));
    }
    Ok(runs)
}

/// A run's record from its events; `None` when it has no start.
fn fold(id: &str, events: &[serde_json::Value]) -> Option<Run> {
    let start = events.iter().find(|e| e["kind"] == "start")?;
    let text = |v: &serde_json::Value| v.as_str().unwrap_or_default().to_string();
    let mut run = Run {
        id: id.to_string(),
        command: text(&start["body"]["command"]),
        arguments: start["body"]["arguments"].clone(),
        status: RunStatus::Running,
        started_at: text(&start["at"]),
        updated_at: text(&start["at"]),
        stages: Vec::new(),
        outputs: serde_json::Value::Null,
        error: None,
    };
    for e in events {
        match e["kind"].as_str() {
            Some("stage") => {
                run.stages.push(RunStage {
                    stage: text(&e["body"]["stage"]),
                    finished_at: text(&e["at"]),
                    summary: e["body"]["summary"].clone(),
                });
                run.updated_at = text(&e["at"]);
            }
            Some("finish") => {
                run.status = serde_json::from_value(e["body"]["status"].clone())
                    .unwrap_or(RunStatus::Failed);
                run.outputs = e["body"]["outputs"].clone();
                run.error = e["body"]["error"].as_str().map(str::to_string);
                run.updated_at = text(&e["at"]);
            }
            _ => {}
        }
    }
    Some(run)
}

/// The record of `run_id`, as of every event any process has committed.
pub fn read_run(workspace: &Workspace, run_id: &str) -> Result<Run, StoreError> {
    events_by_run(workspace)?
        .get(run_id)
        .and_then(|events| fold(run_id, events))
        .ok_or_else(|| StoreError::UnknownRun(run_id.to_string()))
}

/// Every recorded run, oldest first. Events with no start are not a run and
/// are skipped.
pub fn list_runs(workspace: &Workspace) -> Result<Vec<Run>, StoreError> {
    Ok(events_by_run(workspace)?
        .iter()
        .filter_map(|(id, events)| fold(id, events))
        .collect())
}
