// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning pipelines whose every run is
// inspectable and stoppable from any process, for its clients. If your
// team needs expertise in durable operating state for agent systems, you
// can procure our services by sending an email to info@swedishembedded.com.

//! A run: one command's pipeline work, recorded so it can be inspected -
//! and stopped - from any process.
//!
//! ```text
//! runs/<run_id>/
//!   run.json          the record: command, arguments, stages, status,
//!                     outputs; rewritten atomically at every change
//!   cancel.request    present once a cancel was requested
//! ```
//!
//! A [`RunLog`] is the writing side: [`RunLog::start`] records the run as
//! running, [`RunLog::stage`] appends each stage's summary as it finishes,
//! and [`RunLog::finish`] records how it ended and what it produced. A
//! process that dies before finishing leaves its run recorded as running;
//! nothing here can tell a live run from a dead one.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::clock::Clock;
use crate::error::{decode, io, StoreError};
use crate::{new_id_with_prefix, write_atomic, StateRoot};

/// The file a run's record is kept in, inside its directory.
const RECORD: &str = "run.json";

/// The file whose presence asks a run to stop, inside its directory.
const CANCEL_REQUEST: &str = "cancel.request";

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
    /// The run's id, which also names its directory.
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
    dir: PathBuf,
    run: Run,
}

impl RunLog {
    /// Records a new run of `command` with `arguments`, running, stamped by
    /// `clock`.
    pub fn start(
        root: &StateRoot,
        command: &str,
        arguments: serde_json::Value,
        clock: &dyn Clock,
    ) -> Result<Self, StoreError> {
        let id = new_id_with_prefix("run");
        let dir = root.run_dir(&id);
        let runs = root.runs();
        fs::create_dir_all(&runs).map_err(io(&runs))?;
        // Exclusive: an id is never reused, even by a racing process.
        fs::create_dir(&dir).map_err(io(&dir))?;
        let now = clock.utc_now();
        let log = Self {
            dir,
            run: Run {
                id,
                command: command.to_string(),
                arguments,
                status: RunStatus::Running,
                started_at: now.clone(),
                updated_at: now,
                stages: Vec::new(),
                outputs: serde_json::Value::Null,
                error: None,
            },
        };
        log.write()?;
        Ok(log)
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
        self.run.stages.push(RunStage {
            stage: stage.to_string(),
            finished_at: now.clone(),
            summary,
        });
        self.run.updated_at = now;
        self.write()
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
        self.run.status = status;
        self.run.outputs = outputs;
        self.run.error = error;
        self.run.updated_at = clock.utc_now();
        self.write()?;
        Ok(self.run)
    }

    fn write(&self) -> Result<(), StoreError> {
        let text =
            serde_json::to_string_pretty(&self.run).map_err(|source| StoreError::Serialize {
                what: "run",
                source,
            })?;
        let path = self.dir.join(RECORD);
        write_atomic(&path, &text).map_err(io(&path))
    }
}

/// Whether a cancel was requested for the run whose directory is `dir`.
#[must_use]
pub fn cancel_requested(dir: &Path) -> bool {
    dir.join(CANCEL_REQUEST).is_file()
}

/// Asks the process running `run_id` to stop. The request is a file in the
/// run's directory, so it works from any process; the run polls for it.
/// Refused for a run that is not recorded or not in progress.
pub fn request_cancel(root: &StateRoot, run_id: &str) -> Result<PathBuf, StoreError> {
    let run = read_run(root, run_id)?;
    if run.status != RunStatus::Running {
        return Err(StoreError::RunNotInProgress {
            run: run_id.to_string(),
            status: run.status.as_str().to_string(),
        });
    }
    let path = root.run_dir(run_id).join(CANCEL_REQUEST);
    write_atomic(&path, &crate::clock::utc_now()).map_err(io(&path))?;
    Ok(path)
}

/// The record of `run_id`.
pub fn read_run(root: &StateRoot, run_id: &str) -> Result<Run, StoreError> {
    // A run id names a directory: anything that could step out of `runs/`
    // is no run.
    if run_id.is_empty() || run_id.contains(['/', '\\']) || run_id.starts_with('.') {
        return Err(StoreError::UnknownRun(run_id.to_string()));
    }
    let path = root.run_dir(run_id).join(RECORD);
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(StoreError::UnknownRun(run_id.to_string()))
        }
        Err(e) => return Err(io(&path)(e)),
    };
    decode(&path, &bytes)
}

/// Every recorded run, oldest first. A directory without a readable record
/// is not a run and is skipped.
pub fn list_runs(root: &StateRoot) -> Result<Vec<Run>, StoreError> {
    let dir = root.runs();
    let entries = match fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(io(&dir)(e)),
    };
    let mut runs = Vec::new();
    for entry in entries {
        let name = entry.map_err(io(&dir))?.file_name();
        if let Some(run) = name.to_str().and_then(|id| read_run(root, id).ok()) {
            runs.push(run);
        }
    }
    runs.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(runs)
}
