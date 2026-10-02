// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Every command that does pipeline work runs as a recorded run: its
//! arguments, each stage as it finishes, how it ended and what it produced
//! (see `splinter_record::runs`). A cancel requested from another process
//! reaches the run through a token its stages poll and hand to every model
//! run they start.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use splinter_record::runs::{self, list_runs, read_run, Run, RunLog, RunStatus};
use splinter_record::workspace::Workspace;
use sven_sdk::CancelToken;

use crate::context::Context;
use crate::error::CampaignError;

/// How often a running command looks for a cancel request.
const CANCEL_POLL: Duration = Duration::from_millis(200);

/// Recent runs `status` shows.
pub const RECENT_RUNS: usize = 5;

/// A command's report, with the run that produced it.
#[derive(Clone, Debug, Serialize)]
pub struct Recorded<R> {
    /// The run's id (`runs show <run>`).
    pub run: String,
    /// What the command reports.
    #[serde(flatten)]
    pub report: R,
}

/// The handle a recorded command's work gets: its stages and its cancel
/// token.
pub struct Recorder<'a> {
    ctx: &'a Context,
    log: RunLog,
    cancel: CancelToken,
}

impl Recorder<'_> {
    /// Records the finished stage `stage` with `summary`, and reports it to
    /// the context's progress receiver.
    pub fn stage(&mut self, stage: &str, summary: &impl Serialize) -> Result<(), CampaignError> {
        let summary = to_json(stage, summary)?;
        self.log.stage(stage, summary.clone(), self.ctx.clock())?;
        self.ctx.report_stage(stage, &summary);
        Ok(())
    }

    /// The token a cancel request fires; hand it to every model run.
    #[must_use]
    pub fn cancel_token(&self) -> CancelToken {
        self.cancel.clone()
    }

    /// Stops the work with [`CampaignError::Cancelled`] once a cancel was
    /// requested.
    pub fn check_cancelled(&self) -> Result<(), CampaignError> {
        if self.cancel.is_cancelled() {
            Err(CampaignError::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// Runs `work` as a recorded run of `command` with `arguments`, and records
/// how it ended: completed with its report as the outputs, cancelled, or
/// failed with the error.
pub fn record<R: Serialize>(
    ctx: &Context,
    command: &str,
    arguments: &impl Serialize,
    work: impl FnOnce(&mut Recorder<'_>) -> Result<R, CampaignError>,
) -> Result<Recorded<R>, CampaignError> {
    let arguments = to_json("arguments", arguments)?;
    let log = RunLog::start(ctx.workspace(), command, arguments, ctx.clock())?;
    let run = log.id().to_string();
    let cancel = CancelToken::new();
    let done = Arc::new(AtomicBool::new(false));
    let watcher = watch_for_cancel(
        ctx.workspace().clone(),
        run.clone(),
        cancel.clone(),
        Arc::clone(&done),
    );
    let mut recorder = Recorder {
        ctx,
        log,
        cancel: cancel.clone(),
    };
    let result = work(&mut recorder);
    done.store(true, Ordering::Relaxed);
    // The watcher only polls and sets a flag; a panic in it leaves nothing
    // to clean up.
    let _ = watcher.join();
    let Recorder { log, .. } = recorder;
    let clock = ctx.clock();
    match result {
        Ok(report) => {
            let outputs = to_json("report", &report)?;
            log.finish(RunStatus::Completed, outputs, None, clock)?;
            Ok(Recorded { run, report })
        }
        Err(e) => {
            let status = if cancel.is_cancelled() || matches!(e, CampaignError::Cancelled) {
                RunStatus::Cancelled
            } else {
                RunStatus::Failed
            };
            // The work's error is the one worth reporting; a failure to
            // record it would only hide it.
            let _ = log.finish(status, serde_json::Value::Null, Some(e.to_string()), clock);
            Err(e)
        }
    }
}

/// A thread that fires `cancel` once a cancel request for `run` appears,
/// until `done`.
fn watch_for_cancel(
    workspace: Workspace,
    run: String,
    cancel: CancelToken,
    done: Arc<AtomicBool>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        while !done.load(Ordering::Relaxed) {
            if runs::cancel_requested(&workspace, &run) {
                cancel.cancel();
                return;
            }
            std::thread::sleep(CANCEL_POLL);
        }
    })
}

/// `value` as JSON; `what` names it in the error.
pub(crate) fn to_json(
    what: &str,
    value: &impl Serialize,
) -> Result<serde_json::Value, CampaignError> {
    serde_json::to_value(value).map_err(|source| CampaignError::Json {
        what: what.to_string(),
        source,
    })
}

/// One run, as `runs list` shows it.
#[derive(Clone, Debug, Serialize)]
pub struct RunSummary {
    /// The run's id.
    pub id: String,
    /// The command it ran.
    pub command: String,
    /// `running`, `completed`, `failed` or `cancelled`. A run whose process
    /// died stays `running`.
    pub status: RunStatus,
    /// When it started.
    pub started_at: String,
    /// When its record last changed.
    pub updated_at: String,
}

impl From<&Run> for RunSummary {
    fn from(run: &Run) -> Self {
        Self {
            id: run.id.clone(),
            command: run.command.clone(),
            status: run.status,
            started_at: run.started_at.clone(),
            updated_at: run.updated_at.clone(),
        }
    }
}

/// What `runs list` reports.
#[derive(Clone, Debug, Serialize)]
pub struct RunList {
    /// Every recorded run, oldest first.
    pub runs: Vec<RunSummary>,
}

/// Every recorded run.
pub fn list(ctx: &Context) -> Result<RunList, CampaignError> {
    Ok(RunList {
        runs: list_runs(ctx.workspace())?
            .iter()
            .map(RunSummary::from)
            .collect(),
    })
}

/// The run `id`'s whole record.
pub fn show(ctx: &Context, id: &str) -> Result<Run, CampaignError> {
    Ok(read_run(ctx.workspace(), id)?)
}

/// What `runs cancel` reports.
#[derive(Clone, Debug, Serialize)]
pub struct CancelRequested {
    /// The run asked to stop.
    pub run: String,
    /// The signal its process polls for.
    pub request: String,
}

/// Asks the run `id` to stop; refused for a run not in progress.
pub fn cancel(ctx: &Context, id: &str) -> Result<CancelRequested, CampaignError> {
    let request = runs::request_cancel(ctx.workspace(), id)?;
    Ok(CancelRequested {
        run: id.to_string(),
        request,
    })
}
