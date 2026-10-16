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

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::json;
use splinter_expdb::model::Entity;

use crate::error::StoreError;
use crate::new_id_with_prefix;
use crate::workspace::Workspace;
use splinter_core::build::BuildInfo;
use splinter_core::clock::Clock;
use splinter_core::digest::Digest;
use splinter_core::rng::Seed;

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
    /// How long it ran, in milliseconds; absent from a run recorded before
    /// stages were timed, and from a stage that was not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    /// Why it failed, when it did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// How many of the sampling seeds a run record names itself; the rest are
/// counted and covered by the digest.
pub const SAMPLED_FIRST: usize = 16;

/// The seeds a local model drew for the sampled requests of a run, as the run
/// record shows them: bounded, however many requests the run made. The
/// complete list, in request order, is kept in the run's events
/// ([`read_sampled_seeds`]); `digest` is over all of it, so a replay can
/// check it was given the whole list.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SampledSeeds {
    /// How many requests drew a seed.
    pub count: u64,
    /// The seeds of the first [`SAMPLED_FIRST`] of them.
    pub first: Vec<u64>,
    /// The digest of every seed, in order, one per line.
    pub digest: Option<Digest>,
}

impl SampledSeeds {
    /// Whether the run drew no sampling seed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The record of `seeds`, in request order.
    #[must_use]
    pub fn of(seeds: &[u64]) -> Self {
        let lines: String = seeds.iter().map(|seed| format!("{seed}\n")).collect();
        Self {
            count: seeds.len() as u64,
            first: seeds.iter().take(SAMPLED_FIRST).copied().collect(),
            digest: (!seeds.is_empty()).then(|| Digest::sha256_of(lines.as_bytes())),
        }
    }
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
    /// The seed of each random draw the run made, by the draw's name; empty
    /// for a run recorded before seeds were, or one that drew nothing.
    #[serde(default)]
    pub seeds: BTreeMap<String, u64>,
    /// The seeds a local model drew for the run's sampled requests; empty for
    /// a run that sampled none (greedy decoding draws nothing) and for one
    /// recorded before they were kept.
    #[serde(default, skip_serializing_if = "SampledSeeds::is_empty")]
    pub sampled: SampledSeeds,
    /// What the run was made by: commits, tree state, binary and GPU
    /// contexts. Absent from a run recorded before they were kept.
    #[serde(flatten)]
    pub build: BuildInfo,
}

/// The writing side of one run. See the module documentation.
#[derive(Debug)]
pub struct RunLog {
    workspace: Workspace,
    run: Run,
    events: u64,
    /// Every sampling seed recorded so far, in order: the run record shows
    /// a bounded view of them.
    sampled: Vec<u64>,
}

/// The body of a run's start event. The build is named only when something is
/// known of it, so a start without one is byte-for-byte what it was before
/// builds were recorded, and so is its address.
fn start_body(
    command: &str,
    arguments: &serde_json::Value,
    build: &BuildInfo,
) -> serde_json::Value {
    let mut body = json!({ "command": command, "arguments": arguments });
    if !build.is_empty() {
        body["build"] = json!(build);
    }
    body
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
        Self::start_built(workspace, command, arguments, &BuildInfo::default(), clock)
    }

    /// [`Self::start`], recording what the run is `build`t by.
    pub fn start_built(
        workspace: &Workspace,
        command: &str,
        arguments: serde_json::Value,
        build: &BuildInfo,
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
                seeds: BTreeMap::new(),
                sampled: SampledSeeds::default(),
                build: build.clone(),
            },
            events: 1,
            sampled: Vec::new(),
        };
        log.append(&event(
            &id,
            0,
            "start",
            &now,
            start_body(command, &arguments, build),
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

    /// Appends the finished stage `stage` with `summary`, how long it took
    /// when that was measured, and why it failed when it did.
    pub fn stage(
        &mut self,
        stage: &str,
        summary: serde_json::Value,
        duration_ms: Option<u64>,
        error: Option<String>,
        clock: &dyn Clock,
    ) -> Result<(), StoreError> {
        let now = clock.utc_now();
        self.append(&event(
            &self.run.id,
            self.events,
            "stage",
            &now,
            json!({
                "stage": stage,
                "summary": summary,
                "duration_ms": duration_ms,
                "error": error,
            }),
        ))?;
        self.events += 1;
        self.run.stages.push(RunStage {
            stage: stage.to_string(),
            finished_at: now.clone(),
            summary,
            duration_ms,
            error,
        });
        self.run.updated_at = now;
        Ok(())
    }

    /// Records that the draw `name` used `seed`, and whether the run drew it
    /// (`drawn`) or was given it. A name drawn again replaces its seed.
    pub fn seed(
        &mut self,
        name: &str,
        seed: Seed,
        drawn: bool,
        clock: &dyn Clock,
    ) -> Result<(), StoreError> {
        let now = clock.utc_now();
        self.append(&event(
            &self.run.id,
            self.events,
            "seed",
            &now,
            json!({ "name": name, "seed": seed, "drawn": drawn }),
        ))?;
        self.events += 1;
        self.run.seeds.insert(name.to_string(), seed.value());
        self.run.updated_at = now;
        Ok(())
    }

    /// Records the seeds a local model drew for the sampled requests made
    /// since the last call, in request order, as one event: a request is not
    /// an event of its own, so a run of thousands costs one write per call.
    pub fn sampled(&mut self, seeds: &[Seed], clock: &dyn Clock) -> Result<(), StoreError> {
        if seeds.is_empty() {
            return Ok(());
        }
        let now = clock.utc_now();
        self.append(&event(
            &self.run.id,
            self.events,
            "sampled",
            &now,
            json!({ "seeds": seeds }),
        ))?;
        self.events += 1;
        self.sampled.extend(seeds.iter().map(|seed| seed.value()));
        self.run.sampled = SampledSeeds::of(&self.sampled);
        self.run.updated_at = now;
        Ok(())
    }

    /// The bounded view of the sampling seeds recorded so far.
    #[must_use]
    pub fn sampled_so_far(&self) -> &SampledSeeds {
        &self.run.sampled
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
        seeds: BTreeMap::new(),
        sampled: SampledSeeds::default(),
        build: serde_json::from_value(start["body"]["build"].clone()).unwrap_or_default(),
    };
    let sampled = sampled_in(events);
    run.sampled = SampledSeeds::of(&sampled);
    for e in events {
        match e["kind"].as_str() {
            Some("stage") => {
                run.stages.push(RunStage {
                    stage: text(&e["body"]["stage"]),
                    finished_at: text(&e["at"]),
                    summary: e["body"]["summary"].clone(),
                    duration_ms: e["body"]["duration_ms"].as_u64(),
                    error: e["body"]["error"].as_str().map(str::to_string),
                });
                run.updated_at = text(&e["at"]);
            }
            Some("seed") => {
                if let (Some(name), Some(seed)) =
                    (e["body"]["name"].as_str(), e["body"]["seed"].as_u64())
                {
                    run.seeds.insert(name.to_string(), seed);
                }
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

/// Every sampling seed the `events` of a run record, in request order.
fn sampled_in(events: &[serde_json::Value]) -> Vec<u64> {
    events
        .iter()
        .filter(|e| e["kind"] == "sampled")
        .filter_map(|e| e["body"]["seeds"].as_array())
        .flatten()
        .filter_map(serde_json::Value::as_u64)
        .collect()
}

/// Every seed a local model drew for the sampled requests of `run_id`, in
/// request order: the whole list [`Run::sampled`] summarises. Passing one
/// back as the request's seed replays that request's decoding.
pub fn read_sampled_seeds(workspace: &Workspace, run_id: &str) -> Result<Vec<Seed>, StoreError> {
    events_by_run(workspace)?
        .get(run_id)
        .map(|events| sampled_in(events).into_iter().map(Seed::new).collect())
        .ok_or_else(|| StoreError::UnknownRun(run_id.to_string()))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_start_without_a_build_is_the_start_recorded_before_builds_were() {
        let arguments = json!({ "n": 1 });
        assert_eq!(
            start_body("solve", &arguments, &BuildInfo::default()),
            json!({ "command": "solve", "arguments": { "n": 1 } })
        );
    }
}
