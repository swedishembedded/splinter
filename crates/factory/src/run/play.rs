// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent training data that people shape by
// working alongside the agent, for its clients. If your team needs
// expertise in human-in-the-loop agent training, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Interactive passes: trials people play.
//!
//! A rollout pass with `interactive: true` is never run unattended. Each
//! `splinter play` session ([`Factory::play`]) runs one attempt of one of
//! the pass's tasks on the person's terminal through the execution
//! backend; the finished session is graded by the one classifier and kept
//! under `play/<pass>/batches/<n>` of the run, outside any product. The
//! pass's product is the collection of every session played so far: its
//! identity names each session's report, so a new session makes a new
//! product of the pass and everything downstream of it (the dataset, the
//! candidates) is computed again on the next run, while every other stage
//! is reused. A session can continue an earlier one's conversation; the
//! earlier session is never changed.

use std::path::{Path, PathBuf};

use serde_json::json;
use splinter_core::factory::attempt::AttemptReport;
use splinter_core::factory::execution::SlotKey;
use splinter_core::factory::identity::StageIdentity;
use splinter_core::factory::product::TrialResult;
use splinter_store::factory::products::Staging;

use super::rollout::{
    arm, pin_files, Arm, AttemptEntry, Running, ATTEMPTS_FILE, SELECTED_FILE, TRIALS_FILE,
};
use super::rollout_report::report;
use super::{io, stage_error, write_lines, Factory, RunError};
use crate::backend::{read_reports, Batch};
use crate::plan::PlannedStage;
use crate::tree::link_tree;

/// Where an interactive pass's sessions are kept in a run.
fn sessions(factory: &Factory<'_>, pass: &str) -> PathBuf {
    factory.run.join("play").join(pass)
}

/// Every played session of the pass: its number and its report.
fn played(dir: &Path) -> Result<Vec<(u64, AttemptReport)>, RunError> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir.join("batches"))
        .into_iter()
        .flatten()
        .flatten()
    {
        let Some(n) = entry
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<u64>().ok())
        else {
            continue;
        };
        if let Some(report) = read_reports(&entry.path())?.into_iter().next() {
            out.push((n, report));
        }
    }
    out.sort_by_key(|(n, _)| *n);
    Ok(out)
}

pub(crate) fn identity(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    arm: &Arm<'_>,
) -> Result<StageIdentity, RunError> {
    let reports: Vec<_> = played(&sessions(factory, &arm.name))?
        .into_iter()
        .map(|(n, report)| json!([n, splinter_core::digest::digest_of(&report)]))
        .collect();
    let mut parameters = arm.parameters(factory)?;
    parameters["sessions"] = json!(reports);
    let pin = stage
        .source("tasks")
        .ok_or_else(|| stage_error(&stage.key, "its tasks come from no stage"))?;
    Ok(super::identity_of(
        &stage.versioned_operation(),
        &arm.name,
        parameters,
        vec![(pin.to_owned(), factory.product(pin)?.1.digest())],
        &factory.services.backend.identity(),
        Some(arm.pass.seed),
    ))
}

pub(crate) fn produce(
    factory: &Factory<'_>,
    arm: &Arm<'_>,
    staging: &Staging,
) -> Result<(), RunError> {
    let dir = sessions(factory, &arm.name);
    let run = Running::new(factory, arm, dir.clone())?;
    let mut entries = Vec::new();
    for (n, report) in played(&dir)? {
        entries.push(AttemptEntry {
            batch: n,
            finished_slot: true,
            trial: run.trial(&report, n)?,
        });
    }
    let trials: Vec<TrialResult> = entries.iter().map(|e| e.trial.clone()).collect();
    write_lines(&staging.path().join(TRIALS_FILE), &trials)?;
    write_lines(&staging.path().join(ATTEMPTS_FILE), &entries)?;
    let selected: Vec<_> = arm
        .tasks
        .iter()
        .map(|t| json!({ "id": t.id, "digest": t.digest }))
        .collect();
    write_lines(&staging.path().join(SELECTED_FILE), &selected)?;
    for name in ["files", "batches"] {
        if dir.join(name).is_dir() {
            link_tree(&dir.join(name), &staging.path().join(name))?;
        }
    }
    std::fs::write(
        staging.path().join("report.md"),
        report(&arm.name, &trials, &entries),
    )
    .map_err(io(staging.path()))
}

impl Factory<'_> {
    /// Play one session of `task` in the interactive pass `pass` on this
    /// terminal, continuing session `continue_from` when given. The run's
    /// tasks are pinned first when they are not yet. Returns the graded
    /// trial; the next run of the recipe picks it up.
    ///
    /// # Errors
    /// The pass is not interactive or does not offer the task, the session
    /// to continue does not exist, or the backend failed.
    pub fn play(
        &mut self,
        pass: &str,
        task: &str,
        continue_from: Option<u64>,
    ) -> Result<TrialResult, RunError> {
        let key = pass.to_owned();
        let planned: PlannedStage = self
            .resolved
            .plan
            .stage(&key)
            .cloned()
            .ok_or_else(|| stage_error(&key, "is not a stage of the recipe"))?;
        // The pass's tasks are pinned first, as a run would.
        let pin = planned
            .source("tasks")
            .ok_or_else(|| stage_error(&key, "is not a pass: it attempts no tasks"))?
            .to_owned();
        if !self.is_done(&pin) {
            let pin_stage = self
                .resolved
                .plan
                .stage(&pin)
                .cloned()
                .ok_or_else(|| stage_error(&pin, "not planned"))?;
            self.stage(&pin_stage)?;
        }
        let arm = arm(self, &planned)?;
        if !arm.pass.interactive {
            return Err(stage_error(
                &key,
                "is not interactive (`interactive: true`)",
            ));
        }
        if !arm.tasks.iter().any(|t| t.id == task) {
            return Err(stage_error(&key, format!("does not offer task {task}")));
        }
        let dir = sessions(self, pass);
        let run = Running::new(self, &arm, dir.clone())?;
        pin_files(self, &run)?;
        let sessions = played(&dir)?;
        let previous = match continue_from {
            Some(n) => {
                let (_, report) = sessions
                    .iter()
                    .find(|(m, _)| *m == n)
                    .ok_or_else(|| stage_error(&key, format!("has no session {n} to continue")))?;
                Some(run.batch_dir(n).join(&report.native_dir))
            }
            None => None,
        };
        let n = std::fs::read_dir(dir.join("batches"))
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| e.file_name().to_str().and_then(|n| n.parse::<u64>().ok()))
            .map(|n| n + 1)
            .max()
            .unwrap_or(0);
        let slot = SlotKey::new(
            pass,
            task,
            u32::try_from(n).map_err(|e| stage_error(&key, e))?,
        );
        let spec = run.spec(&slot, 0);
        let report = self.services.backend.play(
            &spec,
            Batch {
                specs: std::slice::from_ref(&spec),
                tasks: &self.view(&pin),
                files: &dir.join("files"),
                out: &run.batch_dir(n),
                concurrency: 1,
            },
            previous.as_deref(),
        )?;
        run.trial(&report, n)
    }
}
