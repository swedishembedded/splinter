// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the reuse of agent runs made elsewhere as
// verified training material, for its clients. If your team needs expertise
// in turning existing agent trajectories into training data, you can procure
// our services by sending an email to info@swedishembedded.com.

//! `attempt.import`: the trials of Harbor jobs run elsewhere, as attempts.
//!
//! Nothing runs. The execution backend matches each trial of the stage's
//! jobs (recipe inputs bound to its `jobs` port) to a task its `tasks` pin
//! (by Harbor's content digest of the task directory) and reports it as it
//! reports a trial it ran; the controller grades every report with the one
//! classifier, so an imported trial counts exactly as an `attempt.collect`
//! trial would, and every stage that reads attempts reads it the same way.
//! A trial the backend refused (Harbor never finished it, or its task is not
//! one the stage takes, or changed since) is listed in `refused.jsonl` with
//! its reason.
//!
//! The product is an attempts product: `trials.jsonl`, `attempts.jsonl` (one
//! attempt per trial), `selected-tasks.jsonl`, `report.md`, plus
//! `refused.jsonl` and `import/` (the backend's output: the trials copied
//! under `harbor/<job>/<trial>`). Each trial records the job and trial it
//! came from. The stage's identity holds every job's tree digest, so a
//! trial added to a job imports again; an import is a function of its jobs,
//! so an interrupted one is redone whole.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use splinter_core::digest::digest_of;
use splinter_core::factory::attempt::AttemptReport;
use splinter_core::factory::identity::StageIdentity;
use splinter_core::factory::product::{AgentIdentity, ImportedFrom, TrialResult, TRIAL_SCHEMA};
use splinter_core::factory::task::TaskRef;
use splinter_core::factory::trial::{grade, GradingPolicy};
use splinter_store::factory::products::Staging;

use super::pass::pin;
use super::rollout::{guided, pin_file, AttemptEntry, ATTEMPTS_FILE, SELECTED_FILE, TRIALS_FILE};
use super::rollout_report::report;
use super::{io, stage_error, write_lines, Factory, RunError};
use crate::backend::{Import, ImportRefusal};
use crate::ops::ImportParams;
use crate::plan::PlannedStage;
use crate::tree::link_tree;

/// The trials an import refused, one line each.
pub const REFUSED_FILE: &str = "refused.jsonl";
/// The backend's output in an import product.
const OUTPUT: &str = "import";

/// An `attempt.import` stage resolved: its parameters, the stage pinning
/// its tasks, its jobs (as the recipe names them) and the tasks it takes.
struct Resolved<'r> {
    params: ImportParams,
    pin: &'r str,
    jobs: Vec<String>,
    tasks: Vec<&'r TaskRef>,
}

fn resolved<'r>(
    factory: &'r Factory<'_>,
    stage: &'r PlannedStage,
) -> Result<Resolved<'r>, RunError> {
    let params: ImportParams = stage.params().map_err(|e| stage_error(&stage.key, e))?;
    let pin = pin(stage)?;
    let collection = &factory
        .resolved
        .collections
        .get(pin)
        .ok_or_else(|| stage_error(&stage.key, format!("{pin} pins no tasks")))?
        .collection;
    let jobs = stage
        .inputs_of("jobs")
        .into_iter()
        .map(|input| {
            factory
                .resolved
                .compiled
                .inputs
                .get(input)
                .map(|i| i.path.clone())
                .ok_or_else(|| stage_error(&stage.key, format!("input {input} is not declared")))
        })
        .collect::<Result<_, _>>()?;
    let tasks = collection
        .tasks
        .iter()
        .filter(|t| params.tasks.is_empty() || params.tasks.contains(&t.id))
        .collect();
    Ok(Resolved {
        params,
        pin,
        jobs,
        tasks,
    })
}

pub(crate) fn identity(
    factory: &Factory<'_>,
    stage: &PlannedStage,
) -> Result<StageIdentity, RunError> {
    let import = resolved(factory, stage)?;
    let mut jobs = Vec::new();
    for job in &import.jobs {
        let digest =
            factory.resolved.lock.files.get(job).ok_or_else(|| {
                stage_error(&stage.key, format!("Harbor job {job} was not resolved"))
            })?;
        jobs.push(json!([job, digest]));
    }
    let tasks: Vec<Value> = import
        .tasks
        .iter()
        .map(|t| json!([t.id, t.digest]))
        .collect();
    Ok(super::identity_of(
        &stage.versioned_operation(),
        &stage.key,
        json!({ "harbor_jobs": jobs, "model_label": import.params.model_label, "tasks": tasks }),
        vec![(
            import.pin.to_owned(),
            factory.product(import.pin)?.1.digest(),
        )],
        &factory.services.backend.identity(),
        None,
    ))
}

pub(crate) fn produce(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    staging: &Staging,
) -> Result<(), RunError> {
    let import = resolved(factory, stage)?;
    let tasks: Vec<TaskRef> = import.tasks.iter().map(|t| (*t).clone()).collect();
    let work = factory.work(&stage.key);
    if work.exists() {
        std::fs::remove_dir_all(&work).map_err(io(&work))?;
    }
    std::fs::create_dir_all(&work).map_err(io(&work))?;
    let out = work.join(OUTPUT);
    let jobs: Vec<PathBuf> = import
        .jobs
        .iter()
        .map(|j| factory.resolved.repository.join(j))
        .collect();
    let imported = factory.services.backend.import(Import {
        pass: &stage.key,
        jobs: &jobs,
        tasks: &tasks,
        task_root: &factory.view(import.pin),
        out: &out,
    })?;

    let mut trials = Vec::new();
    for report in &imported.reports {
        trials.push(trial(stage, &import, &tasks, &out, report)?);
    }
    trials.sort_by(|a, b| a.slot.cmp(&b.slot));
    let entries: Vec<AttemptEntry> = trials
        .iter()
        .map(|trial| AttemptEntry {
            batch: 0,
            finished_slot: true,
            trial: trial.clone(),
        })
        .collect();
    let refused: Vec<Value> = imported
        .refused
        .iter()
        .map(|r| refusal(stage, &import, r))
        .collect::<Result<_, _>>()?;

    let product = staging.path();
    write_lines(&product.join(TRIALS_FILE), &trials)?;
    write_lines(&product.join(ATTEMPTS_FILE), &entries)?;
    let taken: Vec<Value> = tasks
        .iter()
        .map(|t| json!({ "id": t.id, "digest": t.digest }))
        .collect();
    write_lines(&product.join(SELECTED_FILE), &taken)?;
    write_lines(&product.join(REFUSED_FILE), &refused)?;
    link_tree(&out, &product.join(OUTPUT))?;
    let mut text = report(&stage.key, &trials, &entries);
    text.push_str(&format!(
        "\n## Imported\n\nFrom {}: {} trials taken, {} refused ({REFUSED_FILE}).\n",
        import.jobs.join(", "),
        trials.len(),
        refused.len()
    ));
    std::fs::write(product.join("report.md"), text).map_err(io(product))
}

/// The job (as the recipe names it) and trial a backend path
/// `harbor/<job index>/<trial>` holds.
fn origin(jobs: &[String], native_dir: &str) -> Option<ImportedFrom> {
    let (index, trial) = native_dir.strip_prefix("harbor/")?.split_once('/')?;
    let job = jobs.get(index.parse::<usize>().ok()?)?;
    (!trial.is_empty() && !trial.contains('/')).then(|| ImportedFrom {
        harbor_job: job.clone(),
        trial: trial.to_owned(),
    })
}

fn refusal(
    stage: &PlannedStage,
    import: &Resolved<'_>,
    refused: &ImportRefusal,
) -> Result<Value, RunError> {
    let job = import.jobs.get(refused.job).ok_or_else(|| {
        stage_error(
            &stage.key,
            format!(
                "the backend refused a trial of job {}, which the import does not name",
                refused.job
            ),
        )
    })?;
    Ok(json!({ "harbor_job": job, "trial": refused.trial, "reason": refused.reason }))
}

/// The graded trial of an imported `report`, with paths relative to the
/// product.
fn trial(
    stage: &PlannedStage,
    import: &Resolved<'_>,
    tasks: &[TaskRef],
    out: &Path,
    report: &AttemptReport,
) -> Result<TrialResult, RunError> {
    let task = tasks
        .iter()
        .find(|t| t.id == report.slot.task)
        .filter(|_| report.slot.pass == stage.key)
        .ok_or_else(|| {
            stage_error(
                &stage.key,
                format!(
                    "the backend reported slot {}, which the import did not ask for",
                    report.slot
                ),
            )
        })?;
    let imported = origin(&import.jobs, &report.native_dir).ok_or_else(|| {
        stage_error(
            &stage.key,
            format!(
                "the backend put a trial at {}, not under harbor/<job>/<trial>",
                report.native_dir
            ),
        )
    })?;
    let result = &report.evidence.result;
    let text = |pointer: &str| result.pointer(pointer).and_then(Value::as_str);
    let recorded_model = match (
        text("/agent_info/model_info/provider"),
        text("/agent_info/model_info/name"),
    ) {
        (Some(provider), Some(name)) => Some(format!("{provider}/{name}")),
        (None, Some(name)) => Some(name.to_owned()),
        _ => None,
    };
    let pinned = |relative: &str| pin_file(out, OUTPUT, relative);
    Ok(TrialResult {
        schema: TRIAL_SCHEMA.into(),
        slot: report.slot.clone(),
        attempt: report.attempt,
        task: task.id.clone(),
        task_digest: task.digest.clone(),
        agent: AgentIdentity {
            name: text("/agent_info/name").unwrap_or("unknown").to_owned(),
            version: report.agent_version.clone(),
            // What the job ran the agent with, as Harbor recorded it.
            invocation: digest_of(&json!({
                "harbor_job": imported.harbor_job,
                "agent": result.pointer("/config/agent"),
            })),
        },
        model: import.params.model_label.clone().or(recorded_model),
        // The tasks' own timeout policy and no budget: what the job ran is
        // graded as a rollout pass with the recipe's defaults would be.
        grade: grade(&report.evidence, &GradingPolicy::default()),
        trace: report.trace.as_deref().map(pinned).transpose()?,
        native_dir: format!("{OUTPUT}/{}", report.native_dir),
        native: report
            .native_files
            .iter()
            .map(|f| pinned(f))
            .collect::<Result<_, _>>()?,
        started_at: report.started_at.clone(),
        finished_at: report.finished_at.clone(),
        usage: report.usage.clone(),
        guided: guided(report),
        sandbox: report.sandbox.clone(),
        imported: Some(imported),
    })
}
