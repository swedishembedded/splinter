// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent rollouts in which no graded trial is
// lost and no completed one is run twice, for its clients. If your team
// needs expertise in large-scale agent evaluation, you can procure our
// services by sending an email to info@swedishembedded.com.

//! A rollout pass: agents attempt tasks, every attempt is graded by the one
//! classifier, and the pass's trials become an immutable collection.
//!
//! **Selection is frozen.** A pass that takes the tasks unsolved by earlier
//! passes reads their committed products once: the tasks with no verified
//! success are part of its identity and written to `selected-tasks.jsonl`.
//!
//! **Slots and attempts.** Each selected task gets `attempts` logical slots.
//! A slot is filled by physical attempts in batches handed to the execution
//! backend. A graded attempt (pass or fail, whatever its score) finishes its
//! slot. An attempt the infrastructure decided (sandbox, model endpoint,
//! verifier, agent crash) returns its slot for another attempt, at most
//! `retries` more times; the last one then finishes the slot as it is, so
//! the trial is kept and counted as what it was. Every attempt's report is
//! kept in the product, the failed ones too. A slot whose attempts all went
//! without a report (the worker lost them) fails the stage once its retries
//! are spent: nothing is invented for it.
//!
//! **Crash recovery.** Each batch writes into its own directory under the
//! stage's work directory, named in the attempt's backend handle. A
//! restarted controller first asks the backend to reconcile each batch of
//! its unfinished attempts; those with a report are recorded, the others are
//! marked lost and their slots launched again.
//!
//! The same machinery runs an evaluation's arms ([`Arm`]).
//!
//! The product holds `trials.jsonl` (one [`TrialResult`] per slot, by slot),
//! `attempts.jsonl` (every physical attempt), `selected-tasks.jsonl`,
//! `files/` (the agent files and mounts the pass pinned), `batches/<n>/`
//! (each batch's native output; `batches/<n>/harbor` opens in Harbor's
//! viewer) and `report.md`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use splinter_core::clock::unix_now;
use splinter_core::digest::{digest_of, Digest};
use splinter_core::factory::attempt::AttemptReport;
use splinter_core::factory::execution::{
    AgentInvocation, EnvValue, ExecutionSpec, FileRef, ModelBinding, Mount, SandboxSpec, SlotKey,
    EXECUTION_SCHEMA,
};
use splinter_core::factory::identity::{StageIdentity, StageKind};
use splinter_core::factory::product::{AgentIdentity, TrialResult, TRIAL_SCHEMA};
use splinter_core::factory::task::TaskRef;
use splinter_core::factory::trial::{grade, GradingPolicy};
use splinter_store::factory::products::Staging;
use splinter_store::factory::runstate::{AttemptRecord, AttemptState, SlotState};

use super::{io, read_lines, stage_error, write_lines, Factory, RunError};
use crate::backend::Batch;
use crate::plan::{stage_key, PlannedStage};
use crate::recipe::RolloutPass;
use crate::tree::{link_tree, tree_digest};

/// The trials of a rollout product.
pub const TRIALS_FILE: &str = "trials.jsonl";
/// Every physical attempt of a rollout product.
pub const ATTEMPTS_FILE: &str = "attempts.jsonl";
/// The tasks a pass selected.
pub const SELECTED_FILE: &str = "selected-tasks.jsonl";

/// One physical attempt as the product keeps it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttemptEntry {
    /// The batch it ran in.
    pub batch: u64,
    /// Whether its trial finished its slot.
    pub finished_slot: bool,
    /// The graded trial (paths relative to the rollout product).
    pub trial: TrialResult,
}

/// The trials of a committed rollout product.
///
/// # Errors
/// The product's trials cannot be read.
pub fn trials_of(view: &Path) -> Result<Vec<TrialResult>, RunError> {
    read_lines(&view.join(TRIALS_FILE))
}

/// One set of tasks attempted by one agent on one model in one environment:
/// a rollout pass, or an arm of an evaluation.
pub(crate) struct Arm<'r> {
    /// The slot namespace in the run state, unique in the run.
    pub name: String,
    /// The stage key (errors, the work directory).
    pub key: String,
    /// The settings.
    pub pass: RolloutPass,
    /// The tasks, in order.
    pub tasks: Vec<&'r TaskRef>,
    /// The directory the task ids are relative to (a tasks product view).
    pub task_root: PathBuf,
    /// The model.
    pub model: Option<ModelBinding>,
}

/// A file pinned by its SHA-256, which the worker checks before it runs.
fn file_ref(repository: &Path, relative: &str) -> Result<FileRef, RunError> {
    let path = repository.join(relative);
    let file = std::fs::File::open(&path).map_err(io(&path))?;
    Ok(FileRef {
        path: relative.to_owned(),
        digest: Digest::sha256_of_reader(file).map_err(io(&path))?,
    })
}

impl Arm<'_> {
    fn sandbox(&self, factory: &Factory<'_>) -> Result<(SandboxSpec, Digest), RunError> {
        let environment = factory
            .resolved
            .environments
            .get(&self.pass.environment)
            .ok_or_else(|| {
                stage_error(
                    &self.key,
                    format!("environment {} is not resolved", self.pass.environment),
                )
            })?;
        let mut mounts = Vec::new();
        for mount in &environment.mounts {
            mounts.push(Mount {
                source: mount.source.clone(),
                digest: tree_digest(&factory.resolved.repository.join(&mount.source))?,
                target: mount.target.clone(),
                read_only: mount.read_only,
            });
        }
        let spec = SandboxSpec {
            provider: environment.sandbox.provider.clone(),
            image: environment.sandbox.image.clone(),
            cpus: environment.sandbox.resources.cpus,
            memory_mib: environment.sandbox.resources.memory_gb.map(|g| g * 1024),
            timeout_seconds: environment.timeout_seconds(),
            network: environment.network.mode,
            mounts,
        };
        Ok((spec, digest_of(environment)))
    }

    fn invocation(&self, factory: &Factory<'_>) -> Result<AgentInvocation, RunError> {
        let repository = &factory.resolved.repository;
        let files = self
            .pass
            .files
            .iter()
            .map(|f| file_ref(repository, f))
            .collect::<Result<_, _>>()?;
        let system_prompt = self
            .pass
            .system_prompt
            .as_deref()
            .map(|f| file_ref(repository, f))
            .transpose()?;
        let env = self
            .pass
            .environment_variables
            .iter()
            .map(|(name, value)| {
                let value = match value.strip_prefix("env:") {
                    Some(variable) => EnvValue::FromEnv(variable.to_owned()),
                    None => EnvValue::Literal(value.clone()),
                };
                (name.clone(), value)
            })
            .collect();
        Ok(AgentInvocation {
            name: self.pass.agent.clone(),
            version: None,
            args: self.pass.args.clone(),
            files,
            env,
            system_prompt,
        })
    }

    /// What the arm's identity is computed from, beside its inputs.
    pub(crate) fn parameters(&self, factory: &Factory<'_>) -> Result<Value, RunError> {
        let (sandbox, environment) = self.sandbox(factory)?;
        let tasks: Vec<_> = self.tasks.iter().map(|t| json!([t.id, t.digest])).collect();
        Ok(json!({
            "agent": self.invocation(factory)?,
            "model": self.model,
            "environment": environment,
            "sandbox": sandbox,
            "tasks": tasks,
            "attempts": self.pass.attempts,
            "retries": self.pass.retries,
            "budget": self.pass.budget,
            "timeout_policy": self.pass.timeout_policy,
        }))
    }
}

fn pass<'r>(factory: &'r Factory<'_>, stage: &PlannedStage) -> Result<&'r RolloutPass, RunError> {
    factory
        .resolved
        .recipe
        .rollouts
        .iter()
        .find(|p| p.name == stage.name)
        .ok_or_else(|| stage_error(&stage.key, "is not a rollout pass of the recipe"))
}

/// The arm of the rollout pass `stage`, its tasks frozen from the committed
/// upstream products.
fn arm<'r>(factory: &'r Factory<'_>, stage: &PlannedStage) -> Result<Arm<'r>, RunError> {
    let pass = pass(factory, stage)?;
    let tasks = &factory.resolved.collection.collection.tasks;
    let selected: Vec<&TaskRef> = if pass.tasks.ids.is_empty() {
        let mut solved = BTreeSet::new();
        for upstream in &pass.tasks.unsolved_by {
            let key = stage_key(StageKind::Rollout, upstream);
            factory.product(&key)?;
            for trial in trials_of(&factory.view(&key))? {
                if trial.verified_success() {
                    solved.insert(trial.task);
                }
            }
        }
        tasks.iter().filter(|t| !solved.contains(&t.id)).collect()
    } else {
        let wanted: BTreeSet<&str> = pass.tasks.ids.iter().map(String::as_str).collect();
        tasks
            .iter()
            .filter(|t| wanted.contains(t.id.as_str()))
            .collect()
    };
    let model = pass
        .model
        .as_ref()
        .map(|m| factory.resolved.connections.bind(m))
        .transpose()
        .map_err(|e| stage_error(&stage.key, e))?;
    Ok(Arm {
        name: pass.name.clone(),
        key: stage.key.clone(),
        pass: pass.clone(),
        tasks: selected,
        task_root: factory.view("tasks"),
        model,
    })
}

pub(crate) fn identity(
    factory: &Factory<'_>,
    stage: &PlannedStage,
) -> Result<StageIdentity, RunError> {
    let arm = arm(factory, stage)?;
    let mut inputs = vec![("tasks".to_owned(), factory.product("tasks")?.1.digest())];
    for upstream in &arm.pass.tasks.unsolved_by {
        let key = stage_key(StageKind::Rollout, upstream);
        inputs.push((key.clone(), factory.product(&key)?.1.digest()));
    }
    Ok(super::identity_of(
        StageKind::Rollout,
        &arm.name,
        arm.parameters(factory)?,
        inputs,
        &factory.services.backend.identity(),
        Some(arm.pass.seed),
    ))
}

pub(crate) fn produce(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    _identity: &StageIdentity,
    staging: &Staging,
) -> Result<(), RunError> {
    run_arm(factory, &arm(factory, stage)?, staging)
}

/// The handle a batch's attempts carry in the run state.
#[derive(Serialize, Deserialize)]
struct Handle {
    batch: u64,
}

/// An arm with everything resolved for execution.
struct Running<'a, 'r> {
    arm: &'a Arm<'r>,
    work: PathBuf,
    tasks: BTreeMap<String, &'r TaskRef>,
    agent: AgentInvocation,
    sandbox: SandboxSpec,
    policy: GradingPolicy,
}

impl Running<'_, '_> {
    fn spec(&self, slot: &SlotKey, attempt: u32) -> ExecutionSpec {
        let task = self.tasks[&slot.task];
        ExecutionSpec {
            schema: EXECUTION_SCHEMA.into(),
            slot: slot.clone(),
            attempt,
            task: task.clone(),
            task_dir: task.id.clone(),
            agent: self.agent.clone(),
            model: self.arm.model.clone(),
            sandbox: self.sandbox.clone(),
            seed: digest_of(&(self.arm.pass.seed, &slot.task, slot.slot, attempt)).leading_u64(),
            budget: self.arm.pass.budget,
            timeout_policy: self.arm.pass.timeout_policy,
        }
    }

    fn batch_dir(&self, batch: u64) -> PathBuf {
        self.work.join("batches").join(batch.to_string())
    }

    /// The graded trial of `report`, with paths relative to the product.
    fn trial(&self, report: &AttemptReport, batch: u64) -> Result<TrialResult, RunError> {
        let task = self.tasks.get(&report.slot.task).ok_or_else(|| {
            stage_error(
                &self.arm.key,
                format!(
                    "the backend reported task {}, which the arm did not run",
                    report.slot.task
                ),
            )
        })?;
        let prefix = format!("batches/{batch}");
        let out = self.batch_dir(batch);
        let pinned = |relative: &str| -> Result<FileRef, RunError> {
            let path = out.join(relative);
            let file = std::fs::File::open(&path).map_err(io(&path))?;
            Ok(FileRef {
                path: format!("{prefix}/{relative}"),
                digest: Digest::of_reader(file).map_err(io(&path))?,
            })
        };
        Ok(TrialResult {
            schema: TRIAL_SCHEMA.into(),
            slot: report.slot.clone(),
            attempt: report.attempt,
            task: task.id.clone(),
            task_digest: task.digest.clone(),
            agent: AgentIdentity {
                name: self.agent.name.clone(),
                version: report.agent_version.clone(),
                invocation: digest_of(&self.agent),
            },
            model: self.arm.model.as_ref().map(ModelBinding::reference),
            grade: grade(&report.evidence, &self.policy),
            trace: report.trace.as_deref().map(pinned).transpose()?,
            native_dir: format!("{prefix}/{}", report.native_dir),
            native: report
                .native_files
                .iter()
                .map(|f| pinned(f))
                .collect::<Result<_, _>>()?,
            started_at: report.started_at.clone(),
            finished_at: report.finished_at.clone(),
            usage: report.usage.clone(),
        })
    }
}

/// Record what `reports` say about the attempts of one batch. An attempt
/// with no report is lost (its slot goes back for another attempt, unless
/// its retries are spent, which fails the stage).
fn record(
    factory: &Factory<'_>,
    run: &Running<'_, '_>,
    batch: u64,
    attempts: &[AttemptRecord],
    reports: &[AttemptReport],
) -> Result<Vec<AttemptEntry>, RunError> {
    let mut entries = Vec::new();
    let allowed = run.arm.pass.retries as usize;
    for attempt in attempts {
        let number = attempt_number(factory, attempt)?;
        let used = factory.state.attempts(&attempt.slot)?.len();
        let Some(report) = reports
            .iter()
            .find(|r| r.slot == attempt.slot && r.attempt == number)
        else {
            factory.state.record_attempt(
                &factory.fence,
                attempt.id,
                AttemptState::Lost,
                None,
                unix_now(),
            )?;
            if used > allowed {
                return Err(stage_error(
                    &run.arm.key,
                    format!(
                        "slot {} lost {used} attempts without a report",
                        attempt.slot
                    ),
                ));
            }
            continue;
        };
        let trial = run.trial(report, batch)?;
        let retry = trial.grade.status.is_retryable() && used <= allowed;
        let digest = trial.digest();
        if retry {
            factory.state.record_attempt(
                &factory.fence,
                attempt.id,
                AttemptState::Failed,
                None,
                unix_now(),
            )?;
        } else {
            factory.state.record_attempt(
                &factory.fence,
                attempt.id,
                AttemptState::Succeeded,
                Some(digest.clone()),
                unix_now(),
            )?;
            factory
                .state
                .finish_slot(&factory.fence, &attempt.slot, &digest, unix_now())?;
        }
        entries.push(AttemptEntry {
            batch,
            finished_slot: !retry,
            trial,
        });
    }
    Ok(entries)
}

/// An attempt's number among its slot's attempts, from 0.
fn attempt_number(factory: &Factory<'_>, attempt: &AttemptRecord) -> Result<u32, RunError> {
    let all = factory.state.attempts(&attempt.slot)?;
    let index = all
        .iter()
        .position(|a| a.id == attempt.id)
        .unwrap_or(all.len());
    u32::try_from(index).map_err(|e| stage_error(&attempt.slot.pass, e))
}

fn append_entries(work: &Path, entries: &[AttemptEntry]) -> Result<(), RunError> {
    let path = work.join(ATTEMPTS_FILE);
    let mut all: Vec<AttemptEntry> = if path.exists() {
        read_lines(&path)?
    } else {
        Vec::new()
    };
    all.extend_from_slice(entries);
    write_lines(&path, &all)
}

/// Run `arm` to completion and write its product into `staging`.
pub(crate) fn run_arm(
    factory: &Factory<'_>,
    arm: &Arm<'_>,
    staging: &Staging,
) -> Result<(), RunError> {
    let (sandbox, _) = arm.sandbox(factory)?;
    let run = Running {
        arm,
        work: factory.work(&arm.key),
        tasks: arm.tasks.iter().map(|t| (t.id.clone(), *t)).collect(),
        agent: arm.invocation(factory)?,
        sandbox,
        policy: GradingPolicy {
            timeout: arm.pass.timeout_policy,
            budget: arm.pass.budget,
        },
    };
    std::fs::create_dir_all(&run.work).map_err(io(&run.work))?;
    pin_files(factory, &run)?;
    let ids: Vec<String> = arm.tasks.iter().map(|t| t.id.clone()).collect();
    factory.state.ensure_slots(
        &factory.fence,
        &arm.name,
        &ids,
        arm.pass.attempts,
        unix_now(),
    )?;

    // Reconcile what an interrupted invocation left running.
    let mut unfinished: BTreeMap<u64, Vec<AttemptRecord>> = BTreeMap::new();
    for attempt in factory.state.unfinished_attempts()? {
        if attempt.slot.pass == arm.name {
            let handle: Handle =
                serde_json::from_str(&attempt.backend).map_err(|e| stage_error(&arm.key, e))?;
            unfinished.entry(handle.batch).or_default().push(attempt);
        }
    }
    for (batch, attempts) in &unfinished {
        let reports = factory.services.backend.reconcile(&run.batch_dir(*batch))?;
        let entries = record(factory, &run, *batch, attempts, &reports)?;
        append_entries(&run.work, &entries)?;
    }

    let mut batch = next_batch(&run.work);
    loop {
        factory.cancelled()?;
        let handle =
            serde_json::to_string(&Handle { batch }).map_err(|e| stage_error(&arm.key, e))?;
        let mut claimed = Vec::new();
        while let Some(attempt) =
            factory
                .state
                .claim_next(&factory.fence, &arm.name, &handle, unix_now())?
        {
            claimed.push(attempt);
        }
        if claimed.is_empty() {
            break;
        }
        let mut specs = Vec::new();
        for attempt in &claimed {
            specs.push(run.spec(&attempt.slot, attempt_number(factory, attempt)?));
        }
        let reports = factory.services.backend.execute(Batch {
            specs: &specs,
            tasks: &arm.task_root,
            files: &run.work.join("files"),
            out: &run.batch_dir(batch),
            concurrency: arm.pass.concurrency,
        })?;
        let entries = record(factory, &run, batch, &claimed, &reports)?;
        append_entries(&run.work, &entries)?;
        batch += 1;
    }
    publish_into(factory, &run, staging)
}

/// The first batch number no earlier invocation used.
fn next_batch(work: &Path) -> u64 {
    std::fs::read_dir(work.join("batches"))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.file_name().to_str().and_then(|n| n.parse::<u64>().ok()))
        .map(|n| n + 1)
        .max()
        .unwrap_or(0)
}

/// Copy the agent files, prompt and mount sources into the arm's work, at
/// their repository paths, so the worker reads the run's copy.
fn pin_files(factory: &Factory<'_>, run: &Running<'_, '_>) -> Result<(), RunError> {
    let repository = &factory.resolved.repository;
    let files = run.work.join("files");
    let wanted = run
        .agent
        .files
        .iter()
        .chain(run.agent.system_prompt.iter())
        .map(|f| (f.path.clone(), f.digest.clone()));
    for (relative, digest) in wanted {
        let to = files.join(&relative);
        if !to.exists() {
            if let Some(parent) = to.parent() {
                std::fs::create_dir_all(parent).map_err(io(parent))?;
            }
            std::fs::copy(repository.join(&relative), &to).map_err(io(&to))?;
        }
        let file = std::fs::File::open(&to).map_err(io(&to))?;
        if Digest::sha256_of_reader(file).map_err(io(&to))? != digest {
            return Err(stage_error(
                &run.arm.key,
                format!("{relative} changed while the arm ran"),
            ));
        }
    }
    for mount in &run.sandbox.mounts {
        let to = files.join(&mount.source);
        if !to.exists() {
            link_tree(&repository.join(&mount.source), &to)?;
        }
        if tree_digest(&to)? != mount.digest {
            return Err(stage_error(
                &run.arm.key,
                format!("mount {} changed while the arm ran", mount.source),
            ));
        }
    }
    Ok(())
}

fn publish_into(
    factory: &Factory<'_>,
    run: &Running<'_, '_>,
    staging: &Staging,
) -> Result<(), RunError> {
    let path = run.work.join(ATTEMPTS_FILE);
    let entries: Vec<AttemptEntry> = if path.exists() {
        read_lines(&path)?
    } else {
        Vec::new()
    };
    let mut trials: Vec<TrialResult> = entries
        .iter()
        .filter(|e| e.finished_slot)
        .map(|e| e.trial.clone())
        .collect();
    trials.sort_by(|a, b| a.slot.cmp(&b.slot));
    let slots = factory.state.list_slots(&run.arm.name)?;
    let unfinished: Vec<String> = slots
        .iter()
        .filter(|s| s.state != SlotState::Finished)
        .map(|s| s.key.to_string())
        .collect();
    if !unfinished.is_empty() {
        return Err(stage_error(
            &run.arm.key,
            format!("slots without a trial: {}", unfinished.join(", ")),
        ));
    }
    if trials.len() != slots.len() {
        return Err(stage_error(
            &run.arm.key,
            format!("{} slots and {} trials", slots.len(), trials.len()),
        ));
    }
    write_lines(&staging.path().join(TRIALS_FILE), &trials)?;
    write_lines(&staging.path().join(ATTEMPTS_FILE), &entries)?;
    let selected: Vec<Value> = run
        .arm
        .tasks
        .iter()
        .map(|t| json!({ "id": t.id, "digest": t.digest }))
        .collect();
    write_lines(&staging.path().join(SELECTED_FILE), &selected)?;
    for name in ["files", "batches"] {
        let source = run.work.join(name);
        if source.is_dir() {
            link_tree(&source, &staging.path().join(name))?;
        }
    }
    std::fs::write(
        staging.path().join("report.md"),
        report(&run.arm.name, &trials, &entries),
    )
    .map_err(io(staging.path()))
}

/// The pass's report for a person.
fn report(name: &str, trials: &[TrialResult], attempts: &[AttemptEntry]) -> String {
    let mut by_status: BTreeMap<String, usize> = BTreeMap::new();
    for trial in trials {
        let status = serde_json::to_value(trial.grade.status)
            .ok()
            .and_then(|v| v.as_str().map(str::to_owned));
        *by_status.entry(status.unwrap_or_default()).or_default() += 1;
    }
    let solved: BTreeSet<&str> = trials
        .iter()
        .filter(|t| t.verified_success())
        .map(|t| t.task.as_str())
        .collect();
    let tasks: BTreeSet<&str> = trials.iter().map(|t| t.task.as_str()).collect();
    let passed = trials.iter().filter(|t| t.verified_success()).count();
    let counted = trials.iter().filter(|t| t.grade.counted).count();
    let mut out = format!(
        "# Rollout {name}\n\n| | |\n|---|---|\n| trials | {} |\n| physical attempts | {} |\n| counted | {counted} |\n\
         | verified successes | {passed} |\n| tasks solved | {} of {} |\n\n## Trials by status\n\n| status | trials |\n|---|---|\n",
        trials.len(),
        attempts.len(),
        solved.len(),
        tasks.len()
    );
    for (status, n) in by_status {
        out.push_str(&format!("| {status} | {n} |\n"));
    }
    let unsolved: Vec<String> = tasks
        .difference(&solved)
        .map(|t| format!("- {t}"))
        .collect();
    if !unsolved.is_empty() {
        out.push_str(&format!("\n## Unsolved tasks\n\n{}\n", unsolved.join("\n")));
    }
    out
}
