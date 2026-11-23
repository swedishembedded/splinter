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
//! The same machinery runs an evaluation's arms.
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
use splinter_core::factory::identity::StageIdentity;
use splinter_core::factory::product::{AgentIdentity, TrialResult, TRIAL_SCHEMA};
use splinter_core::factory::task::TaskRef;
use splinter_core::factory::trial::{grade, GradingPolicy};
use splinter_store::factory::products::Staging;
use splinter_store::factory::runstate::{AttemptRecord, AttemptState, SlotState};

use super::pass::{pass, pin, RolloutPass};
use super::rollout_report::report;
use super::{io, read_lines, stage_error, write_lines, Factory, RunError};
use crate::backend::Batch;
use crate::champion::{Champion, CHAMPION};
use crate::plan::PlannedStage;
use crate::recipe::ModelRef;
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
    /// The model, as the arm's identity sees it: a champion by its digest.
    pub model: Option<ModelBinding>,
    /// The champion the model is, served when the arm runs.
    pub champion: Option<Champion>,
}

/// The binding a stage's identity sees for `model`, and the champion it is:
/// a connection's model as named; a champion as its base's connection and
/// its digest, since the id it is served under is the server's to choose.
pub(crate) fn binding_of(
    factory: &Factory<'_>,
    key: &str,
    model: &ModelRef,
) -> Result<(ModelBinding, Option<Champion>), RunError> {
    let Some(champion) = factory.resolved.champions.get(&model.to_string()) else {
        let binding = factory
            .resolved
            .connections
            .bind(model)
            .map_err(|e| stage_error(key, e))?;
        return Ok((binding, None));
    };
    let mut binding = factory
        .resolved
        .connections
        .bind(&champion.record.base)
        .map_err(|e| stage_error(key, e))?;
    // By content, not by where its run is: the same champion is the same input.
    binding.model = format!("{CHAMPION}@{}", champion.digest.hex());
    Ok((binding, Some(champion.clone())))
}

/// The binding an arm's agent is given: a champion served from its base and
/// adapter through the trainer.
fn served(factory: &Factory<'_>, arm: &Arm<'_>) -> Result<Option<ModelBinding>, RunError> {
    arm.model
        .as_ref()
        .map(|binding| serve(factory, &arm.key, binding, arm.champion.as_ref()))
        .transpose()
}

/// `binding` as a model is reached: itself, or the champion it is, served
/// from its base and adapter through the trainer.
pub(crate) fn serve(
    factory: &Factory<'_>,
    key: &str,
    binding: &ModelBinding,
    champion: Option<&Champion>,
) -> Result<ModelBinding, RunError> {
    let Some(champion) = champion else {
        return Ok(binding.clone());
    };
    let trainer = factory
        .services
        .trainer
        .ok_or_else(|| stage_error(key, "a champion is served through the trainer"))?;
    let id = trainer
        .serve(&champion.record.base, champion.adapter.as_deref())
        .map_err(|e| stage_error(key, e))?;
    Ok(ModelBinding {
        model: id,
        ..binding.clone()
    })
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
            isolation: environment.isolation,
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
            "hint": self.pass.hint,
        }))
    }
}

/// The arm of the rollout pass `stage`, its tasks frozen from the committed
/// upstream products.
pub(crate) fn arm<'r>(factory: &'r Factory<'_>, stage: &PlannedStage) -> Result<Arm<'r>, RunError> {
    let pass = pass(factory, stage)?;
    let pin = pin(stage)?;
    let tasks = &factory
        .resolved
        .collections
        .get(pin)
        .ok_or_else(|| stage_error(&stage.key, format!("{pin} pins no tasks")))?
        .collection
        .tasks;
    let selected: Vec<&TaskRef> = if pass.tasks.ids.is_empty() {
        let mut solved = BTreeSet::new();
        for upstream in &pass.tasks.unsolved_by {
            factory.product(upstream)?;
            for trial in trials_of(&factory.view(upstream))? {
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
    let (model, champion) = match &pass.model {
        Some(m) => {
            let (binding, champion) = binding_of(factory, &stage.key, m)?;
            (Some(binding), champion)
        }
        None => (None, None),
    };
    Ok(Arm {
        name: pass.name.clone(),
        key: stage.key.clone(),
        pass,
        tasks: selected,
        task_root: factory.view(pin),
        model,
        champion,
    })
}

pub(crate) fn identity(
    factory: &Factory<'_>,
    stage: &PlannedStage,
) -> Result<StageIdentity, RunError> {
    let arm = arm(factory, stage)?;
    if arm.pass.interactive {
        return super::play::identity(factory, stage, &arm);
    }
    let pin = pin(stage)?;
    let mut inputs = vec![(pin.to_owned(), factory.product(pin)?.1.digest())];
    for upstream in &arm.pass.tasks.unsolved_by {
        inputs.push((upstream.clone(), factory.product(upstream)?.1.digest()));
    }
    Ok(super::identity_of(
        &stage.versioned_operation(),
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
    let arm = arm(factory, stage)?;
    if arm.pass.interactive {
        return super::play::produce(factory, &arm, staging);
    }
    run_arm(factory, &arm, staging)
}

/// The handle a batch's attempts carry in the run state.
#[derive(Serialize, Deserialize)]
struct Handle {
    batch: u64,
}

/// An arm with everything resolved for execution.
pub(crate) struct Running<'a, 'r> {
    pub(crate) arm: &'a Arm<'r>,
    pub(crate) work: PathBuf,
    tasks: BTreeMap<String, &'r TaskRef>,
    /// The model the agent is given (a champion, served).
    model: Option<ModelBinding>,
    agent: AgentInvocation,
    sandbox: SandboxSpec,
    policy: GradingPolicy,
}

impl<'a, 'r> Running<'a, 'r> {
    /// `arm` resolved to run with its work (batches, pinned files) under
    /// `work`.
    pub(crate) fn new(
        factory: &Factory<'_>,
        arm: &'a Arm<'r>,
        work: PathBuf,
    ) -> Result<Self, RunError> {
        let (sandbox, _) = arm.sandbox(factory)?;
        Ok(Self {
            arm,
            work,
            tasks: arm.tasks.iter().map(|t| (t.id.clone(), *t)).collect(),
            model: served(factory, arm)?,
            agent: arm.invocation(factory)?,
            sandbox,
            policy: GradingPolicy {
                timeout: arm.pass.timeout_policy,
                budget: arm.pass.budget,
            },
        })
    }
}

impl Running<'_, '_> {
    pub(crate) fn spec(&self, slot: &SlotKey, attempt: u32) -> ExecutionSpec {
        let task = self.tasks[&slot.task];
        ExecutionSpec {
            schema: EXECUTION_SCHEMA.into(),
            slot: slot.clone(),
            attempt,
            task: task.clone(),
            task_dir: task.id.clone(),
            agent: self.agent.clone(),
            model: self.model.clone(),
            sandbox: self.sandbox.clone(),
            seed: digest_of(&(self.arm.pass.seed, &slot.task, slot.slot, attempt)).leading_u64(),
            budget: self.arm.pass.budget,
            timeout_policy: self.arm.pass.timeout_policy,
        }
    }

    pub(crate) fn batch_dir(&self, batch: u64) -> PathBuf {
        self.work.join("batches").join(batch.to_string())
    }

    /// The graded trial of `report`, with paths relative to the product.
    pub(crate) fn trial(
        &self,
        report: &AttemptReport,
        batch: u64,
    ) -> Result<TrialResult, RunError> {
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
        let pinned = |relative: &str| pin_file(&out, &prefix, relative);
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
            model: self.model.as_ref().map(ModelBinding::reference),
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
            guided: guided(report),
            sandbox: report.sandbox.clone(),
            imported: None,
        })
    }
}

/// The file `relative` below a backend's output `out`, pinned by its digest
/// under the path it has in the product: `prefix/relative`.
pub(crate) fn pin_file(out: &Path, prefix: &str, relative: &str) -> Result<FileRef, RunError> {
    let path = out.join(relative);
    let file = std::fs::File::open(&path).map_err(io(&path))?;
    Ok(FileRef {
        path: format!("{prefix}/{relative}"),
        digest: Digest::of_reader(file).map_err(io(&path))?,
    })
}

/// Whether a person typed into the attempt: the worker marks a played
/// trial in its Harbor config (`config.agent.kwargs.human_guided`).
pub(crate) fn guided(report: &AttemptReport) -> bool {
    report
        .evidence
        .result
        .pointer("/config/agent/kwargs/human_guided")
        .and_then(Value::as_bool)
        == Some(true)
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
    let run = Running::new(factory, arm, factory.work(&arm.key))?;
    std::fs::create_dir_all(&run.work).map_err(io(&run.work))?;
    pin_files(factory, &run)?;
    // A hinted pass runs the run's own copies of its tasks, the hint in
    // their instruction.
    let hinted = run.work.join("hinted");
    let task_root = match arm.pass.hint {
        Some(hint) => {
            crate::hint::stage(&arm.task_root, &arm.tasks, hint, &hinted)
                .map_err(|e| stage_error(&arm.key, e))?;
            &hinted
        }
        None => &arm.task_root,
    };
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
            tasks: task_root,
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
pub(crate) fn pin_files(factory: &Factory<'_>, run: &Running<'_, '_>) -> Result<(), RunError> {
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
