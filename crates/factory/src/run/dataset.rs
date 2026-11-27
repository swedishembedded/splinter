// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements faithful fine-tuning datasets built from
// verified agent trials, with every exclusion accounted for, for its
// clients. If your team needs expertise in agent training data, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The dataset stage: committed rollout trials projected into training
//! records, with nothing removed silently.
//!
//! **Admission** (`verified-success`): a trial is a candidate when it was
//! counted and passed at full credit with no veto. Every other trial is
//! excluded as `not_a_success:<status>`; a security violation never
//! becomes a demonstration whatever its score.
//!
//! **Projection** (`native-sft`, or `portable-sft` for an agent whose model
//! calls were not recorded, rebuilt from its ATIF trajectory by
//! [`splinter_data::portable_record`]): an admitted trial's own recorded model
//! calls become one record ([`splinter_data::native_record`]): the
//! conversation exactly as the agent's model was sent it, its system prompt
//! and tools included, every action of the agent supervised. A trial whose
//! calls cannot be read, or do not match its trajectory, is excluded with
//! the exact reason; so is one whose record is longer than `max_block`
//! tokens of the target when a bound is set (counted by the trainer).
//!
//! **Source**: every record's `metadata.trial` names the pass that holds its
//! trial, the slot, the agent and model, whether a person guided it, what
//! the target was produced with that the record leaves out (`withheld`: a
//! hint, a fork's guidance; privileged context is never supervised
//! silently), and, for an imported trial, the Harbor job and trial it came
//! from: the records of one dataset are told apart by it.
//!
//! **Validation**: whole groups (tasks, or task families) are held out, so
//! no task is in both files; the groups held out are drawn by the dataset's
//! seed.
//!
//! The product holds `train.jsonl` and `validation.jsonl`
//! (`generic-messages-v2`), `accepted.jsonl` and `excluded.jsonl` (one line
//! per trial), `supervision.jsonl` (each record's trial, split and
//! supervised messages), `dataset.json` (the [`DatasetProduct`]) and
//! `report.md`.

mod composition;
mod control;
mod provenance;
mod report;
mod selection;
mod supervision;
mod views;

pub(crate) use crate::holdout::held_out;

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::json;
use splinter_core::digest::Digest;
use splinter_core::factory::execution::FileRef;
use splinter_core::factory::identity::StageIdentity;
use splinter_core::factory::product::{DatasetProduct, TrialResult, WaterfallStep, DATASET_SCHEMA};
use splinter_data::{
    native_record, portable_record, Objective, Record, RecordBody, RecordMetadata, TrialSource,
};
use splinter_store::factory::products::Staging;

pub use composition::{measure, Composition, RecordMeasure, COMPOSITION_FILE};

use super::corpus::{policy_calls, CALL_LOG};
use super::rollout::trials_of;
use super::{io, stage_error, write_json, write_lines, Factory, RunError};
use crate::ops::{
    Control, Prefer, PreferenceOptions, ProjectParams, RewardOptions, Sessions, Supervise,
    ValidationSplit,
};
use crate::plan::PlannedStage;
use crate::recipe::ModelRef;

/// The dataset product's own description.
pub const DATASET_FILE: &str = "dataset.json";
/// The projections the stage implements.
pub const PROJECTIONS: &[&str] = &["native-sft", "portable-sft"];
/// The admission rules the stage implements.
pub const ADMISSIONS: &[&str] = &["verified-success"];
/// The agent's ATIF trajectory in a trial directory.
pub const TRAJECTORY: &str = "agent/trajectory.json";

/// One trial the dataset admitted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Accepted {
    /// The rollout pass.
    pub pass: String,
    /// The slot.
    pub slot: String,
    /// The task.
    pub task: String,
    /// The agent and model.
    pub agent: String,
    /// The model, when bound.
    pub model: Option<String>,
    /// The supervised actions.
    pub actions: usize,
    /// The split it went to.
    pub split: String,
    /// A person typed into the session.
    pub guided: bool,
}

/// One trial the dataset excluded, with the exact reason.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Excluded {
    /// The rollout pass.
    pub pass: String,
    /// The slot.
    pub slot: String,
    /// The task.
    pub task: String,
    /// The reason code.
    pub reason: String,
    /// What exactly, for a person.
    pub detail: String,
}

/// A `dataset.project` stage with its target resolved and its sources named.
struct Dataset {
    name: String,
    sources: Vec<String>,
    /// `answer.collect` products, projected by a view's objective.
    answers: Vec<String>,
    /// The `tasks.pin` stage whose tasks alone are admitted, when bound.
    tasks: Option<String>,
    objective: String,
    admission: String,
    projection: String,
    preference: Option<PreferenceOptions>,
    reward: Option<RewardOptions>,
    target: ModelRef,
    max_block: Option<u64>,
    thinking: bool,
    validation: Option<ValidationSplit>,
    supervise: Supervise,
    /// The most trials kept per task, and which.
    per_task: Option<(usize, Prefer)>,
    control: Option<Control>,
    /// How a policy-gradient dataset writes a session.
    sessions: Sessions,
    seed: u64,
}

/// An admitted trial projected as its record.
struct Admitted {
    /// The attempt stage that made it.
    source: String,
    trial: TrialResult,
    record: Record,
    /// The actions the record supervises.
    actions: usize,
    /// The actions the trial took, supervised or not.
    taken: usize,
    /// The tools the supervised actions call, in order.
    tools: Vec<String>,
}

/// A trial's record, the actions it supervises, their tools and the calls
/// answered with an error.
type Projection = (Record, usize, Vec<String>, BTreeSet<String>);

fn section(factory: &Factory<'_>, stage: &PlannedStage) -> Result<Dataset, RunError> {
    let params: ProjectParams = stage.params().map_err(|e| stage_error(&stage.key, e))?;
    let target = factory
        .resolved
        .model(&params.target)
        .cloned()
        .ok_or_else(|| {
            stage_error(
                &stage.key,
                format!("model role {} is not bound", params.target),
            )
        })?;
    Ok(Dataset {
        name: stage.key.clone(),
        sources: stage
            .sources("attempts")
            .into_iter()
            .map(str::to_owned)
            .collect(),
        tasks: stage.source("tasks").map(str::to_owned),
        answers: stage
            .sources("answers")
            .into_iter()
            .map(str::to_owned)
            .collect(),
        objective: params.objective,
        admission: params.admission,
        projection: params.projection,
        preference: params.preference,
        reward: params.reward,
        target,
        max_block: params.max_block,
        thinking: params.thinking,
        validation: params.validation,
        supervise: params.supervise,
        sessions: params.sessions,
        per_task: params
            .per_task
            .map(|limit| (limit, params.prefer.unwrap_or_default())),
        control: params.control,
        seed: params.seed,
    })
}

pub(crate) fn identity(
    factory: &Factory<'_>,
    stage: &PlannedStage,
) -> Result<StageIdentity, RunError> {
    let dataset = section(factory, stage)?;
    let sft = dataset.objective == "sft";
    if sft && !PROJECTIONS.contains(&dataset.projection.as_str()) {
        return Err(stage_error(
            &stage.key,
            format!(
                "projection {} is not one of {PROJECTIONS:?}",
                dataset.projection
            ),
        ));
    }
    if sft && !ADMISSIONS.contains(&dataset.admission.as_str()) {
        return Err(stage_error(
            &stage.key,
            format!(
                "admission {} is not one of {ADMISSIONS:?}",
                dataset.admission
            ),
        ));
    }
    if dataset.max_block.is_some() && factory.services.trainer.is_none() {
        return Err(stage_error(
            &stage.key,
            "max_block needs the trainer to count the target's tokens",
        ));
    }
    let mut inputs = Vec::new();
    // An unbound `tasks` adds no input, so the identity is the one it had
    // before the input existed.
    for source in dataset
        .sources
        .iter()
        .chain(&dataset.answers)
        .chain(&dataset.tasks)
    {
        inputs.push((source.clone(), factory.product(source)?.1.digest()));
    }
    let trainer = factory
        .services
        .trainer
        .filter(|_| dataset.max_block.is_some())
        .map(|t| t.identity());
    let parameters = json!({
        "objective": dataset.objective, "admission": dataset.admission, "projection": dataset.projection,
        "target": dataset.target, "max_block": dataset.max_block, "validation": dataset.validation,
        "policy_role": "first-request", "thinking": dataset.thinking,
    });
    let mut parameters = parameters;
    if let Some(options) = &dataset.reward {
        parameters["reward"] = json!(options);
    }
    if let Some(options) = &dataset.preference {
        parameters["preference"] = json!(options);
    }
    // A parameter at its default is left out, so a recipe that does not name
    // it keeps the identity it had before the parameter existed.
    if dataset.supervise != Supervise::All {
        parameters["supervise"] = json!(dataset.supervise);
    }
    if let Some((limit, prefer)) = dataset.per_task {
        parameters["per_task"] = json!(limit);
        parameters["prefer"] = json!(prefer);
    }
    if let Some(control) = dataset.control {
        parameters["control"] = json!(control);
    }
    if dataset.sessions != Sessions::PerCall {
        parameters["sessions"] = json!(dataset.sessions);
    }
    Ok(super::identity_of(
        &stage.versioned_operation(),
        &dataset.name,
        parameters,
        inputs,
        trainer.as_deref().unwrap_or_default(),
        Some(dataset.seed),
    ))
}

/// The record of one admitted trial under `projection`, or why it has none.
fn project(
    view: &Path,
    trial: &TrialResult,
    projection: &str,
) -> Result<Projection, (String, String)> {
    let dir = view.join(&trial.native_dir);
    let trace = std::fs::read(dir.join(TRAJECTORY))
        .map_err(|e| ("no_trajectory".to_owned(), format!("{TRAJECTORY}: {e}")))?;
    let trajectory: atif::Trajectory =
        serde_json::from_slice(&trace).map_err(|e| ("invalid_trace".to_owned(), e.to_string()))?;
    let projected = if projection == "portable-sft" {
        portable_record(&trajectory)
    } else {
        let log = std::fs::read(dir.join(CALL_LOG))
            .map_err(|e| ("no_native_calls".to_owned(), format!("{CALL_LOG}: {e}")))?;
        let policy = policy_calls(&log).map_err(|e| ("calls_unreadable".to_owned(), e))?;
        native_record(&trajectory, &policy)
    };
    let native = projected.map_err(|e| ("unprojectable".to_owned(), e.to_string()))?;
    let mut metadata = RecordMetadata::of(projection, Objective::Sft);
    metadata.group = Some(trial.task.clone());
    Ok((
        Record {
            body: native.record,
            metadata,
        },
        native.actions,
        native.tools,
        native.failed_calls,
    ))
}

/// The group `task` is held out with: itself, or its family (from whichever
/// pinned collection holds it) when the split is by family.
fn group_of(factory: &Factory<'_>, dataset: &Dataset, task: &str) -> String {
    match dataset.validation.as_ref().map(|v| v.group_by.as_str()) {
        Some("task_family") => factory
            .resolved
            .collections
            .values()
            .find_map(|c| c.collection.task(task))
            .and_then(|t| t.family.clone())
            .unwrap_or_else(|| task.to_owned()),
        _ => task.to_owned(),
    }
}

/// How many tasks the sources' passes were offered: the tasks of every
/// `tasks.pin` stage feeding them, each once.
fn tasks_offered(factory: &Factory<'_>, sources: &[String]) -> usize {
    let pins: std::collections::BTreeSet<&str> = sources
        .iter()
        .filter_map(|s| factory.resolved.plan.stage(s))
        .filter_map(|s| s.source("tasks"))
        .collect();
    pins.iter()
        .filter_map(|p| factory.collection(p))
        .map(|c| c.collection.tasks.len())
        .sum()
}

pub(crate) fn file_ref(staging: &Staging, name: &str) -> Result<FileRef, RunError> {
    let path = staging.path().join(name);
    let file = std::fs::File::open(&path).map_err(io(&path))?;
    Ok(FileRef {
        path: name.to_owned(),
        digest: Digest::of_reader(file).map_err(io(&path))?,
    })
}

#[allow(clippy::too_many_lines)]
pub(crate) fn produce(
    factory: &Factory<'_>,
    stage: &PlannedStage,
    staging: &Staging,
) -> Result<(), RunError> {
    let dataset = section(factory, stage)?;
    if dataset.objective != "sft" || !dataset.answers.is_empty() {
        return views::produce(factory, stage, staging, &dataset);
    }
    let mut raw = 0u64;
    let mut successes = 0u64;
    let mut projected: Vec<Admitted> = Vec::new();
    let mut failed_masked = 0u64;
    let mut excluded: Vec<Excluded> = Vec::new();
    let mut sources = Vec::new();
    let admitted_tasks: Option<BTreeSet<String>> = match &dataset.tasks {
        Some(pin) => Some(
            super::tasks::tasks_of(&factory.view(pin))?
                .tasks
                .into_iter()
                .map(|t| t.id)
                .collect(),
        ),
        None => None,
    };
    let mut in_tasks = 0u64;
    for source in &dataset.sources {
        sources.push(factory.product(source)?.1.digest());
        let view = factory.view(source);
        for trial in trials_of(&view)? {
            raw += 1;
            let exclude = |reason: String, detail: String| Excluded {
                pass: source.clone(),
                slot: trial.slot.to_string(),
                task: trial.task.clone(),
                reason,
                detail,
            };
            if let Some(tasks) = &admitted_tasks {
                if !tasks.contains(&trial.task) {
                    excluded.push(exclude(
                        "not_in_tasks".into(),
                        format!(
                            "{} is not a task of {}",
                            trial.task,
                            dataset.tasks.as_deref().unwrap_or_default()
                        ),
                    ));
                    continue;
                }
            }
            in_tasks += 1;
            if !trial.verified_success() {
                let status = serde_json::to_value(trial.grade.status)
                    .ok()
                    .and_then(|v| v.as_str().map(str::to_owned));
                let detail = trial
                    .grade
                    .cause
                    .as_ref()
                    .map(|c| c.message.clone())
                    .unwrap_or_else(|| {
                        format!(
                            "score {:?}, counted {}",
                            trial.grade.score, trial.grade.counted
                        )
                    });
                excluded.push(exclude(
                    format!("not_a_success:{}", status.unwrap_or_default()),
                    detail,
                ));
                continue;
            }
            successes += 1;
            let learned = project(&view, &trial, &dataset.projection).and_then(|mut found| {
                if let Some((hint, briefs)) = provenance::hint_of(factory, source) {
                    // Learned under the task's own instruction, from the
                    // tasks the run pinned.
                    let (instruction, shown) = crate::hint::material(
                        &provenance::pinned_view(factory, source).join(&trial.task),
                        &trial.task,
                        hint,
                        briefs.as_deref(),
                    )
                    .map_err(|e| ("hint_not_found".to_owned(), e))?;
                    crate::hint::unhint(&mut found.0.body, &instruction, hint, &shown)?;
                }
                if let Some(briefs) = provenance::fork_of(factory, source) {
                    // A continuation is learned without the turn it added.
                    let point = super::fork::fork_point(&briefs, &trial.task)
                        .map_err(|e| ("continuation_not_found".to_owned(), e))?;
                    super::fork::uncontinue(&mut found.0.body, &point.guidance)?;
                }
                Ok(found)
            });
            let learned = learned.and_then(|(record, taken, tools, failed)| {
                if dataset.supervise == Supervise::All {
                    return Ok((record, taken, taken, tools));
                }
                let (record, actions, masked) = supervision::productive(record, taken, &failed)?;
                failed_masked += masked as u64;
                let tools = supervision::supervised_tools(&record);
                Ok((record, actions, taken, tools))
            });
            match learned {
                Ok((mut record, actions, taken, tools)) => {
                    record.metadata.trial = Some(TrialSource {
                        pass: source.clone(),
                        slot: trial.slot.to_string(),
                        agent: trial.agent.name.clone(),
                        model: trial.model.clone(),
                        withheld: provenance::withheld(factory, source),
                        guided: trial.guided,
                        imported: trial.imported.clone(),
                    });
                    projected.push(Admitted {
                        source: source.clone(),
                        trial,
                        record,
                        actions,
                        taken,
                        tools,
                    })
                }
                Err((reason, detail)) => excluded.push(exclude(reason, detail)),
            }
        }
    }
    let eligible = projected.len() as u64;
    // A target trained for no-think mode is never shown a turn's reasoning:
    // its prompt opens no thinking block, and a supervised turn holding one
    // would teach it to emit reasoning it is never asked for.
    let mut reasoning_removed = 0u64;
    if !dataset.thinking {
        for admitted in &mut projected {
            if let RecordBody::Chat { messages, .. } = &mut admitted.record.body {
                for message in messages.iter_mut() {
                    if message.reasoning_content.take().is_some() {
                        reasoning_removed += 1;
                    }
                }
            }
        }
    }
    if let (Some(limit), Some(trainer)) = (dataset.max_block, factory.services.trainer) {
        let lines: Vec<String> = projected
            .iter()
            .map(|a| splinter_data::dataset_line(Objective::Sft, &a.record))
            .collect::<Result<_, _>>()
            .map_err(|e| stage_error(&stage.key, e))?;
        let lengths = trainer
            .measure(&dataset.target, &lines)
            .map_err(|e| stage_error(&stage.key, e))?;
        let mut kept = Vec::new();
        for (admitted, tokens) in projected.into_iter().zip(lengths) {
            if tokens > limit {
                excluded.push(Excluded {
                    pass: admitted.source,
                    slot: admitted.trial.slot.to_string(),
                    task: admitted.trial.task.clone(),
                    reason: "longer_than_max_block".into(),
                    detail: format!(
                        "{tokens} tokens of {} where max_block is {limit}",
                        dataset.target
                    ),
                });
            } else {
                kept.push(admitted);
            }
        }
        projected = kept;
    }
    let mut waterfall_limited = None;
    if let Some((limit, prefer)) = dataset.per_task {
        projected =
            selection::limit_per_task(projected, limit, prefer, dataset.seed, &mut excluded);
        waterfall_limited = Some(projected.len() as u64);
    }
    if projected.is_empty() {
        let mut reasons: BTreeMap<&str, u64> = BTreeMap::new();
        for e in &excluded {
            *reasons.entry(e.reason.as_str()).or_default() += 1;
        }
        let first = excluded
            .iter()
            .find(|e| !e.reason.starts_with("not_a_success"))
            .map(|e| e.detail.as_str())
            .unwrap_or("");
        return Err(stage_error(
            &stage.key,
            format!("no trial of {raw} became a record; excluded: {reasons:?} {first}"),
        ));
    }
    // A record of the control keeps its own task (and so its side of the
    // split); only its request is another task's.
    let request_of = match dataset.control {
        Some(Control::ShuffledRequests) => Some(
            control::shuffle_requests(&mut projected, dataset.seed)
                .map_err(|e| stage_error(&stage.key, e))?,
        ),
        None => None,
    };
    let group_of = |trial: &TrialResult| group_of(factory, &dataset, &trial.task);
    let groups: BTreeSet<String> = projected.iter().map(|a| group_of(&a.trial)).collect();
    let held = dataset
        .validation
        .as_ref()
        .map(|v| held_out(&groups, v.percent, dataset.seed))
        .unwrap_or_default();
    let (mut train, mut validation) = (Vec::new(), Vec::new());
    let (mut accepted, mut supervision) = (Vec::new(), Vec::new());
    let mut tool_counts: BTreeMap<String, u64> = BTreeMap::new();
    let mut composition: BTreeMap<String, u64> = BTreeMap::new();
    let mut per_task: BTreeMap<String, u64> = BTreeMap::new();
    let mut supervised_actions = 0u64;
    for (
        at,
        Admitted {
            source,
            trial,
            record,
            actions,
            tools,
            ..
        },
    ) in projected.iter().enumerate()
    {
        let split = if held.contains(&group_of(trial)) {
            "validation"
        } else {
            "train"
        };
        let line = splinter_data::dataset_line(Objective::Sft, record)
            .map_err(|e| stage_error(&stage.key, e))?;
        let target = if split == "train" {
            &mut train
        } else {
            &mut validation
        };
        let index = target.len();
        target.push(line);
        let supervised: Vec<usize> = match &record.body {
            RecordBody::Chat { messages, .. } => messages
                .iter()
                .enumerate()
                .filter(|(_, m)| m.train)
                .map(|(i, _)| i)
                .collect(),
            _ => Vec::new(),
        };
        let mut line = json!({
            "split": split, "line": index, "pass": source, "slot": trial.slot.to_string(), "task": trial.task,
            "trace": trial.trace.as_ref().map(|t| &t.path), "supervised_messages": supervised,
            "withheld": provenance::withheld(factory, source),
        });
        if let Some(request_of) = &request_of {
            line["request_of"] = json!(request_of[at]);
        }
        supervision.push(line);
        let agent = format!(
            "{}{} {}",
            if trial.guided { "guided " } else { "" },
            trial.agent.name,
            trial.agent.version.as_deref().unwrap_or("?")
        );
        accepted.push(Accepted {
            pass: source.clone(),
            slot: trial.slot.to_string(),
            task: trial.task.clone(),
            agent: agent.clone(),
            model: trial.model.clone(),
            actions: *actions,
            split: split.into(),
            guided: trial.guided,
        });
        for tool in tools {
            *tool_counts.entry(tool.clone()).or_default() += 1;
        }
        *composition
            .entry(format!(
                "{agent} / {}",
                trial.model.as_deref().unwrap_or("own model")
            ))
            .or_default() += 1;
        *per_task.entry(trial.task.clone()).or_default() += 1;
        supervised_actions += *actions as u64;
    }
    let leaked: Vec<&String> = {
        let train_groups: BTreeSet<String> = projected
            .iter()
            .map(|a| group_of(&a.trial))
            .filter(|g| !held.contains(g))
            .collect();
        held.iter().filter(|g| train_groups.contains(*g)).collect()
    };
    if !leaked.is_empty() {
        return Err(stage_error(
            &stage.key,
            format!("groups in both splits: {leaked:?}"),
        ));
    }
    let write_text = |name: &str, lines: &[String]| -> Result<(), RunError> {
        let path = staging.path().join(name);
        let mut text = lines.join("\n");
        if !text.is_empty() {
            text.push('\n');
        }
        std::fs::write(&path, text).map_err(io(&path))
    };
    write_text("train.jsonl", &train)?;
    write_text("validation.jsonl", &validation)?;
    write_lines(&staging.path().join("accepted.jsonl"), &accepted)?;
    write_lines(&staging.path().join("excluded.jsonl"), &excluded)?;
    write_lines(&staging.path().join("supervision.jsonl"), &supervision)?;
    let make_up = composition::composition(
        &projected,
        &composition::Build {
            train_records: train.len(),
            validation_records: validation.len(),
            failed_actions_masked: failed_masked,
            supervise: json!(dataset.supervise).as_str().unwrap_or_default(),
            per_task: dataset.per_task.map(|(limit, _)| limit),
            control: dataset.control.map(Control::name),
        },
    );
    write_json(&staging.path().join(COMPOSITION_FILE), &make_up)?;
    let mut exclusions: BTreeMap<String, u64> = BTreeMap::new();
    for e in &excluded {
        *exclusions.entry(e.reason.clone()).or_default() += 1;
    }
    let mut waterfall = vec![WaterfallStep {
        step: "raw_trials".into(),
        remaining: raw,
    }];
    if dataset.tasks.is_some() {
        waterfall.push(WaterfallStep {
            step: "in_tasks".into(),
            remaining: in_tasks,
        });
    }
    waterfall.extend([
        WaterfallStep {
            step: "verified_successes".into(),
            remaining: successes,
        },
        WaterfallStep {
            step: "projected".into(),
            remaining: eligible,
        },
    ]);
    if let Some(remaining) = waterfall_limited {
        waterfall.push(WaterfallStep {
            step: "per_task".into(),
            remaining,
        });
    }
    waterfall.push(WaterfallStep {
        step: "accepted".into(),
        remaining: accepted.len() as u64,
    });
    let product = DatasetProduct {
        schema: DATASET_SCHEMA.into(),
        name: dataset.name.clone(),
        objective: dataset.objective.clone(),
        projection: dataset.projection.clone(),
        admission: dataset.admission.clone(),
        target: dataset.target.to_string(),
        sources,
        train: file_ref(staging, "train.jsonl")?,
        validation: file_ref(staging, "validation.jsonl")?,
        accepted: file_ref(staging, "accepted.jsonl")?,
        excluded: file_ref(staging, "excluded.jsonl")?,
        supervision: file_ref(staging, "supervision.jsonl")?,
        waterfall: waterfall.clone(),
        exclusions: exclusions.clone(),
        thinking: dataset.thinking,
        control: dataset.control.map(|c| c.name().to_owned()),
        composition: Some(file_ref(staging, COMPOSITION_FILE)?),
        sessions: BTreeMap::new(),
    };
    write_json(&staging.path().join(DATASET_FILE), &product)?;
    let report = report::Report {
        thinking: dataset.thinking,
        reasoning_removed,
        name: &dataset.name,
        waterfall: &waterfall,
        exclusions: &exclusions,
        supervised_actions,
        train: train.len(),
        validation: validation.len(),
        tools: &tool_counts,
        composition: &composition,
        per_task: &per_task,
        tasks: tasks_offered(factory, &dataset.sources),
        failed_masked: (dataset.supervise == Supervise::Productive).then_some(failed_masked),
        control: dataset.control.map(Control::name),
        make_up: &make_up,
    };
    std::fs::write(staging.path().join("report.md"), report.render()).map_err(io(staging.path()))
}

/// The dataset product of a committed dataset view.
///
/// # Errors
/// Its description cannot be read.
pub fn dataset_of(view: &Path) -> Result<DatasetProduct, RunError> {
    super::read_json(&view.join(DATASET_FILE))
}
