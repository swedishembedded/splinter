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

mod views;

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::json;
use splinter_core::digest::{digest_of, Digest};
use splinter_core::factory::execution::FileRef;
use splinter_core::factory::identity::StageIdentity;
use splinter_core::factory::product::{DatasetProduct, TrialResult, WaterfallStep, DATASET_SCHEMA};
use splinter_data::{
    native_record, portable_record, Objective, Record, RecordBody, RecordMetadata, TrialSource,
};
use splinter_store::factory::products::Staging;

use super::corpus::{policy_calls, CALL_LOG};
use super::rollout::trials_of;
use super::{io, stage_error, write_json, write_lines, Factory, RunError};
use crate::ops::{
    AttemptParams, Hint, PreferenceOptions, ProjectParams, RewardOptions, ValidationSplit,
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
    objective: String,
    admission: String,
    projection: String,
    preference: Option<PreferenceOptions>,
    reward: Option<RewardOptions>,
    target: ModelRef,
    max_block: Option<u64>,
    thinking: bool,
    validation: Option<ValidationSplit>,
    seed: u64,
}

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
    for source in dataset.sources.iter().chain(&dataset.answers) {
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
    Ok(super::identity_of(
        &stage.versioned_operation(),
        &dataset.name,
        parameters,
        inputs,
        trainer.as_deref().unwrap_or_default(),
        Some(dataset.seed),
    ))
}

/// The record of one admitted trial, or why it has none.
/// The record of one admitted trial under `projection`, or why it has none.
fn project(
    view: &Path,
    trial: &TrialResult,
    projection: &str,
) -> Result<(Record, usize, Vec<String>), (String, String)> {
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
        .filter_map(|p| factory.collections.get(*p))
        .map(|c| c.collection.tasks.len())
        .sum()
}

/// The view of the tasks the stage `source` attempted.
fn pinned_view(factory: &Factory<'_>, source: &str) -> std::path::PathBuf {
    let pin = factory
        .resolved
        .plan
        .stage(source)
        .and_then(|s| s.source("tasks"))
        .unwrap_or_default();
    factory.view(pin)
}

/// The hint the stage `source` showed, if it is a hinted pass.
/// The briefs a fork correction `source` took its fork points from.
fn fork_of(factory: &Factory<'_>, source: &str) -> Option<std::path::PathBuf> {
    let stage = factory.resolved.plan.stage(source)?;
    stage.source("continue_from")?;
    stage.source("briefs").map(|b| factory.view(b))
}

/// What the attempt stage `source` showed its targets that a record of
/// them leaves out: its hint, the guidance a fork added.
fn withheld(factory: &Factory<'_>, source: &str) -> Vec<String> {
    let mut out = Vec::new();
    if let Some((hint, _)) = hint_of(factory, source) {
        out.push(match hint {
            Hint::Solution => "hint:solution".to_owned(),
            Hint::Brief => "hint:brief".to_owned(),
        });
    }
    if fork_of(factory, source).is_some() {
        out.push("continuation".to_owned());
    }
    out
}

/// The hint the attempt stage `source` showed, and the view of its briefs.
fn hint_of(factory: &Factory<'_>, source: &str) -> Option<(Hint, Option<std::path::PathBuf>)> {
    let stage = factory.resolved.plan.stage(source)?;
    let hint = stage.params::<AttemptParams>().ok()?.hint?;
    Some((hint, stage.source("briefs").map(|b| factory.view(b))))
}

pub(crate) fn file_ref(staging: &Staging, name: &str) -> Result<FileRef, RunError> {
    let path = staging.path().join(name);
    let file = std::fs::File::open(&path).map_err(io(&path))?;
    Ok(FileRef {
        path: name.to_owned(),
        digest: Digest::of_reader(file).map_err(io(&path))?,
    })
}

/// The groups held out: `percent` of them, drawn by `seed`, never all.
pub(crate) fn held_out(groups: &BTreeSet<String>, percent: u32, seed: u64) -> BTreeSet<String> {
    let mut drawn: Vec<(u64, &String)> = groups
        .iter()
        .map(|g| (digest_of(&(seed, g)).leading_u64(), g))
        .collect();
    drawn.sort();
    let wanted = (groups.len() * percent as usize)
        .div_ceil(100)
        .min(groups.len().saturating_sub(1));
    drawn
        .into_iter()
        .take(wanted)
        .map(|(_, g)| g.clone())
        .collect()
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
    let mut projected: Vec<(String, TrialResult, Record, usize, Vec<String>)> = Vec::new();
    let mut excluded: Vec<Excluded> = Vec::new();
    let mut sources = Vec::new();
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
                if let Some((hint, briefs)) = hint_of(factory, source) {
                    // Learned under the task's own instruction, from the
                    // tasks the run pinned.
                    let (instruction, shown) = crate::hint::material(
                        &pinned_view(factory, source).join(&trial.task),
                        &trial.task,
                        hint,
                        briefs.as_deref(),
                    )
                    .map_err(|e| ("hint_not_found".to_owned(), e))?;
                    crate::hint::unhint(&mut found.0.body, &instruction, hint, &shown)?;
                }
                if let Some(briefs) = fork_of(factory, source) {
                    // A continuation is learned without the turn it added.
                    let point = super::fork::fork_point(&briefs, &trial.task)
                        .map_err(|e| ("continuation_not_found".to_owned(), e))?;
                    super::fork::uncontinue(&mut found.0.body, &point.guidance)?;
                }
                Ok(found)
            });
            match learned {
                Ok((mut record, actions, tools)) => {
                    record.metadata.trial = Some(TrialSource {
                        pass: source.clone(),
                        slot: trial.slot.to_string(),
                        agent: trial.agent.name.clone(),
                        model: trial.model.clone(),
                        withheld: withheld(factory, source),
                        guided: trial.guided,
                        imported: trial.imported.clone(),
                    });
                    projected.push((source.clone(), trial, record, actions, tools))
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
        for (_, _, record, _, _) in &mut projected {
            if let RecordBody::Chat { messages, .. } = &mut record.body {
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
            .map(|(_, _, record, _, _)| splinter_data::dataset_line(Objective::Sft, record))
            .collect::<Result<_, _>>()
            .map_err(|e| stage_error(&stage.key, e))?;
        let lengths = trainer
            .measure(&dataset.target, &lines)
            .map_err(|e| stage_error(&stage.key, e))?;
        let mut kept = Vec::new();
        for ((source, trial, record, actions, tools), tokens) in projected.into_iter().zip(lengths)
        {
            if tokens > limit {
                excluded.push(Excluded {
                    pass: source,
                    slot: trial.slot.to_string(),
                    task: trial.task.clone(),
                    reason: "longer_than_max_block".into(),
                    detail: format!(
                        "{tokens} tokens of {} where max_block is {limit}",
                        dataset.target
                    ),
                });
            } else {
                kept.push((source, trial, record, actions, tools));
            }
        }
        projected = kept;
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
    let group_of = |trial: &TrialResult| group_of(factory, &dataset, &trial.task);
    let groups: BTreeSet<String> = projected.iter().map(|(_, t, ..)| group_of(t)).collect();
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
    for (source, trial, record, actions, tools) in &projected {
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
        supervision.push(json!({
            "split": split, "line": index, "pass": source, "slot": trial.slot.to_string(), "task": trial.task,
            "trace": trial.trace.as_ref().map(|t| &t.path), "supervised_messages": supervised,
            "withheld": withheld(factory, source),
        }));
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
            .filter(|(_, t, ..)| !held.contains(&group_of(t)))
            .map(|(_, t, ..)| group_of(t))
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
    let mut exclusions: BTreeMap<String, u64> = BTreeMap::new();
    for e in &excluded {
        *exclusions.entry(e.reason.clone()).or_default() += 1;
    }
    let waterfall = vec![
        WaterfallStep {
            step: "raw_trials".into(),
            remaining: raw,
        },
        WaterfallStep {
            step: "verified_successes".into(),
            remaining: successes,
        },
        WaterfallStep {
            step: "projected".into(),
            remaining: eligible,
        },
        WaterfallStep {
            step: "accepted".into(),
            remaining: accepted.len() as u64,
        },
    ];
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
    };
    write_json(&staging.path().join(DATASET_FILE), &product)?;
    let report = Report {
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
    };
    std::fs::write(staging.path().join("report.md"), report.render()).map_err(io(staging.path()))
}

/// What the dataset's report shows.
struct Report<'a> {
    thinking: bool,
    reasoning_removed: u64,
    name: &'a str,
    waterfall: &'a [WaterfallStep],
    exclusions: &'a BTreeMap<String, u64>,
    supervised_actions: u64,
    train: usize,
    validation: usize,
    tools: &'a BTreeMap<String, u64>,
    composition: &'a BTreeMap<String, u64>,
    per_task: &'a BTreeMap<String, u64>,
    tasks: usize,
}

impl Report<'_> {
    fn render(&self) -> String {
        let table = |rows: &BTreeMap<String, u64>, head: &str| -> String {
            let mut out = format!("| {head} | n |\n|---|---|\n");
            for (k, v) in rows {
                out.push_str(&format!("| {k} | {v} |\n"));
            }
            out
        };
        let mut out = format!(
            "# Dataset {}\n\n## Waterfall\n\n| step | remaining |\n|---|---|\n",
            self.name
        );
        for step in self.waterfall {
            out.push_str(&format!("| {} | {} |\n", step.step, step.remaining));
        }
        out.push_str(&format!(
            "\n{} records ({} train, {} validation), {} supervised actions, {} of {} tasks covered.\n",
            self.train + self.validation,
            self.train,
            self.validation,
            self.supervised_actions,
            self.per_task.len(),
            self.tasks
        ));
        out.push_str(&if self.thinking {
            "\nReasoning is kept (thinking: true): the target is trained to reason before it answers.\n".to_owned()
        } else {
            format!(
                "\nThe target is trained for no-think mode (thinking: false): reasoning removed from {} turns.\n",
                self.reasoning_removed
            )
        });
        out.push_str(&format!(
            "\n## Exclusions\n\n{}",
            table(self.exclusions, "reason")
        ));
        out.push_str(&format!(
            "\n## Agents and models\n\n{}",
            table(self.composition, "agent / model")
        ));
        out.push_str(&format!(
            "\n## Supervised tool calls\n\n{}",
            table(self.tools, "tool")
        ));
        out.push_str(&format!(
            "\n## Records per task\n\n{}",
            table(self.per_task, "task")
        ));
        out
    }
}

/// The dataset product of a committed dataset view.
///
/// # Errors
/// Its description cannot be read.
pub fn dataset_of(view: &Path) -> Result<DatasetProduct, RunError> {
    super::read_json(&view.join(DATASET_FILE))
}

#[cfg(test)]
mod held_out_spec {
    use super::*;

    fn groups(n: usize) -> BTreeSet<String> {
        (0..n).map(|i| format!("g{i}")).collect()
    }

    /// `percent` of the groups, rounded up, are held out, but never every
    /// one: a split with no training group trains on nothing.
    #[test]
    fn the_share_is_rounded_up_and_never_takes_every_group() {
        assert_eq!(held_out(&groups(10), 25, 1).len(), 3);
        assert_eq!(held_out(&groups(2), 99, 1).len(), 1);
        assert!(held_out(&groups(1), 50, 1).is_empty());
        assert_eq!(held_out(&groups(10), 25, 1), held_out(&groups(10), 25, 1));
    }
}
