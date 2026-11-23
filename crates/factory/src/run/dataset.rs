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
//! **Projection** (`native-sft`): an admitted trial's own recorded model
//! calls become one record ([`splinter_data::native_record`]): the
//! conversation exactly as the agent's model was sent it, its system prompt
//! and tools included, every action of the agent supervised. A trial whose
//! calls cannot be read, or do not match its trajectory, is excluded with
//! the exact reason; so is one whose record is longer than `max_block`
//! tokens of the target when a bound is set (counted by the trainer).
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

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::json;
use splinter_core::digest::{digest_of, Digest};
use splinter_core::factory::execution::FileRef;
use splinter_core::factory::identity::{StageIdentity, StageKind};
use splinter_core::factory::product::{DatasetProduct, TrialResult, WaterfallStep, DATASET_SCHEMA};
use splinter_data::request_log::read_request_log;
use splinter_data::{native_record, Objective, Record, RecordBody, RecordMetadata};
use splinter_store::factory::products::Staging;

use super::rollout::trials_of;
use super::{io, stage_error, write_json, write_lines, Factory, RunError};
use crate::plan::{stage_key, PlannedStage};
use crate::recipe::DatasetSection;

/// The dataset product's own description.
pub const DATASET_FILE: &str = "dataset.json";
/// The projections the stage implements.
pub const PROJECTIONS: &[&str] = &["native-sft"];
/// The admission rules the stage implements.
pub const ADMISSIONS: &[&str] = &["verified-success"];
/// The model role whose calls are the policy's.
pub const POLICY_ROLE: &str = "main";
/// The agent's request log in a trial directory.
pub const CALL_LOG: &str = "agent/requests.jsonl";
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

fn section<'r>(
    factory: &'r Factory<'_>,
    stage: &PlannedStage,
) -> Result<&'r DatasetSection, RunError> {
    factory
        .resolved
        .recipe
        .dataset
        .as_ref()
        .filter(|d| d.name == stage.name)
        .ok_or_else(|| stage_error(&stage.key, "is not the recipe's dataset"))
}

pub(crate) fn identity(
    factory: &Factory<'_>,
    stage: &PlannedStage,
) -> Result<StageIdentity, RunError> {
    let dataset = section(factory, stage)?;
    if !PROJECTIONS.contains(&dataset.projection.as_str()) {
        return Err(stage_error(
            &stage.key,
            format!(
                "projection {} is not one of {PROJECTIONS:?}",
                dataset.projection
            ),
        ));
    }
    if !ADMISSIONS.contains(&dataset.admission.as_str()) {
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
    for source in &dataset.sources {
        let key = stage_key(StageKind::Rollout, source);
        inputs.push((key.clone(), factory.product(&key)?.1.digest()));
    }
    let trainer = factory
        .services
        .trainer
        .filter(|_| dataset.max_block.is_some())
        .map(|t| t.identity());
    let parameters = json!({
        "objective": dataset.objective, "admission": dataset.admission, "projection": dataset.projection,
        "target": dataset.target, "max_block": dataset.max_block, "validation": dataset.validation,
        "policy_role": POLICY_ROLE,
    });
    Ok(super::identity_of(
        StageKind::Dataset,
        &dataset.name,
        parameters,
        inputs,
        trainer.as_deref().unwrap_or_default(),
        Some(dataset.seed),
    ))
}

/// The record of one admitted trial, or why it has none.
fn project(
    view: &Path,
    trial: &TrialResult,
) -> Result<(Record, usize, Vec<String>), (String, String)> {
    let dir = view.join(&trial.native_dir);
    let read = |name: &str| {
        std::fs::read(dir.join(name))
            .map_err(|e| ("no_native_calls".to_owned(), format!("{name}: {e}")))
    };
    let log = read(CALL_LOG)?;
    let trace = read(TRAJECTORY).map_err(|(_, d)| ("no_trajectory".to_owned(), d))?;
    let trajectory: atif::Trajectory =
        serde_json::from_slice(&trace).map_err(|e| ("invalid_trace".to_owned(), e.to_string()))?;
    let logged = read_request_log(&log).map_err(|e| ("calls_unreadable".to_owned(), e))?;
    let policy: Vec<_> = logged
        .iter()
        .filter(|r| r.role.as_deref().unwrap_or(POLICY_ROLE) == POLICY_ROLE)
        .filter_map(|r| r.call.clone())
        .collect();
    let native = native_record(&trajectory, &policy)
        .map_err(|e| ("unprojectable".to_owned(), e.to_string()))?;
    let mut metadata = RecordMetadata::of("native-sft", Objective::Sft);
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

fn file_ref(staging: &Staging, name: &str) -> Result<FileRef, RunError> {
    let path = staging.path().join(name);
    let file = std::fs::File::open(&path).map_err(io(&path))?;
    Ok(FileRef {
        path: name.to_owned(),
        digest: Digest::of_reader(file).map_err(io(&path))?,
    })
}

/// The groups held out: `percent` of them, drawn by `seed`, never all.
fn held_out(groups: &BTreeSet<String>, percent: u32, seed: u64) -> BTreeSet<String> {
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
    let mut raw = 0u64;
    let mut successes = 0u64;
    let mut projected: Vec<(String, TrialResult, Record, usize, Vec<String>)> = Vec::new();
    let mut excluded: Vec<Excluded> = Vec::new();
    let mut sources = Vec::new();
    for source in &dataset.sources {
        let key = stage_key(StageKind::Rollout, source);
        sources.push(factory.product(&key)?.1.digest());
        let view = factory.view(&key);
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
            match project(&view, &trial) {
                Ok((record, actions, tools)) => {
                    projected.push((source.clone(), trial, record, actions, tools))
                }
                Err((reason, detail)) => excluded.push(exclude(reason, detail)),
            }
        }
    }
    let eligible = projected.len() as u64;
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
    let group_of = |trial: &TrialResult| -> String {
        match dataset.validation.as_ref().map(|v| v.group_by.as_str()) {
            Some("task_family") => factory
                .resolved
                .collection
                .collection
                .task(&trial.task)
                .and_then(|t| t.family.clone())
                .unwrap_or_else(|| trial.task.clone()),
            _ => trial.task.clone(),
        }
    };
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
    };
    write_json(&staging.path().join(DATASET_FILE), &product)?;
    let report = Report {
        name: &dataset.name,
        waterfall: &waterfall,
        exclusions: &exclusions,
        supervised_actions,
        train: train.len(),
        validation: validation.len(),
        tools: &tool_counts,
        composition: &composition,
        per_task: &per_task,
        tasks: factory.resolved.collection.collection.tasks.len(),
    };
    std::fs::write(staging.path().join("report.md"), report.render()).map_err(io(staging.path()))
}

/// What the dataset's report shows.
struct Report<'a> {
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
