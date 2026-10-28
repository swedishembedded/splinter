// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements held-out evaluation that stays clean of the
// data a model is trained on, for its clients. If your team needs expertise
// in keeping a benchmark's evaluation tasks out of fine-tuning data, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The split between the benchmark tasks a model is trained on and the ones
//! it is examined on, and the register that keeps it.
//!
//! Every bench experience records the split of its task in the task's own
//! environment spec, so the split is part of the experience's address and
//! cannot be edited away. The register records every trial that was admitted,
//! with the split it was admitted under and where that came from. Training
//! data is refused when it holds any experience that is not of a `train`
//! task, and any experience of a task that the register has seen on the
//! `eval` side anywhere: a task cannot be train in one export and eval in
//! another.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use splinter_core::experience::{Environment, Experience, ExperienceId};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// The class the admitted trials are registered under.
const TRIAL: &str = "bench_trial";

/// Which side of the benchmark's split a task is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    /// Training data.
    Train,
    /// Examination material: never training data.
    Eval,
}

impl Split {
    /// The name the benchmark exports it under.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Eval => "eval",
        }
    }

    /// The split an export names, when it names a known one.
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "train" => Ok(Self::Train),
            "eval" => Ok(Self::Eval),
            other => Err(format!("unknown set {other:?}: expected train or eval")),
        }
    }
}

/// Where the split of an admitted trial came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitSource {
    /// The export's manifest states it.
    Manifest,
    /// The export states none and the operator said which it is.
    Operator,
}

/// One trial that was admitted as an experience.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrialRecord {
    /// The trial's name; a name is admitted once.
    pub trial: String,
    /// The digest of the manifest it was admitted from.
    pub manifest_sha256: String,
    /// The benchmark task's id.
    pub task: String,
    /// The split it was admitted under.
    pub split: Split,
    /// Where that came from.
    pub split_source: SplitSource,
    /// The experience it became.
    pub experience: ExperienceId,
}

/// Every trial admitted so far.
pub fn trials(ctx: &Context) -> Result<Vec<TrialRecord>, OrchestratorError> {
    let mut found = Vec::new();
    for id in ctx.workspace().document_ids(TRIAL)? {
        if let Some(record) = ctx.workspace().get_document::<TrialRecord>(TRIAL, &id)? {
            found.push(record);
        }
    }
    Ok(found)
}

/// Registers an admitted trial; registering the same one again changes
/// nothing.
pub fn register(ctx: &Context, record: &TrialRecord) -> Result<(), OrchestratorError> {
    ctx.workspace().put_document(TRIAL, record)?;
    Ok(())
}

/// The tasks the register has seen on the `eval` side.
fn eval_tasks(trials: &[TrialRecord]) -> BTreeSet<&str> {
    trials
        .iter()
        .filter(|t| t.split == Split::Eval)
        .map(|t| t.task.as_str())
        .collect()
}

/// Why a trial of `task` cannot be admitted under `split`, when the register
/// holds the task on the other side.
pub fn conflict(trials: &[TrialRecord], task: &str, split: Split) -> Option<String> {
    let other = trials.iter().find(|t| t.task == task && t.split != split)?;
    Some(format!(
        "task {task} was admitted as {} (trial {}); it cannot also be {}",
        other.split.as_str(),
        other.trial,
        split.as_str()
    ))
}

/// Why `experience` may not be training data, when it may not: it is of a
/// bench task that is not `train`, or of one the register holds on the
/// `eval` side. `None` for any experience that is not a bench one.
pub fn exam_only(trials: &[TrialRecord], experience: &Experience) -> Option<String> {
    if experience.environment.kind != Environment::BENCH {
        return None;
    }
    let spec = &experience.environment.spec;
    let task = spec.pointer("/task/id").and_then(|v| v.as_str())?;
    let split = spec.get("set").and_then(|v| v.as_str());
    if split != Some(Split::Train.as_str()) {
        return Some(format!(
            "bench task {task} is in the {} set: exam material, never training data",
            split.unwrap_or("unknown")
        ));
    }
    eval_tasks(trials).contains(task).then(|| {
        format!("bench task {task} is registered as an eval task elsewhere: never training data")
    })
}

/// Why the experience `id` may not be training data, from the register
/// alone: for a caller that holds ids and not experiences.
pub fn exam_only_id(trials: &[TrialRecord], id: &ExperienceId) -> Option<String> {
    let trial = trials.iter().find(|t| &t.experience == id)?;
    (trial.split != Split::Train || eval_tasks(trials).contains(trial.task.as_str())).then(|| {
        format!(
            "experience {id} (bench trial {}, task {}) is exam material, never training data",
            trial.trial, trial.task
        )
    })
}
