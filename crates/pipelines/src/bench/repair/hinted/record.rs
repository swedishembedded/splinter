// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// turning agent failures into training data you can procure our services by
// sending an email to info@swedishembedded.com.

//! What a rationalisation keeps: one record per attempt and per own pass,
//! with every filter's verdict, and a kept trajectory's experience: on the
//! train side, with the verifier's pass and its lineage.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use splinter_core::annotation::{
    Annotation, AnnotationBody, Outcome as Verdicts, Producer, Strength,
};
use splinter_core::experience::{Environment, Experience, ExperienceId, Provenance, Task};
use splinter_core::kinds::BENCH_TASK;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

use super::filters::Verdict;
use super::material::{StudentTrial, TrainTask};
use super::{Checks, RATIONALISE_FORMAT};
use crate::bench::artifact::AgentContext;
use crate::bench::repair::route::InterventionKind;
use crate::bench::split::{self, Split, SplitSource, TrialRecord};
use crate::bench::{add_agent_context, VERIFIER_PRODUCER};

/// The class rationalisation records are stored under.
pub const RECORD_CLASS: &str = "bench_rationalise";

/// The key of a kept trajectory's `extra` its lineage is recorded under.
const LINEAGE: &str = "splinter_rationalise";

/// Where a kept trajectory came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    /// The student's own unaided pass.
    Own,
    /// A hinted attempt that passed every filter.
    Rationalised,
}

/// How a record ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// Every filter passed: it is experience.
    Kept,
    /// A filter failed.
    Rejected,
    /// The first three filters passed; the sufficiency run is owed.
    Candidate,
    /// The infrastructure stopped it: it decided nothing and is run again.
    Refused,
}

impl Outcome {
    /// Whether the record decides its attempt for good.
    #[must_use]
    pub fn decided(self) -> bool {
        matches!(self, Self::Kept | Self::Rejected)
    }

    /// Which of two records of one key stands: a decided one over a
    /// candidate over a refused one.
    fn standing(self) -> u8 {
        match self {
            Self::Refused => 0,
            Self::Candidate => 1,
            Self::Kept | Self::Rejected => 2,
        }
    }
}

/// Every filter's verdict, as far as the record got.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Filters {
    /// 1: the verifier passed the attempt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verified: Option<Verdict>,
    /// 2: its first tool call read the files it changes before writing them
    /// (reached only when the run opts in).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub real_observations: Option<Verdict>,
    /// 3, no answer present: its text holds none of the answer's content
    /// (the hint's wording, a file line nothing showed yet) and no term of
    /// the run's word list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub no_leak: Option<Verdict>,
    /// 4: its reasoning alone let the model pass again (reached only when
    /// the run opts in).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sufficient: Option<Verdict>,
}

impl Filters {
    /// Whether the screening filters `checks` applies passed: verified, no
    /// answer present, and real observations when the run opts in.
    #[must_use]
    pub fn screened(&self, checks: Checks) -> bool {
        let passed = |v: &Option<Verdict>| v.as_ref().is_some_and(|v| v.passed);
        passed(&self.verified)
            && passed(&self.no_leak)
            && (!checks.real_observations || passed(&self.real_observations))
    }
}

/// The earlier attempt a re-graded record judges again.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegradeOf {
    /// The earlier run's directory.
    pub run: PathBuf,
    /// The earlier record's key.
    pub key: String,
    /// The hinted trial as the earlier run recorded it.
    pub hinted_trial: PathBuf,
}

/// The record of one attempt, or of one own pass.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RationaliseRecord {
    /// Always [`RATIONALISE_FORMAT`].
    pub format: String,
    /// What it is keyed by.
    pub key: String,
    /// Where its trajectory came from.
    pub origin: Origin,
    /// The task's name.
    pub task: String,
    /// Its folder in the dataset.
    pub folder: String,
    /// Its difficulty rung.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub difficulty: Option<i64>,
    /// The student's job.
    pub student_job: PathBuf,
    /// The student's trial: the pass, or the failure the attempt answers.
    pub student_trial: String,
    /// The model that rationalised; `None` for an own pass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rationaliser: Option<String>,
    /// The intervention; `None` for an own pass.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intervention: Option<String>,
    /// Which sample of the task this is, from 0.
    pub sample: usize,
    /// How it ended.
    pub outcome: Outcome,
    /// Why it was refused or could not be kept, when it was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Every filter's verdict reached.
    pub filters: Filters,
    /// The hinted trial's directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hinted_trial: Option<PathBuf>,
    /// The sufficiency trial's directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sufficiency_trial: Option<PathBuf>,
    /// The kept experience.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experience: Option<ExperienceId>,
    /// The earlier attempt this record re-grades, when it does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regrade_of: Option<RegradeOf>,
}

impl RationaliseRecord {
    /// A record of `task` answering (or being) `student`, not yet decided.
    pub(super) fn new(
        key: &str,
        origin: Origin,
        task: &TrainTask,
        student: &StudentTrial,
        rationaliser: Option<&str>,
        sample: usize,
    ) -> Self {
        Self {
            format: RATIONALISE_FORMAT.into(),
            key: key.into(),
            origin,
            task: task.entry.name.clone(),
            folder: task.rel.clone(),
            difficulty: task.difficulty,
            student_job: student.job.clone(),
            student_trial: student.trial.clone(),
            rationaliser: rationaliser.map(str::to_string),
            intervention: rationaliser.map(|_| InterventionKind::HintedRetry.as_str().into()),
            sample,
            outcome: Outcome::Refused,
            reason: None,
            filters: Filters::default(),
            hinted_trial: None,
            sufficiency_trial: None,
            experience: None,
            regrade_of: None,
        }
    }

    /// This record refused, and why.
    pub(super) fn refuse(mut self, why: impl Into<String>) -> Self {
        self.outcome = Outcome::Refused;
        self.reason = Some(why.into());
        self
    }
}

/// A trajectory to keep, and what its experience records of it.
pub(super) struct Kept {
    /// The trajectory, its user turn the task's instruction.
    pub(super) trajectory: atif::Trajectory,
    /// What the model was given before its first turn.
    pub(super) context: AgentContext,
    /// The trial it is of.
    pub(super) trial: String,
    /// The sha256 of that trial's `result.json`.
    pub(super) result_sha256: String,
    /// The model that acted.
    pub(super) solver: String,
    /// Whether it saw material the student does not (the hint).
    pub(super) helped: bool,
}

/// Every record the state root holds, by key: the one that stands.
pub(super) fn held(
    ctx: &Context,
) -> Result<BTreeMap<String, RationaliseRecord>, OrchestratorError> {
    let mut held: BTreeMap<String, RationaliseRecord> = BTreeMap::new();
    let workspace = ctx.workspace();
    for id in workspace.documents_in_order(RECORD_CLASS)? {
        let Some(record) = workspace.get_document::<RationaliseRecord>(RECORD_CLASS, &id)? else {
            continue;
        };
        let stands = held
            .get(&record.key)
            .is_none_or(|h| record.outcome.standing() >= h.outcome.standing());
        if stands {
            held.insert(record.key.clone(), record);
        }
    }
    Ok(held)
}

/// Stores `record`, and, with `kept`, its experience first: the record then
/// names it and is kept.
pub(super) fn keep(
    ctx: &Context,
    mut record: RationaliseRecord,
    task: &TrainTask,
    kept: Option<&Kept>,
) -> Result<RationaliseRecord, OrchestratorError> {
    if let Some(kept) = kept {
        record.outcome = Outcome::Kept;
        record.reason = None;
        record.experience = Some(store_experience(ctx, &record, task, kept)?);
    }
    ctx.workspace().put_document(RECORD_CLASS, &record)?;
    Ok(record)
}

/// The kept trajectory as an experience of `task` on the train side, with
/// the verifier's pass and its trial registered.
fn store_experience(
    ctx: &Context,
    record: &RationaliseRecord,
    task: &TrainTask,
    kept: &Kept,
) -> Result<ExperienceId, OrchestratorError> {
    let refused =
        |why: String| OrchestratorError::Refused(format!("task {}: {why}", task.entry.name));
    if let Some(reason) = split::conflict(&split::trials(ctx)?, &task.entry.name, Split::Train) {
        return Err(refused(reason));
    }
    let mut trajectory = kept.trajectory.clone();
    let added = add_agent_context(&mut trajectory, &kept.context);
    if let Some(residue) = super::filters::hint_residue(&trajectory) {
        return Err(refused(residue));
    }
    let mut lineage = json!({
        "format": RATIONALISE_FORMAT,
        "origin": record.origin,
        "key": record.key,
        "student_trial": record.student_trial,
        "trial": kept.trial,
        "intervention": record.intervention,
        "rationaliser": record.rationaliser,
        "sample": record.sample,
        "filters": record.filters,
        "context": {"source": "agent/requests.jsonl", "source_sha256": kept.context.source_sha256,
            "added": added},
    });
    if let Some(source) = &record.regrade_of {
        lineage["regrade_of"] = json!(source);
    }
    let mut extra = match trajectory.extra.take() {
        Some(Value::Object(map)) => map,
        Some(other) => serde_json::Map::from_iter([("recorded".to_string(), other)]),
        None => serde_json::Map::new(),
    };
    extra.insert(LINEAGE.into(), lineage.clone());
    let created = extra
        .get("sven")
        .and_then(|s| s.get("created_at"))
        .and_then(Value::as_str)
        .map(str::to_string);
    trajectory.extra = Some(Value::Object(extra));
    // What is the task's, not the attempt's: every attempt at the task has
    // the same task address, the side the compile record states.
    let spec = json!({
        "task": {"id": task.entry.name, "version": null, "digest": task.entry.digest,
            "dimension": task.dimension},
        "set": Split::Train.as_str(),
        "snapshot": null,
        "verifier": {"digest": task.entry.digest, "scoring": null},
    });
    let task_value = Task::new(
        BENCH_TASK,
        Vec::new(),
        Environment::new(Environment::BENCH, spec),
        task.instruction.clone(),
        Vec::new(),
    )
    .map_err(|e| refused(e.to_string()))?;
    let mut provenance = Provenance::new(kept.solver.clone(), ctx.clock());
    provenance.teacher = kept.helped;
    if let Some(created) = created {
        provenance.created_at = created;
    }
    let experience = Experience::new(task_value, trajectory, None, provenance)
        .map_err(|e| refused(e.to_string()))?;
    let store = ctx.experiences();
    let id = store.put(&experience)?;
    store.annotate(&Annotation {
        experience: id.clone(),
        producer: Producer {
            name: VERIFIER_PRODUCER.into(),
            version: task.entry.digest.clone(),
        },
        body: AnnotationBody::Verdict {
            outcome: Verdicts::Pass,
            strength: Strength::Executable,
            evidence: json!({"trial": kept.trial, "result_sha256": kept.result_sha256,
                "rationalise": lineage}),
        },
    })?;
    // The trial register's split source: the compile record is the record
    // that states the side, as an export's manifest does.
    split::register(
        ctx,
        &TrialRecord {
            trial: kept.trial.clone(),
            conversation: None,
            manifest_sha256: kept.result_sha256.clone(),
            task: task.entry.name.clone(),
            split: Split::Train,
            split_source: SplitSource::Manifest,
            experience: id.clone(),
        },
    )?;
    Ok(id)
}
