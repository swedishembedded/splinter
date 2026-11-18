// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the loop from a failed agent attempt to a
// verified corrected one, for its clients. If your team needs expertise in
// turning agent failures into training data you can procure our services by
// sending an email to info@swedishembedded.com.

//! Repairs of failed practice attempts: a teacher continues the student's own
//! attempt from its last good step, and the task's verifier decides.
//!
//! The failed trials of practice runs (`tasks harbor practice` directories,
//! each with `report.json`, `export/` and `staged/`) are diagnosed
//! ([`super::diagnose`]) and routed by their class ([`route`]): a model
//! failure whose decisive step the diagnosis placed goes to the fork
//! correction ([`ForkCorrection`]); an infrastructure or task failure, and a
//! model failure with no decisive step, are not repaired, and the report says
//! why. Every intervention is an [`Intervention`]; fork correction is the one
//! there is for an examiner's tasks.
//!
//! A fork correction lays out the task's pristine workspace through the
//! examiner (`bench task workspace`), keeps the student's steps through the
//! last good one, has sven's fork replay them into that tree (each replayed
//! observation held to the recorded one) and the teacher continue the same
//! tool loop under the profile the student ran. The resulting workspace and
//! the composed trajectory go back to the examiner, whose task verifier grades
//! them outside any agent (`bench task grade`). Only a pass is accepted; a
//! fail, or a fork the replay refused, is counted and kept as a record, and
//! never becomes experience.
//!
//! An accepted repair is an experience of the student's task whose
//! trajectory is the student's stored trajectory through the kept step (its
//! system prompt and those steps, marked `is_copied_context` so no view
//! supervises them) followed by the teacher's steps. Its lineage is in the
//! trajectory (`extra.correction`: the student experience and trial, the
//! diagnosis, the kept step, the shared prefix length, the teacher, the
//! verifier's grade) and in its annotations: the verifier's pass and a
//! `RevisionOf` relation to the student's attempt. A preference pair is
//! therefore the shared prefix, the teacher's suffix and
//! the student's own suffix from the same step; supervised training on it
//! learns the teacher's suffix only.
//!
//! **Idempotent.** A repair is keyed by the student's experience, the teacher
//! and the intervention, and its record is stored under the state root; a
//! rerun finds a graded record (accepted or rejected) and does nothing. A
//! refused one (the fork did not replay, the teacher or the examiner failed)
//! decided nothing about the attempt and is tried again; it stores nothing. A
//! crash between storing an accepted experience and its record leaves an
//! experience no set names, and the rerun repairs the attempt again.

mod commands;
mod compose;
mod fork;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use splinter_agent::CancelToken;
use splinter_core::annotation::{
    Annotation, AnnotationBody, Outcome, Producer, RelationKind, Strength,
};
use splinter_core::diagnosis::DiagnosticResult;
use splinter_core::digest::{canonical_json, Digest};
use splinter_core::experience::{Experience, ExperienceId};
use splinter_core::model_ref::ModelRef;
use splinter_core::outcome::{AttemptReport, FailureClass};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::experiences::{ExperienceSet, SetId};

pub use commands::{Grade, GRADE_SCHEMA};
pub use fork::ForkCorrection;

use super::diagnose::{diagnose_dirs, DiagnosedTrial, DiagnosisReport};
use super::practice::BenchCheckout;
use super::VERIFIER_PRODUCER;

/// The format tag of a [`RepairReport`] and of the records it keeps.
pub const REPAIR_FORMAT: &str = "splinter.bench-repair.v1";

/// The producer of the relations and reports a repair records.
pub const PRODUCER: &str = "splinter/repair";

/// The most repairs one run makes when the request does not say.
pub const DEFAULT_MAX_REPAIRS: usize = 32;

/// Repairs run at once when the request does not say: enough to keep a
/// served teacher's decode slots busy.
pub const DEFAULT_REPAIR_JOBS: usize = 8;

/// The wall-clock time one continuation may take when the request does not
/// say.
pub const DEFAULT_REPAIR_DEADLINE: Duration = Duration::from_secs(20 * 60);

/// The class repair records are stored under.
const RECORD: &str = "bench_repair";

/// One repair run.
pub struct RepairRequest<'a> {
    /// The practice run directories whose failed trials are repaired.
    pub runs: &'a [PathBuf],
    /// The examiner.
    pub checkout: &'a BenchCheckout,
    /// The model that continues the attempts.
    pub teacher: &'a ModelRef,
    /// The most repairs this run makes.
    pub max: usize,
    /// Repairs run at once.
    pub jobs: usize,
    /// The time one continuation may take.
    pub deadline: Duration,
    /// The classes repaired; empty for every class [`route`] sends to an
    /// intervention.
    pub classes: &'a [FailureClass],
    /// Stops the run between repairs.
    pub cancel: CancelToken,
}

/// Where a diagnosed failure goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "route", content = "why", rename_all = "snake_case")]
pub enum Route {
    /// To the fork correction, after the last good step.
    ForkCorrection,
    /// Not repaired.
    Skip(&'static str),
}

/// The route of a diagnosis: a model failure with a last good step to fork
/// after is corrected; nothing else is.
#[must_use]
pub fn route(diagnosis: &DiagnosticResult) -> Route {
    if !diagnosis.class.is_model() {
        return Route::Skip("not a model failure: the infrastructure or the task decided it");
    }
    if diagnosis.class == FailureClass::HistoryProbing {
        return Route::Skip(
            "the attempt read material it must never have seen; its prefix is not replayed",
        );
    }
    if diagnosis.last_good_step().is_none() {
        return Route::Skip("the diagnosis places no decisive step after a good one");
    }
    Route::ForkCorrection
}

/// How one repair ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepairOutcome {
    /// The verifier passed the repaired attempt: it is experience.
    Accepted,
    /// The verifier did not pass it.
    Rejected,
    /// It could not be made (the fork was refused, the teacher or the
    /// examiner failed).
    Refused,
}

/// The record of one repair, stored under the state root and in the report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepairRecord {
    /// Always [`REPAIR_FORMAT`].
    pub format: String,
    /// What the repair is keyed by: the student's experience, the teacher and
    /// the intervention.
    pub key: String,
    /// The student's attempt.
    pub student: ExperienceId,
    /// Its trial.
    pub trial: String,
    /// Its task.
    pub task: String,
    /// Its domain.
    pub domain: String,
    /// Why it failed.
    pub diagnosis: DiagnosticResult,
    /// The intervention.
    pub intervention: String,
    /// The last student step kept, in the exported trajectory's numbering.
    pub keep_through: u64,
    /// The teacher.
    pub teacher: String,
    /// How it ended.
    pub outcome: RepairOutcome,
    /// Why it was refused, when it was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// The verifier's grade, when it graded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grade: Option<Grade>,
    /// The accepted repair's experience.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experience: Option<ExperienceId>,
    /// The leading steps the repair shares with the student's stored
    /// trajectory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_steps: Option<usize>,
    /// The teacher's agent steps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher_actions: Option<u32>,
}

/// Repairs of one class, counted.
#[derive(Clone, Debug, Default, Serialize)]
pub struct ClassYield {
    /// Diagnosed failures of the class.
    pub failed: usize,
    /// Of those, sent to an intervention.
    pub routed: usize,
    /// Accepted by the verifier.
    pub accepted: usize,
    /// Graded and not passed.
    pub rejected: usize,
    /// Not made.
    pub refused: usize,
    /// Found already repaired, changed in no way.
    pub held: usize,
}

/// What a repair run did.
#[derive(Clone, Debug, Serialize)]
pub struct RepairReport {
    /// Always [`REPAIR_FORMAT`].
    pub format: &'static str,
    /// The teacher.
    pub teacher: String,
    /// Per class.
    pub by_class: BTreeMap<String, ClassYield>,
    /// The records of this run's repairs and of the ones held.
    pub repairs: Vec<RepairRecord>,
    /// The failures not routed, and why.
    pub skipped: Vec<(String, String)>,
    /// The accepted repairs of this run, the set datasets build from.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repair_set: Option<SetId>,
    /// The diagnosis it ran on.
    pub diagnosis: DiagnosisReport,
}

/// One failed attempt to repair.
pub struct Case<'a> {
    /// The trial as diagnosed.
    pub trial: &'a DiagnosedTrial,
    /// Its diagnosis.
    pub diagnosis: &'a DiagnosticResult,
    /// The practice run it came from.
    pub run: &'a Path,
    /// The student's stored experience.
    pub student: &'a Experience,
    /// Its id.
    pub student_id: &'a ExperienceId,
}

/// What an intervention made of a case: the grade, and the experience to
/// keep when the grade is a pass.
pub struct Attempted {
    /// The last student step kept.
    pub keep_through: u64,
    /// The verifier's grade.
    pub grade: Grade,
    /// The repaired attempt.
    pub experience: Experience,
    /// The leading steps shared with the student's stored trajectory.
    pub prefix_steps: usize,
    /// The teacher's agent steps.
    pub teacher_actions: u32,
}

/// A way of turning a failed attempt into a verified one.
pub trait Intervention: Sync {
    /// The name its records carry.
    fn name(&self) -> &'static str;

    /// Makes a repaired attempt of `case` and has it graded.
    ///
    /// # Errors
    /// Why no repaired attempt could be made or graded.
    fn attempt(&self, ctx: &Context, case: &Case<'_>) -> Result<Attempted, String>;
}

/// The key of a repair.
fn key_of(
    student: &ExperienceId,
    teacher: &str,
    intervention: &str,
) -> Result<String, OrchestratorError> {
    #[derive(Serialize)]
    struct Key<'a> {
        format: &'static str,
        student: &'a ExperienceId,
        teacher: &'a str,
        intervention: &'a str,
    }
    let bytes = canonical_json(&Key {
        format: REPAIR_FORMAT,
        student,
        teacher,
        intervention,
    })
    .map_err(|e| OrchestratorError::Refused(format!("a repair key: {e}")))?;
    Ok(Digest::of(&bytes).hex().to_string())
}

/// Every repair record the state root holds, by key.
fn held_records(ctx: &Context) -> Result<BTreeMap<String, RepairRecord>, OrchestratorError> {
    let mut held = BTreeMap::new();
    for id in ctx.workspace().document_ids(RECORD)? {
        if let Some(record) = ctx.workspace().get_document::<RepairRecord>(RECORD, &id)? {
            // A graded record decides; a refused one only stands in for none.
            let graded = held
                .get(&record.key)
                .is_some_and(|r: &RepairRecord| r.outcome != RepairOutcome::Refused);
            if !graded {
                held.insert(record.key.clone(), record);
            }
        }
    }
    Ok(held)
}

/// Diagnoses the failed trials of `request`'s runs, routes them, repairs the
/// routed ones with the teacher and keeps the accepted repairs; see the
/// module documentation.
///
/// # Errors
/// [`OrchestratorError`] when the store cannot be read or written, the
/// teacher cannot be resolved, or the run is cancelled.
pub fn repair(
    ctx: &Context,
    request: &RepairRequest<'_>,
) -> Result<RepairReport, OrchestratorError> {
    let teacher = ctx.model(request.teacher)?;
    let intervention = ForkCorrection {
        checkout: request.checkout,
        teacher,
        deadline: request.deadline,
        cancel: request.cancel.clone(),
    };
    repair_with(ctx, request, &intervention)
}

/// [`repair`] with the intervention given.
///
/// # Errors
/// As [`repair`].
pub fn repair_with(
    ctx: &Context,
    request: &RepairRequest<'_>,
    intervention: &(dyn Intervention + '_),
) -> Result<RepairReport, OrchestratorError> {
    let teacher_identity = ctx.model(request.teacher)?.identity;
    let diagnosis = diagnose_dirs(request.runs);
    let held = held_records(ctx)?;
    // The student's experience in this state root: the intake's register of
    // the trial (an export admitted here), else what the run's report says.
    let mut admitted: BTreeMap<String, Vec<ExperienceId>> = BTreeMap::new();
    for record in super::split::trials(ctx)? {
        admitted
            .entry(record.trial)
            .or_default()
            .push(record.experience);
    }
    let store = ctx.experiences();
    let mut by_class: BTreeMap<String, ClassYield> = BTreeMap::new();
    let mut skipped = Vec::new();
    let mut repairs = Vec::new();
    let mut cases = Vec::new();
    for trial in &diagnosis.attempts {
        let Some(result) = &trial.diagnosis else {
            continue;
        };
        if !request.classes.is_empty() && !request.classes.contains(&result.class) {
            continue;
        }
        let tally = by_class
            .entry(result.class.as_str().to_string())
            .or_default();
        tally.failed += 1;
        let experiences = admitted.get(&trial.trial).unwrap_or(&trial.experiences);
        let why = match (route(result), experiences.as_slice(), trial.exported) {
            (Route::Skip(why), _, _) => Some(why.to_string()),
            (_, _, false) => Some("the export skipped the trial: no trajectory".to_string()),
            (Route::ForkCorrection, [student], true) => {
                let key = key_of(student, &teacher_identity, intervention.name())?;
                if let Some(record) = held
                    .get(&key)
                    .filter(|r| r.outcome != RepairOutcome::Refused)
                {
                    tally.held += 1;
                    repairs.push(record.clone());
                    continue;
                }
                tally.routed += 1;
                cases.push((trial, result, student.clone(), key));
                None
            }
            (Route::ForkCorrection, _, true) => {
                Some("the trial is not one experience of the store".to_string())
            }
        };
        if let Some(why) = why {
            skipped.push((trial.trial.clone(), why));
        }
    }
    cases.truncate(request.max);
    let run_of: BTreeMap<PathBuf, &Path> = request
        .runs
        .iter()
        .map(|run| (run.join("export"), run.as_path()))
        .collect();
    let next = Mutex::new(cases.into_iter());
    let done: Mutex<Vec<(RepairRecord, Option<Experience>)>> = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..request.jobs.max(1) {
            scope.spawn(|| loop {
                if request.cancel.is_cancelled() {
                    return;
                }
                let Some((trial, result, student_id, key)) =
                    next.lock().ok().and_then(|mut n| n.next())
                else {
                    return;
                };
                let Some(run) = run_of.get(trial.export.as_path()) else {
                    continue;
                };
                let made = attempt_case(
                    ctx,
                    intervention,
                    trial,
                    result,
                    run,
                    &student_id,
                    &key,
                    &teacher_identity,
                );
                if let Ok(mut done) = done.lock() {
                    done.push(made);
                }
            });
        }
    });
    if request.cancel.is_cancelled() {
        return Err(OrchestratorError::Cancelled);
    }
    let mut accepted = Vec::new();
    let mut made = done.into_inner().unwrap_or_default();
    made.sort_by(|a, b| a.0.trial.cmp(&b.0.trial));
    for (mut record, experience) in made {
        if let Some(experience) = experience {
            let id = keep_accepted(ctx, &record, &experience)?;
            record.experience = Some(id.clone());
            accepted.push(id);
        }
        annotate_failure(ctx, &record)?;
        ctx.workspace().put_document(RECORD, &record)?;
        let tally = by_class
            .entry(record.diagnosis.class.as_str().to_string())
            .or_default();
        match record.outcome {
            RepairOutcome::Accepted => tally.accepted += 1,
            RepairOutcome::Rejected => tally.rejected += 1,
            RepairOutcome::Refused => tally.refused += 1,
        }
        repairs.push(record);
    }
    let repair_set = if accepted.is_empty() {
        None
    } else {
        Some(store.put_set(&ExperienceSet {
            name: format!("repairs continued by {teacher_identity}"),
            members: accepted,
        })?)
    };
    Ok(RepairReport {
        format: REPAIR_FORMAT,
        teacher: teacher_identity,
        by_class,
        repairs,
        skipped,
        repair_set,
        diagnosis,
    })
}

/// Runs the intervention on one case and makes its record.
#[allow(clippy::too_many_arguments)]
fn attempt_case(
    ctx: &Context,
    intervention: &(dyn Intervention + '_),
    trial: &DiagnosedTrial,
    result: &DiagnosticResult,
    run: &Path,
    student_id: &ExperienceId,
    key: &str,
    teacher: &str,
) -> (RepairRecord, Option<Experience>) {
    let mut record = RepairRecord {
        format: REPAIR_FORMAT.into(),
        key: key.into(),
        student: student_id.clone(),
        trial: trial.trial.clone(),
        task: trial.task.clone(),
        domain: trial.domain.clone(),
        diagnosis: result.clone(),
        intervention: intervention.name().into(),
        keep_through: result.last_good_step().unwrap_or_default(),
        teacher: teacher.into(),
        outcome: RepairOutcome::Refused,
        reason: None,
        grade: None,
        experience: None,
        prefix_steps: None,
        teacher_actions: None,
    };
    let student = match ctx.experiences().get(student_id) {
        Ok(student) => student,
        Err(e) => {
            record.reason = Some(format!("the student's experience: {e}"));
            return (record, None);
        }
    };
    let case = Case {
        trial,
        diagnosis: result,
        run,
        student: &student,
        student_id,
    };
    match intervention.attempt(ctx, &case) {
        Err(reason) => {
            record.reason = Some(reason);
            (record, None)
        }
        Ok(attempted) => {
            record.keep_through = attempted.keep_through;
            record.prefix_steps = Some(attempted.prefix_steps);
            record.teacher_actions = Some(attempted.teacher_actions);
            let passed = attempted.grade.passed();
            record.outcome = if passed {
                RepairOutcome::Accepted
            } else {
                RepairOutcome::Rejected
            };
            record.grade = Some(attempted.grade);
            (record, passed.then_some(attempted.experience))
        }
    }
}

/// Stores an accepted repair with the verifier's pass and its relation to
/// the student's attempt.
fn keep_accepted(
    ctx: &Context,
    record: &RepairRecord,
    experience: &Experience,
) -> Result<ExperienceId, OrchestratorError> {
    let store = ctx.experiences();
    let id = store.put(experience)?;
    let verifier = store
        .annotations(&record.student)?
        .annotations
        .into_iter()
        .find(|a| a.producer.name == VERIFIER_PRODUCER)
        .map(|a| a.producer)
        .unwrap_or_else(|| Producer {
            name: VERIFIER_PRODUCER.into(),
            version: record
                .grade
                .as_ref()
                .map(|g| g.digest.clone())
                .unwrap_or_default(),
        });
    store.annotate(&Annotation {
        experience: id.clone(),
        producer: verifier,
        body: AnnotationBody::Verdict {
            outcome: Outcome::Pass,
            strength: Strength::Executable,
            evidence: serde_json::json!({ "grade": record.grade, "repair": record.key }),
        },
    })?;
    let producer = Producer {
        name: PRODUCER.into(),
        version: splinter_eval::diagnosis::RULES_VERSION.into(),
    };
    // `RevisionOf` and not `PreferredOver`: the preference view finds a
    // repair by its copied prefix and pairs it with the failed attempt that
    // prefix is, and reads `PreferredOver` as a pair of final answers, which an
    // agent's trial does not have.
    store.annotate(&Annotation {
        experience: id.clone(),
        producer,
        body: AnnotationBody::Relation {
            kind: RelationKind::RevisionOf,
            other: record.student.clone(),
        },
    })?;
    Ok(id)
}

/// Records the diagnosed class on the student's attempt, once.
fn annotate_failure(ctx: &Context, record: &RepairRecord) -> Result<(), OrchestratorError> {
    let store = ctx.experiences();
    let producer = Producer {
        name: splinter_eval::diagnosis::PRODUCER.into(),
        version: splinter_eval::diagnosis::RULES_VERSION.into(),
    };
    let annotated = store
        .annotations(&record.student)?
        .annotations
        .iter()
        .any(|a| a.producer == producer);
    if !annotated {
        store.annotate(&Annotation {
            experience: record.student.clone(),
            producer,
            body: AnnotationBody::AttemptReport(AttemptReport {
                failure: Some(record.diagnosis.class),
                ..AttemptReport::default()
            }),
        })?;
    }
    Ok(())
}
