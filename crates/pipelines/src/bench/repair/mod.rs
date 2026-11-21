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
//! ([`super::diagnose`]) and routed by their class ([`Routing`]): a model
//! failure whose decisive step the diagnosis placed goes to a chain of
//! interventions, tried in order until the verifier accepts one or the
//! failure's budget of attempts is spent; an infrastructure or task failure,
//! and a model failure with no decisive step, are not repaired, and the
//! report says why. Every intervention is an [`Intervention`], and all of
//! them fork the student's attempt at its last good step in the examiner's
//! session and are graded there:
//!
//! * [`ForkCorrection`]: a teacher continues;
//! * [`KnowledgeCorrection`]: a teacher continues with task-relevant public
//!   reference material in front of it, retrieved from a knowledge base that
//!   never holds sealed material, the retrieval recorded;
//! * [`Retry`]: the student's own model samples again under another seed;
//! * [`Decompose`]: a proposer (the teacher or the student's own model)
//!   splits the task into ordered subgoals with checkable intermediate
//!   states, and the student works them turn by turn, each subgoal's state
//!   checked and recorded.
//!
//! The hinted retry ([`hinted`], rationalisation) is of the same family but
//! runs compiled task folders as Harbor trials, not forks of practice runs.
//!
//! A fork correction opens the examiner's sandbox session for the task
//! (`bench task session`: the pristine workspace bound where a trial's agent
//! works, its account and image, the task's service), keeps the student's
//! steps through the last good one, has sven's fork replay them (each
//! replayed observation held to the recorded one) with every shell command run
//! in the session, and the teacher continue the same tool loop under the
//! profile the student ran. The session then grades the state the commands
//! left, with the composed trajectory, by the task's own verifier outside any
//! agent. Only a pass is accepted; a
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
//! Each repair is kept as soon as it is graded, so a stopped run keeps every
//! repair it finished.
//!
//! **Idempotent.** A repair is keyed by the student's experience, the model
//! that acts (the teacher, or the student's for a retry or a
//! decomposition), the intervention, the sample and what else makes the
//! intervention what it is (the knowledge's snapshot, the proposer), and
//! its record is stored under the state root; a
//! rerun finds a graded record (accepted or rejected) and does nothing. A
//! refused one (the fork did not replay, the teacher or the examiner failed)
//! decided nothing about the attempt and is tried again; it stores nothing. A
//! crash between storing an accepted experience and its record leaves an
//! experience no set names, and the rerun repairs the attempt again.

mod attempt;
mod commands;
mod compose;
mod decompose;
mod fork;
mod guided;
pub mod hinted;
mod keep;
mod knowledge;
mod retry;
mod route;
mod select;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use splinter_agent::CancelToken;
use splinter_core::diagnosis::DiagnosticResult;
use splinter_core::experience::{Experience, ExperienceId};
use splinter_core::model_ref::ModelRef;
use splinter_core::outcome::FailureClass;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::experiences::{ExperienceSet, SetId};

pub use commands::{Grade, GRADE_SCHEMA};
pub use decompose::{
    Check, Decompose, Decomposition, Proposer, Subgoal, SubgoalOutcome, SubgoalState, MAX_SUBGOALS,
};
pub use fork::ForkCorrection;
pub use guided::{fork_points, unguided, PathGuided, PrivilegedPaths};
pub use knowledge::{KnowledgeBase, KnowledgeCorrection, KnowledgeUse, KNOWLEDGE_PASSAGES};
pub use retry::Retry;
pub use route::{route, InterventionKind, Route, Routing, DEFAULT_BUDGET};
pub use select::{spread, Spread};

use attempt::attempt_case;
use keep::{held_records, is_zero, keep, key_of};

use super::diagnose::{diagnose_dirs, DiagnosedTrial, DiagnosisReport};
use super::practice::BenchCheckout;

/// The format tag of a [`RepairReport`] and of the records it keeps.
pub const REPAIR_FORMAT: &str = "splinter.bench-repair.v1";

/// The producer of the relations and reports a repair records.
pub const PRODUCER: &str = "splinter/repair";

/// The most failures one run repairs when the request does not say.
pub const DEFAULT_MAX_REPAIRS: usize = 32;

/// Failures repaired at once when the request does not say: enough to keep
/// a served teacher's decode slots busy.
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
    /// The student's own model, which a retry samples again and a
    /// decomposition's subgoals are worked by; `None`: neither runs.
    pub student: Option<&'a ModelRef>,
    /// Who proposes a decomposition's subgoals.
    pub proposer: Proposer,
    /// Public reference documents a knowledge-assisted correction retrieves
    /// from; empty: none runs.
    pub knowledge: &'a [PathBuf],
    /// The interventions the run may use; empty for every one it has the
    /// models and material for. Naming one it cannot run is refused.
    pub interventions: &'a [InterventionKind],
    /// Attempts one failure may cost, every intervention and sample counted.
    pub budget: usize,
    /// The most failures this run repairs.
    pub max: usize,
    /// Failures repaired at once (each one's chain runs in order).
    pub jobs: usize,
    /// The time one continuation may take.
    pub deadline: Duration,
    /// The classes repaired; empty for every class [`route`] sends to an
    /// intervention.
    pub classes: &'a [FailureClass],
    /// The most repairs of one task, counting the ones already graded;
    /// `None` for no bound.
    pub per_task: Option<usize>,
    /// The seed that orders the tasks and the attempts of each task
    /// ([`spread`]).
    pub seed: u64,
    /// Samples of each intervention of a failure's chain, each its own
    /// repair (keyed by its sample index): a model samples, and one sample
    /// of a hard task is often wrong.
    pub samples: usize,
    /// Stops the run between repairs.
    pub cancel: CancelToken,
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
    /// Which of the teacher's samples of the failure this is, from 0.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub sample: usize,
    /// The last student step kept, in the exported trajectory's numbering.
    pub keep_through: u64,
    /// The model that acted: the teacher, or the student's own for a retry
    /// or a decomposition.
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
    /// The acting model's agent steps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teacher_actions: Option<u32>,
    /// What the intervention records besides: a retry's seed, a
    /// decomposition's subgoals and their outcomes, the knowledge retrieved.
    #[serde(flatten)]
    pub notes: Notes,
}

/// What an intervention records about how it made its attempt, in the
/// record and in the repaired experience's lineage.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Notes {
    /// The sampling a retry asked for: the practice run's, under its own
    /// seed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<serde_json::Value>,
    /// A decomposition: the subgoals, who proposed them, and how each
    /// ended.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decomposition: Option<Decomposition>,
    /// The knowledge a correction was shown, and where it came from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub knowledge: Option<KnowledgeUse>,
    /// A path-guided resampling: the privileged path forked, after which of
    /// its steps, how many states it offered and how many turns of a person
    /// were taken out of it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub privileged_path: Option<serde_json::Value>,
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
    /// The routing table the run followed.
    pub routing: Routing,
    /// The interventions the run had.
    pub interventions: Vec<InterventionKind>,
    /// Per intervention: attempts made (`routed`) and how they ended.
    pub by_intervention: BTreeMap<String, ClassYield>,
    /// The seed that ordered the selection.
    pub seed: u64,
    /// The bound on repairs of one task, when there was one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_task: Option<usize>,
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
    /// The repair's key, which names its working directory.
    pub key: &'a str,
    /// Which sample of its intervention the repair is, from 0.
    pub sample: usize,
}

/// What an intervention made of a case: the grade, and the experience to
/// keep when the grade is a pass.
pub struct Attempted {
    /// What the intervention records besides.
    pub notes: Notes,
    /// The last student step kept.
    pub keep_through: u64,
    /// The verifier's grade.
    pub grade: Grade,
    /// The repaired attempt.
    pub experience: Experience,
    /// The leading steps shared with the student's stored trajectory.
    pub prefix_steps: usize,
    /// The acting model's agent steps.
    pub teacher_actions: u32,
}

/// A way of turning a failed attempt into a verified one.
pub trait Intervention: Sync {
    /// The name its records carry.
    fn name(&self) -> &'static str;

    /// The model that acts, as the record names it.
    fn actor(&self) -> &str;

    /// What else distinguishes its repairs of one attempt by one actor (the
    /// knowledge's snapshot, the proposer), part of the key; `None` for
    /// nothing.
    fn variant(&self) -> Option<String> {
        None
    }

    /// Whether it can repair a failure of `task` at all (a path-guided
    /// resampling needs a privileged path of the task); one that cannot is
    /// left out of the failure's chain and costs none of its budget.
    fn applies(&self, _task: &str) -> bool {
        true
    }

    /// Makes a repaired attempt of `case` and has it graded.
    ///
    /// # Errors
    /// Why no repaired attempt could be made or graded.
    fn attempt(&self, ctx: &Context, case: &Case<'_>) -> Result<Attempted, String>;
}

/// Diagnoses the failed trials of `request`'s runs, routes them, repairs the
/// routed ones with the interventions the request has the models and
/// material for, and keeps the accepted repairs; see the module
/// documentation.
///
/// # Errors
/// [`OrchestratorError`] when the store cannot be read or written, a model
/// cannot be resolved, the knowledge cannot be read (or names sealed
/// material), an intervention is named the request cannot run, or the run
/// is cancelled.
pub fn repair(
    ctx: &Context,
    request: &RepairRequest<'_>,
) -> Result<RepairReport, OrchestratorError> {
    let teacher = ctx.model(request.teacher)?;
    let student = request.student.map(|s| ctx.model(s)).transpose()?;
    let wanted = |kind| request.interventions.is_empty() || request.interventions.contains(&kind);
    let refuse = |kind: InterventionKind, why: &str| {
        Err(OrchestratorError::Refused(format!(
            "the intervention {kind} cannot run: {why}"
        )))
    };
    if request
        .interventions
        .contains(&InterventionKind::HintedRetry)
    {
        return refuse(
            InterventionKind::HintedRetry,
            "it works compiled task folders as Harbor trials, not practice runs: run \
             `experiences rationalise`",
        );
    }
    let fork = ForkCorrection {
        checkout: request.checkout,
        teacher: teacher.clone(),
        deadline: request.deadline,
        cancel: request.cancel.clone(),
    };
    let knowledge = if request.knowledge.is_empty() {
        if !request.interventions.is_empty() && wanted(InterventionKind::KnowledgeCorrection) {
            return refuse(
                InterventionKind::KnowledgeCorrection,
                "no knowledge was given",
            );
        }
        None
    } else {
        Some(KnowledgeBase::load(ctx, request.knowledge)?)
    };
    let knowledge_correction = knowledge.as_ref().map(|base| KnowledgeCorrection {
        fork: &fork,
        knowledge: base,
    });
    if student.is_none() && !request.interventions.is_empty() {
        for kind in [
            InterventionKind::Retry,
            InterventionKind::Decompose,
            InterventionKind::PathGuided,
        ] {
            if wanted(kind) {
                return refuse(kind, "no student model was given");
            }
        }
    }
    let retry = student.as_ref().map(|student| Retry {
        checkout: request.checkout,
        student: student.clone(),
        deadline: request.deadline,
        cancel: request.cancel.clone(),
    });
    let paths = if student.is_some() && wanted(InterventionKind::PathGuided) {
        PrivilegedPaths::load(ctx)?
    } else {
        PrivilegedPaths::default()
    };
    let path_guided = student.as_ref().map(|student| PathGuided {
        checkout: request.checkout,
        student: student.clone(),
        paths: &paths,
        deadline: request.deadline,
        cancel: request.cancel.clone(),
    });
    let decompose = student.as_ref().map(|student| Decompose {
        checkout: request.checkout,
        student: student.clone(),
        proposer: match request.proposer {
            Proposer::Teacher => teacher.clone(),
            Proposer::Student => student.clone(),
        },
        proposer_role: request.proposer,
        deadline: request.deadline,
        cancel: request.cancel.clone(),
    });
    let made: [(InterventionKind, Option<&dyn Intervention>); 5] = [
        (
            InterventionKind::PathGuided,
            path_guided.as_ref().map(|p| p as &dyn Intervention),
        ),
        (InterventionKind::ForkCorrection, Some(&fork)),
        (
            InterventionKind::KnowledgeCorrection,
            knowledge_correction
                .as_ref()
                .map(|k| k as &dyn Intervention),
        ),
        (
            InterventionKind::Retry,
            retry.as_ref().map(|r| r as &dyn Intervention),
        ),
        (
            InterventionKind::Decompose,
            decompose.as_ref().map(|d| d as &dyn Intervention),
        ),
    ];
    let interventions: BTreeMap<InterventionKind, &dyn Intervention> = made
        .into_iter()
        .filter(|(kind, _)| wanted(*kind))
        .filter_map(|(kind, intervention)| Some((kind, intervention?)))
        .collect();
    let routing = Routing::standard(request.budget).map_err(OrchestratorError::Refused)?;
    repair_with(ctx, request, &routing, &interventions)
}

/// One failure the run repairs: its trial, diagnosis and student, and the
/// attempts its chain still owes, in order.
struct Pending<'a> {
    trial: &'a DiagnosedTrial,
    result: &'a DiagnosticResult,
    student: ExperienceId,
    steps: Vec<(InterventionKind, usize, String)>,
}

/// [`repair`] under `routing`, with `interventions` the ones available.
///
/// # Errors
/// As [`repair`].
pub fn repair_with(
    ctx: &Context,
    request: &RepairRequest<'_>,
    routing: &Routing,
    interventions: &BTreeMap<InterventionKind, &dyn Intervention>,
) -> Result<RepairReport, OrchestratorError> {
    let teacher_identity = ctx.model(request.teacher)?.identity;
    let available: BTreeSet<InterventionKind> = interventions.keys().copied().collect();
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
    let mut by_intervention: BTreeMap<String, ClassYield> = BTreeMap::new();
    let mut skipped = Vec::new();
    let mut repairs = Vec::new();
    let mut cases = Vec::new();
    let mut held_per_task: BTreeMap<&str, usize> = BTreeMap::new();
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
        let why = match (
            routing.route(result, &available),
            experiences.as_slice(),
            trial.exported,
        ) {
            (Route::Skip(why), _, _) => Some(why.to_string()),
            (_, _, false) => Some("the export skipped the trial: no trajectory".to_string()),
            (Route::To(chain), [student], true) => {
                // The failure's attempts in the order its chain tries them,
                // as many as its budget allows.
                let mut steps = Vec::new();
                for kind in chain {
                    if !interventions[&kind].applies(&trial.task) {
                        continue;
                    }
                    for sample in 0..request.samples.max(1) {
                        let key = key_of(student, interventions[&kind], sample)?;
                        steps.push((kind, sample, key));
                    }
                }
                steps.truncate(routing.budget);
                let graded: Vec<&RepairRecord> = steps
                    .iter()
                    .filter_map(|(_, _, key)| held.get(key))
                    .filter(|r| r.outcome != RepairOutcome::Refused)
                    .collect();
                if !graded.is_empty() {
                    tally.held += 1;
                    *held_per_task.entry(trial.task.as_str()).or_default() += 1;
                    repairs.extend(graded.iter().map(|r| (*r).clone()));
                }
                let repaired = graded.iter().any(|r| r.outcome == RepairOutcome::Accepted);
                steps.retain(|(_, _, key)| {
                    held.get(key)
                        .is_none_or(|r| r.outcome == RepairOutcome::Refused)
                });
                if repaired || steps.is_empty() {
                    continue;
                }
                cases.push(Pending {
                    trial,
                    result,
                    student: student.clone(),
                    steps,
                });
                None
            }
            (Route::To(_), _, true) => {
                Some("the trial is not one experience of the store".to_string())
            }
        };
        if let Some(why) = why {
            skipped.push((trial.trial.clone(), why));
        }
    }
    let (cases, unselected) = spread(
        cases,
        |case| case.trial.task.as_str(),
        |case| case.trial.trial.as_str(),
        &Spread {
            per_task: request.per_task,
            max: request.max,
            seed: request.seed,
            held: &held_per_task,
        },
    );
    for case in unselected {
        skipped.push((
            case.trial.trial.clone(),
            "not selected: the run's bound on repairs, or of one task, is reached".to_string(),
        ));
    }
    for case in &cases {
        by_class
            .entry(case.result.class.as_str().to_string())
            .or_default()
            .routed += 1;
    }
    let run_of: BTreeMap<PathBuf, &Path> = request
        .runs
        .iter()
        .map(|run| (run.join("export"), run.as_path()))
        .collect();
    let next = Mutex::new(cases.into_iter());
    // Each repair is kept as soon as it is graded, under one lock: a run that
    // is stopped keeps every repair it finished, and a rerun holds them.
    let done: Mutex<(Vec<RepairRecord>, Vec<ExperienceId>)> = Mutex::new((Vec::new(), Vec::new()));
    let failed: Mutex<Option<OrchestratorError>> = Mutex::new(None);
    let stopped = || request.cancel.is_cancelled() || failed.lock().is_ok_and(|f| f.is_some());
    std::thread::scope(|scope| {
        for _ in 0..request.jobs.max(1) {
            scope.spawn(|| loop {
                if stopped() {
                    return;
                }
                let Some(case) = next.lock().ok().and_then(|mut n| n.next()) else {
                    return;
                };
                let Some(run) = run_of.get(case.trial.export.as_path()) else {
                    continue;
                };
                // The chain in order: the first accepted attempt ends it.
                for (kind, sample, key) in &case.steps {
                    if stopped() {
                        return;
                    }
                    let (record, experience) = attempt_case(
                        ctx,
                        interventions[kind],
                        case.trial,
                        case.result,
                        run,
                        &case.student,
                        key,
                        *sample,
                    );
                    let accepted = record.outcome == RepairOutcome::Accepted;
                    let Ok(mut done) = done.lock() else {
                        return;
                    };
                    match keep(ctx, record, experience.as_ref()) {
                        Ok((record, id)) => {
                            done.0.push(record);
                            done.1.extend(id);
                        }
                        Err(error) => {
                            if let Ok(mut failed) = failed.lock() {
                                failed.get_or_insert(error);
                            }
                            return;
                        }
                    }
                    if accepted {
                        break;
                    }
                }
            });
        }
    });
    if let Some(error) = failed.into_inner().ok().flatten() {
        return Err(error);
    }
    if request.cancel.is_cancelled() {
        return Err(OrchestratorError::Cancelled);
    }
    let (mut made, accepted) = done.into_inner().unwrap_or_default();
    made.sort_by(|a, b| (&a.trial, &a.key).cmp(&(&b.trial, &b.key)));
    for record in made {
        for tally in [
            by_class
                .entry(record.diagnosis.class.as_str().to_string())
                .or_default(),
            by_intervention
                .entry(record.intervention.clone())
                .or_default(),
        ] {
            match record.outcome {
                RepairOutcome::Accepted => tally.accepted += 1,
                RepairOutcome::Rejected => tally.rejected += 1,
                RepairOutcome::Refused => tally.refused += 1,
            }
        }
        by_intervention
            .entry(record.intervention.clone())
            .or_default()
            .routed += 1;
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
        routing: routing.clone(),
        interventions: available.into_iter().collect(),
        by_intervention,
        seed: request.seed,
        per_task: request.per_task,
        by_class,
        repairs,
        skipped,
        repair_set,
        diagnosis,
    })
}
