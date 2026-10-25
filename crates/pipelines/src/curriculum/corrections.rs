// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that learn from their own failures by
// having a stronger teacher continue the state they actually reached, for its
// clients. If your team needs expertise in on-policy agent data then you can
// procure our services by sending an email to info@swedishembedded.com.

//! On-policy corrections: a teacher continues the attempt the student could
//! not finish, from the state it really reached.
//!
//! A demonstration from a clean start does not tell a student what to do
//! from the state its own actions lead it to; that is the covariate shift
//! supervised imitation has. This stage selects the attempts that failed or
//! got stuck, and for each one asks the teacher to continue the student's
//! own transcript: sven's fork replays the recorded actions into a fresh
//! copy of the task's snapshot, refusing unless every observation and the
//! workspace it reached are the recorded ones, and the teacher then takes
//! the next turn under the same system prompt, tools and step budget
//! ([`splinter_agent::correction`]).
//!
//! The continuation is graded exactly as a student attempt is: its patch is
//! applied to a clean copy of the snapshot and the project's hidden checks
//! run over it, with the exposure audit reading the whole trajectory. A
//! continuation that does not verify is refused, never softened into a
//! weaker label; only verified ones enter the correction set the dataset
//! builds from. The student's prefix rides along in the same ATIF
//! trajectory, marked as copied context, so the sft-step view supervises
//! the teacher's actions and only those.
//!
//! One correction per task, and at most `max` in a run: failure states
//! oversample the hardest tasks if every attempt is continued, and a task
//! learned twice from its own failure adds cost, not information.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use serde::Serialize;
use splinter_agent::correction::{continue_attempt, CorrectionError, CorrectionRequest};
use splinter_agent::solve::SolveOptions;
use splinter_agent::work::WorkProtocol;
use splinter_agent::CancelToken;
use splinter_core::annotation::{AnnotationBody, Outcome, Strength};
use splinter_core::digest::Digest;
use splinter_core::experience::{Environment, ExperienceId, Provenance, Task};
use splinter_core::model_ref::ModelRef;
use splinter_core::outcome::{AttemptReport, FailureClass};
use splinter_sandbox::ResolvedEnvironment;
use splinter_store::experiences::{ExperienceSet, SetId};

use crate::tasks::remaining;
use crate::verify::{verify_set, Grading};
use crate::workspace_solve::WorkspaceInputs;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// The most corrections one run admits when the request does not say.
pub const DEFAULT_MAX_CORRECTIONS: usize = 8;

/// What one correction stage was asked.
pub struct CorrectionsRequest<'a> {
    /// The tasks the student attempted.
    pub task_set: &'a splinter_store::tasks::TaskSetId,
    /// The student's graded attempts; a failed or stuck one may be
    /// continued.
    pub attempts: &'a SetId,
    /// The model that continues them.
    pub teacher: &'a ModelRef,
    /// The most corrections admitted in this run.
    pub max: usize,
    /// The bounds an episode runs under, the student's own.
    pub work: WorkProtocol,
    /// No continuation starts after this, and none runs past it.
    pub deadline: Option<Instant>,
    /// Stops the stage.
    pub cancel: CancelToken,
}

/// One attempt continued, with the provenance of its actions.
#[derive(Clone, Debug, Serialize)]
pub struct CorrectionItem {
    /// The task.
    pub task: Digest,
    /// The student attempt that was continued.
    pub student: ExperienceId,
    /// The recorded continuation.
    pub experience: ExperienceId,
    /// The student's steps, context and never supervised.
    pub prefix_actions: u32,
    /// The teacher's steps, the training material.
    pub teacher_actions: u32,
    /// Whether the hidden checks passed and the continuation is admitted.
    pub admitted: bool,
}

/// One attempt that was not continued, and why.
#[derive(Clone, Debug, Serialize)]
pub struct CorrectionRefusal {
    /// The task.
    pub task: Digest,
    /// The student attempt.
    pub student: ExperienceId,
    /// Why it was refused.
    pub reason: String,
}

/// What the correction stage did.
#[derive(Clone, Debug, Serialize)]
pub struct Corrections {
    /// The model that continued the attempts.
    pub teacher: String,
    /// Every attempt continued, admitted or not.
    pub corrections: Vec<CorrectionItem>,
    /// The attempts that were selected but could not be continued.
    pub refused: Vec<CorrectionRefusal>,
    /// The most corrections this run allowed.
    pub max: usize,
    /// The verified corrections, which the dataset builds from.
    pub experience_set: SetId,
}

/// Why a selected attempt is one to continue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stuck {
    /// Graded and did not pass.
    Failed,
    /// The episode was stopped as going nowhere or out of calls.
    Stalled,
}

/// An attempt chosen for continuing: the task, the attempt, its report when
/// one was kept, and how it got stuck.
type StuckAttempt = (Digest, ExperienceId, Option<AttemptReport>, Stuck);

/// The failed or stuck attempt to continue for each task, at most `max`.
///
/// One attempt per task: the first in the set's order, and tasks in the
/// order of their content address, so the choice does not move with the
/// order a store happens to list its files in. An attempt the exposure
/// audit failed is never selected: replaying it would replay material the
/// solver must never have seen.
fn stuck_attempts(
    ctx: &Context,
    task_set: &splinter_store::tasks::TaskSetId,
    attempts: &SetId,
    max: usize,
) -> Result<Vec<StuckAttempt>, OrchestratorError> {
    // Only a task this run is learning is ever continued: a task of the exam
    // has no correction, whatever set an attempt was filed in.
    let allowed: BTreeSet<Digest> = ctx
        .tasks()
        .get_set(task_set)?
        .members
        .iter()
        .map(|entry| entry.task.clone())
        .collect();
    let store = ctx.experiences();
    let members = store.get_set(attempts)?.members;
    let decisions = store.decisions(&members)?;
    let mut by_task: BTreeMap<Digest, (ExperienceId, Option<AttemptReport>, Stuck)> =
        BTreeMap::new();
    for id in members {
        let notes = store.annotations(&id)?.annotations;
        let exposed = notes.iter().any(|note| {
            matches!(
                note.body,
                AnnotationBody::Verdict {
                    outcome: Outcome::Fail,
                    strength: Strength::Constraint,
                    ..
                }
            )
        });
        if exposed {
            continue;
        }
        let report = notes.iter().find_map(|note| match &note.body {
            AnnotationBody::AttemptReport(report) => Some(report.clone()),
            _ => None,
        });
        if report
            .as_ref()
            .is_some_and(|r| r.failure == Some(FailureClass::Environment))
        {
            // An environment failure says nothing about the student, so
            // there is no state of its own to correct.
            continue;
        }
        let stalled = report
            .as_ref()
            .is_some_and(|r| r.stopped_by_repetition || r.stopped_by_tool_cap);
        let stuck = if decisions.get(&id).is_some_and(|decision| !decision.passed) {
            Stuck::Failed
        } else if stalled {
            Stuck::Stalled
        } else {
            continue;
        };
        let experience = store.get(&id)?;
        if experience.environment.kind != Environment::WORKSPACE {
            continue;
        }
        let task = experience.task.id.clone();
        if !allowed.contains(&task) {
            continue;
        }
        by_task.entry(task).or_insert((id, report, stuck));
    }
    Ok(by_task
        .into_iter()
        .take(max)
        .map(|(task, (id, report, stuck))| (task, id, report, stuck))
        .collect())
}

/// Continues the failed or stuck attempts of `request` with its teacher,
/// verifies every continuation, and keeps the verified ones as the
/// correction set; see the module documentation.
pub fn correct(
    ctx: &Context,
    request: &CorrectionsRequest<'_>,
) -> Result<Corrections, OrchestratorError> {
    let store = ctx.experiences();
    let selected = stuck_attempts(ctx, request.task_set, request.attempts, request.max)?;
    let prompt = request.work.prompt.text();
    let task_wall = ctx.config().solve_deadline;
    let model = ctx.model(request.teacher)?.with_system(prompt);
    let mut corrections = Vec::new();
    let mut refused = Vec::new();
    let mut members = Vec::new();
    for (task_id, student, report, _stuck) in selected {
        if request.cancel.is_cancelled() {
            return Err(OrchestratorError::Cancelled);
        }
        if request
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            refused.push(CorrectionRefusal {
                task: task_id,
                student,
                reason: "the budget was spent before the attempt was continued".into(),
            });
            continue;
        }
        let task = ctx.tasks().get(&task_id)?;
        let experience = store.get(&student)?;
        let (environment, inputs) = match workspace_inputs(ctx, &task) {
            Ok(parts) => parts,
            Err(reason) => {
                refused.push(CorrectionRefusal {
                    task: task_id,
                    student,
                    reason,
                });
                continue;
            }
        };
        // What the student's episode had left, in tool-call rounds: the
        // teacher may take no more steps than the student could have.
        let rounds = request.work.max_tool_calls.map(|cap| {
            report
                .as_ref()
                .and_then(|report| report.tool_calls)
                .map_or(cap, |used| cap.saturating_sub(used))
                .max(1)
        });
        let continuation = CorrectionRequest {
            task: &task,
            environment: &environment,
            archive: inputs.archive(),
            student: &experience,
            system: prompt,
            remaining_rounds: rounds,
        };
        let mut options = SolveOptions::new(remaining(request.deadline, task_wall));
        options.cancel = Some(request.cancel.clone());
        // The episode's recording policy is the request's: a continuation that
        // compacted could not be projected into a record.
        options.compaction = request.work.compaction;
        match ctx.block_on(continue_attempt(
            &continuation,
            &model,
            model.solving(options),
        )) {
            Ok(correction) => {
                let provenance = Provenance {
                    prompt_digests: vec![Digest::of(prompt.as_bytes())],
                    attempt: None,
                    teacher: false,
                    ..Provenance::new(model.identity.clone(), ctx.clock())
                };
                let experience = correction
                    .solution
                    .into_experience(task.clone(), provenance)?;
                let id = store.put(&experience)?;
                corrections.push(CorrectionItem {
                    task: task_id,
                    student,
                    experience: id.clone(),
                    prefix_actions: correction.prefix_actions,
                    teacher_actions: correction.teacher_actions,
                    admitted: false,
                });
                members.push(id);
            }
            Err(error) => refused.push(CorrectionRefusal {
                task: task_id,
                student,
                reason: refusal(&error),
            }),
        }
    }
    let admitted = verify_corrections(ctx, request, members, &mut corrections)?;
    let experience_set = store.put_set(&ExperienceSet {
        name: format!("corrections continued by {}", model.identity),
        members: admitted,
    })?;
    Ok(Corrections {
        teacher: model.identity,
        corrections,
        refused,
        max: request.max,
        experience_set,
    })
}

/// Grades every recorded continuation with the task's own hidden checks and
/// the exposure audit, marks the admitted ones, and returns their ids.
fn verify_corrections(
    ctx: &Context,
    request: &CorrectionsRequest<'_>,
    members: Vec<ExperienceId>,
    corrections: &mut [CorrectionItem],
) -> Result<Vec<ExperienceId>, OrchestratorError> {
    if members.is_empty() {
        return Ok(members);
    }
    let store = ctx.experiences();
    let pending = store.put_set(&ExperienceSet {
        name: "corrections awaiting verification".into(),
        members: members.clone(),
    })?;
    verify_set(ctx, &pending, Grading::CodeOnly, &request.cancel)?;
    let decisions = store.decisions(&members)?;
    let mut admitted = Vec::new();
    for item in corrections.iter_mut() {
        let passed = decisions
            .get(&item.experience)
            .is_some_and(|decision| decision.passed);
        item.admitted = passed;
        if passed {
            admitted.push(item.experience.clone());
        }
    }
    Ok(admitted)
}

/// Why a continuation was not recorded.
fn refusal(error: &CorrectionError) -> String {
    match error {
        CorrectionError::Fork(error) => format!("the recorded transcript did not replay: {error}"),
        CorrectionError::Call(error) => format!("the teacher could not run: {error}"),
        CorrectionError::Sandbox(error) => format!("the environment failed: {error}"),
        CorrectionError::StudentPatch(error) => {
            format!("the student's patch does not reproduce its workspace: {error}")
        }
        CorrectionError::Trajectory(reason) => {
            format!("the student's trajectory cannot be forked: {reason}")
        }
        CorrectionError::Provenance { expected, got } => format!(
            "the continuation exported {got} steps for a {expected}-message transcript, so its \
             actions cannot be attributed"
        ),
        CorrectionError::NotWorkspace => "the task is not a change to a repository".into(),
    }
}

/// The workspace a task is solved in, and the archive and visible checks its
/// episode needs; the reason when it has none.
fn workspace_inputs(
    ctx: &Context,
    task: &Task,
) -> Result<(splinter_sandbox::WorkspaceEnvironment, WorkspaceInputs), String> {
    let environment = match ctx.environments().for_record(&task.environment) {
        Ok(ResolvedEnvironment::Workspace(environment)) => environment,
        Ok(_) => return Err("the task is not a change to a repository".into()),
        Err(error) => return Err(error.to_string()),
    };
    let inputs = WorkspaceInputs::of(ctx, &environment, task)?;
    Ok((environment, inputs))
}
