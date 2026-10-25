// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that learn tasks they cannot yet
// solve from verified demonstrations, for its clients. If your team needs
// expertise in curricula for code-editing agents you can procure our
// services by sending an email to info@swedishembedded.com.

//! The teacher of a change to a repository: a verified demonstration of a
//! task the student never solves.
//!
//! [`super::teacher`] shows a teacher the passage a task is grounded in. A
//! repository task has no passage, so on it `practice` used to drop every
//! task the policy never solved. Two teachers close that gap, and neither
//! is trusted:
//!
//! * [`RepositoryTeacher::Reference`] - the change that solves the task
//!   (the inverse of the seeded fault, kept with the task as its reference)
//!   replayed as the actions of an agent ([`splinter_agent::demonstration`]):
//!   a real episode in a fresh copy of the task's snapshot with sven's own
//!   tools, so the trajectory records what the tools returned.
//! * [`RepositoryTeacher::Model`] - a stronger model (`remote:` or
//!   `local:`) attempting the task as an agent, exactly as the student
//!   does, marked the teacher's.
//!
//! Either way the teacher only proposes. Its experience is graded by the
//! same workspace verifier as a student's, hidden tests laid over the
//! patch, and the same exposure audit reads its trajectory for anything
//! only the grader holds; an experience either fails is not kept
//! ([`super::frontier::select_frontier`] keeps a never-solved task only when
//! a teacher's answer was verified). The teacher is only asked about tasks
//! the student's graded attempts never solved, and `practice` asks it only
//! about tasks outside the exam, so no exam task is taught.

use std::collections::BTreeMap;
use std::str::FromStr;
use std::time::Instant;

use crate::solving::{ATTEMPT_PRODUCER, ATTEMPT_VERSION};
use serde::{Serialize, Serializer};
use splinter_agent::demonstration::{Demonstration, DEMONSTRATOR};
use splinter_agent::solve::Solution;
use splinter_agent::solve::{Model, SolveError, SolveOptions};
use splinter_agent::work::WorkProtocol;
use splinter_agent::CancelToken;
use splinter_core::annotation::{Annotation, AnnotationBody, Producer};
use splinter_core::digest::Digest;
use splinter_core::experience::{ExperienceId, PrivilegedKind, Provenance, Task};
use splinter_core::model_ref::{ModelRef, RefError};
use splinter_core::outcome::AttemptReport;
use splinter_sandbox::ResolvedEnvironment;
use splinter_store::experiences::{ExperienceSet, SetId};
use splinter_store::tasks::TaskEntry;
use splinter_store::tasks::{TaskSet, TaskSetId};

use super::teacher::{never_solved, Taught};
use crate::solving::{conclusion_name, solve_tasks, SamplingChoice, Skipped, SolveRequest, Solved};
use crate::tasks::remaining;
use crate::verify::{verify_set, Grading};
use crate::workspace_solve::{solve_attempt, WorkspaceInputs};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// Who proposes a verified answer to a repository task the student never
/// solves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RepositoryTeacher {
    /// The task's own reference change, replayed as an agent's actions.
    Reference,
    /// A stronger model, solving the task as an agent.
    Model(ModelRef),
}

/// What `--teacher` names for a repository run besides a model.
pub const REFERENCE: &str = "reference";

impl FromStr for RepositoryTeacher {
    type Err = RefError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        if text == REFERENCE {
            Ok(Self::Reference)
        } else {
            text.parse().map(Self::Model)
        }
    }
}

impl std::fmt::Display for RepositoryTeacher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reference => f.write_str(REFERENCE),
            Self::Model(model) => model.fmt(f),
        }
    }
}

impl Serialize for RepositoryTeacher {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

/// One repository teach stage's inputs and bounds.
pub struct TeachRepositoryRequest<'a> {
    /// The tasks the student attempted.
    pub task_set: &'a TaskSetId,
    /// The student's graded attempts at them; `None` when it made none, and
    /// every task is taught.
    pub attempts: Option<&'a SetId>,
    /// Who teaches.
    pub teacher: &'a RepositoryTeacher,
    /// How an episode is run: the bounds the student's attempts ran under.
    pub work: WorkProtocol,
    /// No solve starts after this, and none runs past it.
    pub deadline: Option<Instant>,
    /// Stops the stage.
    pub cancel: CancelToken,
}

/// Has the teacher demonstrate every task of `request.task_set` that the
/// student's graded attempts never solved, and grades each demonstration
/// with the workspace verifier and the exposure audit; see the module
/// documentation.
pub fn teach_repository(
    ctx: &Context,
    request: &TeachRepositoryRequest<'_>,
) -> Result<Taught, OrchestratorError> {
    let (members, name) = never_solved(ctx, request.task_set, request.attempts)?;
    crate::sealed::admit_set(ctx, request.task_set)?;
    let task_set = ctx.tasks().put_set(&TaskSet { name, members })?;
    let solve = match request.teacher {
        RepositoryTeacher::Reference => demonstrate(ctx, &task_set, request)?,
        RepositoryTeacher::Model(model) => solve_tasks(
            ctx,
            &SolveRequest {
                task_set: &task_set,
                solver: model,
                attempts: 1,
                sampling: SamplingChoice::Own,
                teacher: true,
                system: None,
                work: request.work,
                deadline: request.deadline,
                harness: None,
                resume: None,
                cancel: request.cancel.clone(),
            },
        )?,
    };
    // Code verifiers alone: the teacher is never graded by a model, nor by
    // itself.
    let verify = verify_set(
        ctx,
        &solve.experience_set,
        Grading::CodeOnly,
        &request.cancel,
    )?;
    Ok(Taught {
        task_set,
        solve,
        verify,
    })
}

/// Replays each task's reference change in an episode of its own.
fn demonstrate(
    ctx: &Context,
    task_set: &TaskSetId,
    request: &TeachRepositoryRequest<'_>,
) -> Result<Solved, OrchestratorError> {
    let store = ctx.experiences();
    let batch = ctx.workspace().batch();
    let task_wall = ctx.config().solve_deadline;
    let mut report = Solved {
        experience_set: SetId(Digest::of(b"")),
        solver: DEMONSTRATOR.into(),
        policy: None,
        attempts: 1,
        teacher: true,
        sampling: None,
        solved: 0,
        answered: 0,
        conclusions: BTreeMap::new(),
        skipped: Vec::new(),
        stopped: None,
    };
    let mut members = Vec::new();
    for entry in ctx.tasks().get_set(task_set)?.members {
        if request.cancel.is_cancelled() {
            return Err(OrchestratorError::Cancelled);
        }
        if request.deadline.is_some_and(|d| Instant::now() >= d) {
            report.stopped = Some("the budget was spent before every task was taught".into());
            break;
        }
        let task = ctx.tasks().get(&entry.task)?;
        let skip = |why: String| Skipped {
            task: entry.task.clone(),
            reason: why,
        };
        let Some(reference) = task
            .privileged
            .iter()
            .find(|p| p.kind == PrivilegedKind::Reference)
        else {
            report
                .skipped
                .push(skip("the task keeps no reference change to replay".into()));
            continue;
        };
        let demonstration = match Demonstration::from_patch(&reference.content) {
            Ok(demonstration) => demonstration,
            Err(e) => {
                report.skipped.push(skip(e.to_string()));
                continue;
            }
        };
        let environment = match ctx.environments().for_record(&task.environment) {
            Ok(ResolvedEnvironment::Workspace(env)) => env,
            Ok(_) => {
                report
                    .skipped
                    .push(skip("the task is not a change to a repository".into()));
                continue;
            }
            Err(e) => {
                report.skipped.push(skip(e.to_string()));
                continue;
            }
        };
        let inputs = match WorkspaceInputs::of(ctx, &environment, &task) {
            Ok(inputs) => inputs,
            Err(why) => {
                report.skipped.push(skip(why));
                continue;
            }
        };
        let model = Model::new(demonstration.provider(), DEMONSTRATOR);
        let mut options = SolveOptions::new(remaining(request.deadline, task_wall));
        options.cancel = Some(request.cancel.clone());
        let solved = ctx.block_on(solve_attempt(
            &task,
            &environment,
            &inputs,
            &model,
            model.solving(options),
            &request.work,
        ));
        let (solution, attempt_report) = match solved {
            Ok(solved) => solved,
            // An environment that is not what the task records is a task not
            // taught, as in any solve; any other failure stops the stage.
            Err(e @ (SolveError::EnvironmentMismatch { .. } | SolveError::Environment(_))) => {
                report.skipped.push(skip(e.to_string()));
                continue;
            }
            Err(e) => return Err(e.into()),
        };
        *report
            .conclusions
            .entry(conclusion_name(solution.conclusion))
            .or_default() += 1;
        report.answered += usize::from(solution.final_output.is_some());
        let id = keep_attempt(
            ctx,
            &Attempt {
                entry: &entry,
                task: &task,
                solver: DEMONSTRATOR,
                policy: None,
                teacher: true,
                attempt: None,
            },
            solution,
            attempt_report,
        )?;
        if !members.contains(&id) {
            members.push(id);
        }
    }
    report.solved = members.len();
    report.experience_set = store.put_set(&ExperienceSet {
        name: format!("{task_set} demonstrated from their reference changes"),
        members,
    })?;
    batch.commit()?;
    Ok(report)
}

/// One attempt at a task, to be kept as an experience.
struct Attempt<'a> {
    entry: &'a TaskEntry,
    task: &'a Task,
    /// The solver's identity, as the experience's provenance names it.
    solver: &'a str,
    /// The release label of a policy solver.
    policy: Option<&'a str>,
    /// Whether the solver was the teacher.
    teacher: bool,
    /// Which of several attempts this is, counted from zero.
    attempt: Option<u32>,
}

/// Keeps `solution` as an experience of the task, with the facts of the
/// attempt as an annotation; the experience's id.
fn keep_attempt(
    ctx: &Context,
    attempt: &Attempt<'_>,
    solution: Solution,
    report: AttemptReport,
) -> Result<ExperienceId, OrchestratorError> {
    let provenance = Provenance {
        generator: attempt.entry.generator.clone(),
        policy: attempt.policy.map(str::to_string),
        prompt_digests: attempt.entry.prompt.iter().cloned().collect(),
        attempt: attempt.attempt,
        teacher: attempt.teacher,
        ..Provenance::new(attempt.solver, ctx.clock())
    };
    let experience = solution.into_experience(attempt.task.clone(), provenance)?;
    let store = ctx.experiences();
    let id = store.put(&experience)?;
    store.annotate(&Annotation {
        experience: id.clone(),
        producer: Producer {
            name: ATTEMPT_PRODUCER.into(),
            version: ATTEMPT_VERSION.into(),
        },
        body: AnnotationBody::AttemptReport(report),
    })?;
    Ok(id)
}
