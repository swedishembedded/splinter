// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent solvers whose every run is replayable
// evidence, for its clients. If your team needs expertise in agent
// environments or learning from agent experience, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The solve stage: a task set becomes an experience set.
//!
//! Each task is solved in exactly the environment it records - closed-book,
//! its runtime in the process sandbox, or a copy of its workspace's
//! snapshot ([`crate::workspace_solve`]) - and refused (and reported) when
//! that environment is not what it was. Every solve is stored as an
//! experience, answered or not: a run stopped by its deadline is evidence
//! too. A task may be solved several times (a pass@k measurement): each
//! solve is its own experience, its provenance numbering the attempt. A
//! solve by a policy alias records, as the experience's policy, the release
//! the alias pointed at (or the base): what concept mastery is tallied by.
//!
//! A teacher's solve ([`SolveRequest::teacher`]) is the same solve,
//! open-book: the solver is prompted with the task's grounding material
//! ([`splinter_knowledge::material`]) before its instruction, and the
//! experience's provenance marks it the teacher's. The experience records
//! the task as it is - its own instruction, its own environment - so its
//! verdicts are the task's verifiers' and a view of it shows the student
//! the instruction alone. A task grounded in nothing a teacher could be
//! shown is skipped. A solve runs under the system prompt the request names
//! ([`SolveRequest::system`]) - the one the student is trained under - so
//! what is learned from was written under the prompt it is learned under.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use serde::Serialize;
use splinter_agent::converse::{converse_prompted, Dialogue};
use splinter_agent::solve::{
    open_book_prompt, solve_prompted, Model, Solution, SolveError, SolveOptions,
};
use splinter_agent::work::WorkProtocol;
use splinter_agent::{CancelToken, RunConclusion};
use splinter_core::annotation::{Annotation, AnnotationBody, Producer};
use splinter_core::digest::Digest;
use splinter_core::experience::ExperienceId;
use splinter_core::experience::{Provenance, Task};
use splinter_core::outcome::AttemptReport;
use splinter_eval::gate::AgenticConfig;
use splinter_knowledge::material::teacher_material;
use splinter_knowledge::tasks::Catalogue;
use splinter_model::Sampling;
use splinter_sandbox::ResolvedEnvironment;
use splinter_store::experiences::{ExperienceSet, SetId};
use splinter_store::tasks::{TaskEntry, TaskSetId};

use crate::curriculum::policy_label;
use crate::dialogue::{teacher_instruction, Student, DIALOGUE_TURNS};
use crate::learn::PolicyUsed;
use crate::tasks::remaining;
use crate::workspace_solve::{attempt_seed, solve_attempt, solve_harness_attempt, WorkspaceInputs};
use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::concurrency::fan_out_each;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// The producer name of the annotation that keeps an attempt's facts.
pub const ATTEMPT_PRODUCER: &str = "splinter-pipelines/attempt-report";

/// The version of that producer.
pub const ATTEMPT_VERSION: &str = "1";

pub use splinter_orchestrator::config::DEFAULT_SOLVE_DEADLINE;

/// A task left unsolved, and why.
#[derive(Clone, Debug, Serialize)]
pub struct Skipped {
    /// The task.
    pub task: Digest,
    /// Why it was not solved.
    pub reason: String,
}

/// What the solve stage reports.
#[derive(Clone, Debug, Serialize)]
pub struct Solved {
    /// The experience set (`verify <experience_set>`).
    pub experience_set: SetId,
    /// The solver's identity.
    pub solver: String,
    /// The release a policy solver was; `None` for a solver that is not a
    /// policy alias.
    pub policy: Option<PolicyUsed>,
    /// Solves per task.
    pub attempts: usize,
    /// Whether the solver was the teacher, shown each task's grounding
    /// material.
    pub teacher: bool,
    /// How the solver sampled; `None` when it sampled as its provider does.
    pub sampling: Option<Sampling>,
    /// Experiences recorded.
    pub solved: usize,
    /// Of those, how many gave a final answer.
    pub answered: usize,
    /// How the runs ended, by conclusion.
    pub conclusions: BTreeMap<String, usize>,
    /// Tasks not solved, with why.
    pub skipped: Vec<Skipped>,
    /// Why solving stopped before every task was tried, if it did.
    pub stopped: Option<String>,
}

/// How a solver samples its replies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SamplingChoice {
    /// As its provider does.
    Own,
    /// As given where the solver's sampling can be set here (a local model
    /// this context loaded), as its provider does elsewhere.
    Prefer(Sampling),
    /// As given; refused where the solver's sampling cannot be set.
    Require(Sampling),
}

/// One solve stage's inputs and bounds.
pub struct SolveRequest<'a> {
    /// The tasks.
    pub task_set: &'a TaskSetId,
    /// The model that solves them.
    pub solver: &'a ModelRef,
    /// Solves per task; at least one.
    pub attempts: usize,
    /// How the solver samples.
    pub sampling: SamplingChoice,
    /// Whether the solver is the teacher: prompted with each task's
    /// grounding material before its instruction, open-book; see the module
    /// documentation.
    pub teacher: bool,
    /// The system prompt the solver answers under, in place of its own: the
    /// one the student is trained under, so that a teacher's answers and a
    /// student's attempts are written as the person the records open with.
    /// `None` keeps the solver's.
    pub system: Option<&'a str>,
    /// How an episode in a workspace is run: the follow-up turns after red
    /// checks and the bounds on tool calls. Tasks in other environments
    /// ignore it.
    pub work: WorkProtocol,
    /// A sven harness document run in place of the conversational agent loop
    /// on a workspace task ([`splinter_agent::harness_work`]); `work` is not
    /// used then, the document has its own steps and bounds. Tasks in other
    /// environments ignore it.
    pub harness: Option<&'a str>,
    /// No solve starts after this, and none runs past it.
    pub deadline: Option<Instant>,
    /// Stops the stage.
    pub cancel: CancelToken,
    /// The experience set of an earlier solve of these tasks that did not
    /// finish (its last progress line names it): a task it already holds is
    /// not solved again, and its experiences are members of the new set.
    /// Needs one attempt per task.
    pub resume: Option<&'a SetId>,
}

/// The episode bounds every comparison runs under: the follow-ups and the
/// tool-call and repetition caps of [`AgenticConfig::default`]. A solve of a
/// whole set that took [`WorkProtocol::default`] instead ran with no
/// follow-up and no cap but the sandbox wall, which is not the harness the
/// arms of a comparison are probed in.
#[must_use]
pub fn harness_protocol() -> WorkProtocol {
    let config = AgenticConfig::default();
    WorkProtocol {
        follow_ups: config.follow_ups,
        max_tool_calls: config.max_tool_calls,
        max_repeated_results: config.max_repeated_results,
    }
}

/// Solves every task of `task_set` once with `solver`, as its provider
/// samples; no solve starts after `deadline` or runs past it. Workspace
/// tasks run under [`harness_protocol`]. With `resume`, the tasks that
/// earlier, unfinished set already answers are not solved again.
pub fn solve_set(
    ctx: &Context,
    task_set: &TaskSetId,
    solver: &ModelRef,
    deadline: Option<Instant>,
    cancel: &CancelToken,
    resume: Option<&SetId>,
) -> Result<Solved, OrchestratorError> {
    solve_tasks(
        ctx,
        &SolveRequest {
            task_set,
            solver,
            attempts: 1,
            sampling: SamplingChoice::Own,
            teacher: false,
            system: None,
            work: harness_protocol(),
            deadline,
            harness: None,
            cancel: cancel.clone(),
            resume,
        },
    )
}

/// The solver `request` names, sampling as it asks, and the sampling
/// applied (`None`: the provider's own).
fn sampled_solver(
    ctx: &Context,
    request: &SolveRequest<'_>,
) -> Result<(Model, Option<Sampling>), OrchestratorError> {
    let (sampling, required) = match request.sampling {
        SamplingChoice::Own => return Ok((ctx.model(request.solver)?, None)),
        SamplingChoice::Prefer(sampling) => (sampling, false),
        SamplingChoice::Require(sampling) => (sampling, true),
    };
    match ctx.resampled(request.solver, sampling)? {
        Some(model) => Ok((model, Some(sampling))),
        None if required => Err(OrchestratorError::Refused(format!(
            "{} samples as its provider does; its sampling cannot be set here",
            request.solver
        ))),
        None => Ok((ctx.model(request.solver)?, None)),
    }
}

/// Solves every task of `request.task_set` `request.attempts` times.
pub fn solve_tasks(ctx: &Context, request: &SolveRequest<'_>) -> Result<Solved, OrchestratorError> {
    if request.attempts == 0 {
        return Err(OrchestratorError::Refused(
            "a task is solved at least once".into(),
        ));
    }
    let set = ctx.tasks().get_set(request.task_set)?;
    let (mut model, sampling) = sampled_solver(ctx, request)?;
    if let Some(system) = request.system {
        model = model.with_system(system);
    }
    let policy = match request.solver {
        ModelRef::Policy(alias) => Some(PolicyUsed {
            alias: alias.clone(),
            release: ctx.policy_pin(alias)?.map(|pin| pin.release),
        }),
        _ => None,
    };
    let label = policy.as_ref().map(|p| policy_label(p.release.as_ref()));
    let attempts = u32::try_from(request.attempts)
        .map_err(|_| OrchestratorError::Refused("too many attempts per task".into()))?;
    let store = ctx.experiences();
    let batch = ctx.workspace().batch();
    let (mut members, already) = resumed(ctx, request)?;
    let mut report = Solved {
        experience_set: SetId(Digest::of(b"")),
        solver: model.identity.clone(),
        policy,
        attempts: request.attempts,
        teacher: request.teacher,
        sampling,
        solved: 0,
        answered: 0,
        conclusions: BTreeMap::new(),
        skipped: Vec::new(),
        stopped: None,
    };
    // Plan: what each task is asked, or why it is skipped. Reading the stores
    // and building the prompts is quick and stays in task order.
    let mut plans: Vec<Planned> = Vec::new();
    for entry in &set.members {
        if already.contains(&entry.task) {
            continue;
        }
        let task = ctx.tasks().get(&entry.task)?;
        let environment = match ctx.environments().for_record(&task.environment) {
            Ok(environment) => environment,
            Err(e) => {
                report.skipped.push(skip(&entry.task, &e));
                continue;
            }
        };
        let workspace = match &environment {
            ResolvedEnvironment::Workspace(env) => match WorkspaceInputs::of(ctx, env, &task) {
                Ok(inputs) => Some(inputs),
                Err(why) => {
                    report.skipped.push(skip(&entry.task, &why));
                    continue;
                }
            },
            _ => None,
        };
        let dialogue = request.teacher
            && Catalogue::builtin()
                .get(&task.task.kind)
                .is_some_and(|kind| kind.dialogue);
        let prompt = if request.teacher {
            let material = teacher_material(&ctx.sources(), &task)?;
            if material.is_empty() {
                report.skipped.push(skip(&entry.task, &NOTHING_TO_SHOW));
                continue;
            }
            let instruction = if dialogue {
                teacher_instruction(&task.instruction)
            } else {
                task.instruction.clone()
            };
            open_book_prompt(&instruction, &material)
        } else {
            task.instruction.clone()
        };
        plans.push(Planned {
            entry,
            task,
            environment,
            workspace,
            dialogue,
            prompt,
        });
    }

    // Execute: every attempt of every planned task, as many at once as the
    // model takes, the results back in task order. An attempt that begins
    // after a cancel or a spent budget does not run.
    let mut items: Vec<(usize, u32, Model)> = Vec::with_capacity(plans.len() * request.attempts);
    for (plan, planned) in plans.iter().enumerate() {
        for attempt in 0..attempts {
            let seeded = if planned.workspace.is_some() {
                seeded_model(ctx, request, &model, &planned.entry.task, attempt)?
            } else {
                model.clone()
            };
            items.push((plan, attempt, seeded));
        }
    }
    let task_wall = ctx.config().solve_deadline;
    let total = plans.len() * request.attempts;
    let mut finished = 0usize;
    let mut so_far = members.clone();
    let results = ctx.block_on(fan_out_each(
        items,
        ctx.concurrency_for(request.solver),
        |(plan, attempt, model)| {
            let (plans, store, label) = (&plans, &store, &label);
            async move {
                let outcome = attempt_once(&plans[plan], &model, request, task_wall).await;
                let recorded = record_attempt(
                    outcome,
                    &Recording {
                        store,
                        planned: &plans[plan],
                        model: &model,
                        label: label.as_ref(),
                        teacher: request.teacher,
                        attempt: (attempts > 1).then_some(attempt),
                        ctx,
                    },
                );
                (plan, attempt, recorded)
            }
        },
        |(_, _, recorded)| {
            // What is done is made durable while the rest runs, and a
            // partial set names it, so a solve that dies (a server, the
            // process) is resumed from its last progress line.
            finished += 1;
            if let Recorded::Done { id, .. } = recorded {
                so_far.push(id.clone());
            }
            if finished.is_multiple_of(PROGRESS_EVERY) || finished == total {
                match checkpoint(ctx, &so_far, &model.identity) {
                    Ok(set) => ctx.report_stage(
                        "solve-progress",
                        &serde_json::json!({ "finished": finished, "of": total, "partial_set": set }),
                    ),
                    Err(e) => ctx.report_stage(
                        "solve-progress",
                        &serde_json::json!({ "finished": finished, "of": total, "no_checkpoint": e.to_string() }),
                    ),
                }
            }
        },
    ));

    // Fold in order, as one attempt after another would have been: the first
    // cancel, spent budget or unusable environment decides what comes after.
    let mut unusable: Vec<usize> = Vec::new();
    for (plan, _attempt, recorded) in results {
        if unusable.contains(&plan) {
            continue;
        }
        let entry = plans[plan].entry;
        match recorded {
            Recorded::Cancelled => return Err(OrchestratorError::Cancelled),
            Recorded::BudgetSpent => {
                report.stopped = Some("the budget was spent before every task was tried".into());
                break;
            }
            Recorded::Unusable(e) => {
                report.skipped.push(skip(&entry.task, &e));
                unusable.push(plan);
            }
            Recorded::Infrastructure(e) => {
                // The attempt did not complete: not a failure of the solver,
                // and no experience. It is solved by a resume.
                report
                    .skipped
                    .push(skip(&entry.task, &format!("infrastructure error: {e}")));
            }
            Recorded::Outgrew(e) => {
                // Not infrastructure: a resume would be refused the same way.
                report.skipped.push(skip(
                    &entry.task,
                    &format!("episode outgrew the served context: {e}"),
                ));
            }
            Recorded::Failed(e) => return Err(e),
            Recorded::Done {
                id,
                conclusion,
                answered,
            } => {
                *report.conclusions.entry(conclusion).or_default() += 1;
                report.answered += usize::from(answered);
                if !members.contains(&id) {
                    members.push(id);
                }
            }
        }
    }
    report.solved = members.len();
    let times = if attempts > 1 {
        format!(" {attempts} times each")
    } else {
        String::new()
    };
    let by = if request.teacher {
        "taught open-book"
    } else {
        "solved"
    };
    report.experience_set = store.put_set(&ExperienceSet {
        name: format!("{} {by}{times} by {}", request.task_set, model.identity),
        members,
    })?;
    batch.commit()?;
    Ok(report)
}

/// `base` answering one attempt at one task, drawing from the seed that
/// attempt has ([`attempt_seed`]) where the request sets the solver's
/// sampling and the solver's sampling can be set here: the same attempt at
/// the same task decodes the same way again, and two attempts differ. The
/// solver as it is otherwise: a model reached over an API samples as its
/// server does.
fn seeded_model(
    ctx: &Context,
    request: &SolveRequest<'_>,
    base: &Model,
    task: &Digest,
    attempt: u32,
) -> Result<Model, OrchestratorError> {
    let (SamplingChoice::Prefer(sampling) | SamplingChoice::Require(sampling)) = request.sampling
    else {
        return Ok(base.clone());
    };
    let seeded = Sampling {
        seed: Some(attempt_seed(task, attempt)),
        ..sampling
    };
    Ok(match ctx.resampled(request.solver, seeded)? {
        Some(model) => match request.system {
            Some(system) => model.with_system(system),
            None => model,
        },
        None => base.clone(),
    })
}

/// A task ready to be asked.
struct Planned<'a> {
    entry: &'a TaskEntry,
    task: Task,
    environment: ResolvedEnvironment,
    /// What a solve in a workspace needs besides the task; `Some` exactly
    /// when `environment` is a workspace.
    workspace: Option<WorkspaceInputs>,
    dialogue: bool,
    prompt: String,
}

/// Attempts between two durable checkpoints of a solve in progress.
const PROGRESS_EVERY: usize = 8;

/// The members of the set to resume, and the tasks they answer.
fn resumed(
    ctx: &Context,
    request: &SolveRequest<'_>,
) -> Result<(Vec<ExperienceId>, BTreeSet<Digest>), OrchestratorError> {
    let Some(earlier) = request.resume else {
        return Ok((Vec::new(), BTreeSet::new()));
    };
    if request.attempts != 1 {
        return Err(OrchestratorError::Refused(
            "a solve is resumed with one attempt per task".into(),
        ));
    }
    let members = ctx.experiences().get_set(earlier)?.members;
    let mut tasks = BTreeSet::new();
    for id in &members {
        tasks.insert(ctx.experiences().get(id)?.task.id.clone());
    }
    Ok((members, tasks))
}

/// Makes everything recorded so far durable and stores the set of `members`.
fn checkpoint(
    ctx: &Context,
    members: &[ExperienceId],
    solver: &str,
) -> Result<SetId, OrchestratorError> {
    ctx.workspace().commit()?;
    let set = ctx.experiences().put_set(&ExperienceSet {
        name: format!("partial solve by {solver}"),
        members: members.to_vec(),
    })?;
    ctx.workspace().commit()?;
    Ok(set)
}

/// What one attempt came to once it was recorded.
enum Recorded {
    Cancelled,
    BudgetSpent,
    /// The task's environment is not the one offered.
    Unusable(SolveError),
    /// The solver could not be reached or failed to run: no outcome of its.
    Infrastructure(SolveError),
    /// The episode's prompt outgrew the context the server is built for: the
    /// server refuses the request, and asking again refuses it again.
    Outgrew(SolveError),
    /// The store refused the experience.
    Failed(OrchestratorError),
    Done {
        id: ExperienceId,
        /// How it ended, named as the report names it ([`conclusion_label`]).
        conclusion: String,
        answered: bool,
    },
}

/// Where and how an attempt's experience is stored.
struct Recording<'a> {
    ctx: &'a Context,
    store: &'a splinter_store::experiences::ExperienceStore,
    planned: &'a Planned<'a>,
    model: &'a Model,
    label: Option<&'a String>,
    teacher: bool,
    attempt: Option<u32>,
}

/// Whether the server refused the request because it needs more tokens than
/// the context it serves (brain's `context_length_exceeded`).
fn outgrew_context(error: &SolveError) -> bool {
    error.to_string().contains("context_length_exceeded")
}

/// Stores the experience of a finished attempt, with its report, the moment
/// it finishes.
fn record_attempt(outcome: Attempted, at: &Recording<'_>) -> Recorded {
    let solved = match outcome {
        Attempted::Cancelled => return Recorded::Cancelled,
        Attempted::BudgetSpent => return Recorded::BudgetSpent,
        Attempted::Solved(solved) => solved,
    };
    let (solution, attempt_report) = match solved {
        Ok(solved) => *solved,
        Err(e @ (SolveError::EnvironmentMismatch { .. } | SolveError::Environment(_))) => {
            return Recorded::Unusable(e)
        }
        Err(e) if outgrew_context(&e) => return Recorded::Outgrew(e),
        Err(e) => return Recorded::Infrastructure(e),
    };
    let (conclusion, answered) = (
        conclusion_label(solution.conclusion, &attempt_report),
        solution.final_output.is_some(),
    );
    let provenance = Provenance {
        generator: at.planned.entry.generator.clone(),
        policy: at.label.cloned(),
        prompt_digests: at.planned.entry.prompt.iter().cloned().collect(),
        attempt: at.attempt,
        teacher: at.teacher,
        ..Provenance::new(at.model.identity.clone(), at.ctx.clock())
    };
    let stored = solution
        .into_experience(at.planned.task.clone(), provenance)
        .map_err(OrchestratorError::from)
        .and_then(|experience| Ok(at.store.put(&experience)?))
        .and_then(|id| {
            at.store.annotate(&Annotation {
                experience: id.clone(),
                producer: Producer {
                    name: ATTEMPT_PRODUCER.into(),
                    version: ATTEMPT_VERSION.into(),
                },
                body: AnnotationBody::AttemptReport(attempt_report),
            })?;
            Ok(id)
        });
    match stored {
        Ok(id) => Recorded::Done {
            id,
            conclusion,
            answered,
        },
        Err(e) => Recorded::Failed(e),
    }
}

/// How one attempt went.
enum Attempted {
    /// A cancel was requested before it began.
    Cancelled,
    /// The budget was spent before it began.
    BudgetSpent,
    /// It ran.
    Solved(Result<Box<(Solution, AttemptReport)>, SolveError>),
}

/// One attempt at `planned`, unless a cancel or the budget forbids beginning.
async fn attempt_once(
    planned: &Planned<'_>,
    model: &Model,
    request: &SolveRequest<'_>,
    task_wall: Duration,
) -> Attempted {
    if request.cancel.is_cancelled() {
        return Attempted::Cancelled;
    }
    if request.deadline.is_some_and(|d| Instant::now() >= d) {
        return Attempted::BudgetSpent;
    }
    let started = Instant::now();
    let mut options = SolveOptions::new(remaining(request.deadline, task_wall));
    options.cancel = Some(request.cancel.clone());
    let options = model.solving(options);
    let Planned {
        entry,
        task,
        environment,
        workspace,
        dialogue,
        prompt,
    } = planned;
    if let (ResolvedEnvironment::Workspace(env), Some(inputs)) = (environment, workspace) {
        let solved = match request.harness {
            Some(harness) => {
                solve_harness_attempt(task, env, inputs, harness, model, options).await
            }
            None => solve_attempt(task, env, inputs, model, options, &request.work).await,
        };
        return Attempted::Solved(solved.map(Box::new));
    }
    let solved = if *dialogue {
        match Student::new(
            model,
            &entry.task,
            DIALOGUE_TURNS,
            &options,
            &request.cancel,
        ) {
            Ok(student) => {
                converse_prompted(
                    &Dialogue {
                        task,
                        opening: prompt,
                        environment,
                        interlocutor: &student,
                        turns: DIALOGUE_TURNS,
                    },
                    model.provider.clone(),
                    options,
                )
                .await
            }
            Err(e) => Err(e.into()),
        }
    } else {
        solve_prompted(task, prompt, environment, model.provider.clone(), options).await
    };
    // Only an episode in a workspace reports more than its duration.
    Attempted::Solved(solved.map(|solution| {
        let report = AttemptReport {
            wall_seconds: Some(started.elapsed().as_secs_f64()),
            ..AttemptReport::default()
        };
        Box::new((solution, report))
    }))
}

/// Why a teacher's solve of a task was skipped: it has nothing to show.
const NOTHING_TO_SHOW: &str =
    "nothing to show a teacher: the task has no evidence passage and no hint";

fn skip(task: &Digest, why: &dyn std::fmt::Display) -> Skipped {
    Skipped {
        task: task.clone(),
        reason: why.to_string(),
    }
}

/// How an attempt ended, as the solve report names it: the run's conclusion,
/// except that an episode splinter's own watch stopped (the same answer
/// repeated, or too many tool calls) is named for that bound. The agent runtime
/// reports every stop that came from a cancel token as `cancelled`, and these
/// bounds stop episodes by firing the token; the attempt report knows which
/// fired it.
pub(crate) fn conclusion_label(conclusion: RunConclusion, attempt: &AttemptReport) -> String {
    if attempt.stopped_by_repetition {
        "repetition_stopped".to_string()
    } else if attempt.stopped_by_tool_cap {
        "tool_cap_stopped".to_string()
    } else {
        conclusion_name(conclusion)
    }
}

/// How a run ended, as reports name it: `success`, `timeout`, ...
pub(crate) fn conclusion_name(conclusion: RunConclusion) -> String {
    let debug = format!("{conclusion:?}");
    let mut name = String::with_capacity(debug.len() + 4);
    for (i, c) in debug.chars().enumerate() {
        if c.is_ascii_uppercase() && i > 0 {
            name.push('_');
        }
        name.push(c.to_ascii_lowercase());
    }
    name
}

#[cfg(test)]
mod harness_tests {
    use super::{conclusion_label, harness_protocol};
    use splinter_agent::RunConclusion;
    use splinter_core::outcome::AttemptReport;

    /// An episode stopped by splinter's own bound is a stuck agent, not a
    /// cancelled run: the report must say which, or a campaign reads its own
    /// bounds as infrastructure noise (34 of 61 conclusions were once labelled
    /// "cancelled" for this reason).
    #[test]
    fn an_episode_stopped_by_its_own_bounds_is_named_for_the_bound_not_for_a_cancel() {
        let by = |tool_cap, repetition| AttemptReport {
            stopped_by_tool_cap: tool_cap,
            stopped_by_repetition: repetition,
            ..AttemptReport::default()
        };
        assert_eq!(
            conclusion_label(RunConclusion::Cancelled, &by(false, true)),
            "repetition_stopped"
        );
        assert_eq!(
            conclusion_label(RunConclusion::Cancelled, &by(true, false)),
            "tool_cap_stopped"
        );
        assert_eq!(
            conclusion_label(RunConclusion::Cancelled, &by(false, false)),
            "cancelled"
        );
        assert_eq!(
            conclusion_label(RunConclusion::Success, &by(false, false)),
            "success"
        );
        // A bound that fired after the agent had finished does not rename a
        // finished run's own conclusion... except that the bound is what
        // ended it, so the bound is named.
        assert_eq!(
            conclusion_label(RunConclusion::Cancelled, &by(true, true)),
            "repetition_stopped"
        );
    }

    #[test]
    fn a_set_is_solved_under_the_registered_episode_bounds() {
        let protocol = harness_protocol();
        assert_eq!(protocol.follow_ups, 2);
        assert_eq!(protocol.max_tool_calls, Some(60));
        assert_eq!(protocol.max_repeated_results, Some(3));
    }
}
