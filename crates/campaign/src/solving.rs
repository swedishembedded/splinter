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
//! or its runtime in the process sandbox - and refused (and reported) when
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
//! shown is skipped.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use serde::Serialize;
use splinter_agent::converse::converse_prompted;
use splinter_agent::solve::{open_book_prompt, solve_prompted, Model, SolveError, SolveOptions};
use splinter_agent::{CancelToken, RunConclusion};
use splinter_core::digest::Digest;
use splinter_core::experience::Provenance;
use splinter_knowledge::material::teacher_material;
use splinter_knowledge::tasks::Catalogue;
use splinter_model::Sampling;
use splinter_store::experiences::{ExperienceSet, SetId};
use splinter_store::tasks::TaskSetId;

use crate::context::Context;
use crate::curriculum::policy_label;
use crate::dialogue::{teacher_instruction, Student, DIALOGUE_TURNS};
use crate::error::CampaignError;
use crate::learn::PolicyUsed;
use crate::model_ref::ModelRef;
use crate::tasks::remaining;

/// How long one task's solve may take.
pub const DEFAULT_SOLVE_DEADLINE: Duration = Duration::from_secs(300);

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
    /// No solve starts after this, and none runs past it.
    pub deadline: Option<Instant>,
    /// Stops the stage.
    pub cancel: CancelToken,
}

/// Solves every task of `task_set` once with `solver`, as its provider
/// samples; no solve starts after `deadline` or runs past it.
pub fn solve_set(
    ctx: &Context,
    task_set: &TaskSetId,
    solver: &ModelRef,
    deadline: Option<Instant>,
    cancel: &CancelToken,
) -> Result<Solved, CampaignError> {
    solve_tasks(
        ctx,
        &SolveRequest {
            task_set,
            solver,
            attempts: 1,
            sampling: SamplingChoice::Own,
            teacher: false,
            deadline,
            cancel: cancel.clone(),
        },
    )
}

/// The solver `request` names, sampling as it asks, and the sampling
/// applied (`None`: the provider's own).
fn sampled_solver(
    ctx: &Context,
    request: &SolveRequest<'_>,
) -> Result<(Model, Option<Sampling>), CampaignError> {
    let (sampling, required) = match request.sampling {
        SamplingChoice::Own => return Ok((ctx.model(request.solver)?, None)),
        SamplingChoice::Prefer(sampling) => (sampling, false),
        SamplingChoice::Require(sampling) => (sampling, true),
    };
    match ctx.resampled(request.solver, sampling)? {
        Some(model) => Ok((model, Some(sampling))),
        None if required => Err(CampaignError::Refused(format!(
            "{} samples as its provider does; its sampling cannot be set here",
            request.solver
        ))),
        None => Ok((ctx.model(request.solver)?, None)),
    }
}

/// Solves every task of `request.task_set` `request.attempts` times.
pub fn solve_tasks(ctx: &Context, request: &SolveRequest<'_>) -> Result<Solved, CampaignError> {
    if request.attempts == 0 {
        return Err(CampaignError::Refused(
            "a task is solved at least once".into(),
        ));
    }
    let set = ctx.tasks().get_set(request.task_set)?;
    let (model, sampling) = sampled_solver(ctx, request)?;
    let policy = match request.solver {
        ModelRef::Policy(alias) => Some(PolicyUsed {
            alias: alias.clone(),
            release: ctx.policy_pin(alias)?.map(|pin| pin.release),
        }),
        _ => None,
    };
    let label = policy.as_ref().map(|p| policy_label(p.release.as_ref()));
    let attempts = u32::try_from(request.attempts)
        .map_err(|_| CampaignError::Refused("too many attempts per task".into()))?;
    let store = ctx.experiences();
    let batch = ctx.workspace().batch();
    let mut members = Vec::new();
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
    'tasks: for entry in &set.members {
        let task = ctx.tasks().get(&entry.task)?;
        let environment = match ctx.environments().for_record(&task.environment) {
            Ok(environment) => environment,
            Err(e) => {
                report.skipped.push(skip(&entry.task, &e));
                continue;
            }
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
        for attempt in 0..attempts {
            if request.cancel.is_cancelled() {
                return Err(CampaignError::Cancelled);
            }
            if request.deadline.is_some_and(|d| Instant::now() >= d) {
                report.stopped = Some("the budget was spent before every task was tried".into());
                break 'tasks;
            }
            let mut options =
                SolveOptions::new(remaining(request.deadline, DEFAULT_SOLVE_DEADLINE));
            options.cancel = Some(request.cancel.clone());
            options.stream_idle = model.stream_idle;
            let solved = if dialogue {
                ctx.block_on(async {
                    let student = Student::new(
                        &model,
                        &entry.task,
                        DIALOGUE_TURNS,
                        &options,
                        &request.cancel,
                    )?;
                    converse_prompted(
                        &task,
                        &prompt,
                        &environment,
                        model.provider.clone(),
                        &student,
                        DIALOGUE_TURNS,
                        options,
                    )
                    .await
                })
            } else {
                ctx.block_on(solve_prompted(
                    &task,
                    &prompt,
                    &environment,
                    model.provider.clone(),
                    options,
                ))
            };
            let solution = match solved {
                Ok(solution) => solution,
                Err(e @ (SolveError::EnvironmentMismatch { .. } | SolveError::Environment(_))) => {
                    report.skipped.push(skip(&entry.task, &e));
                    continue 'tasks;
                }
                Err(e) => return Err(e.into()),
            };
            *report
                .conclusions
                .entry(conclusion_name(solution.conclusion))
                .or_default() += 1;
            report.answered += usize::from(solution.final_output.is_some());
            let provenance = Provenance {
                generator: entry.generator.clone(),
                policy: label.clone(),
                prompt_digests: entry.prompt.iter().cloned().collect(),
                attempt: (attempts > 1).then_some(attempt),
                teacher: request.teacher,
                ..Provenance::new(model.identity.clone(), ctx.clock())
            };
            let experience = solution.into_experience(task.clone(), provenance)?;
            let id = store.put(&experience)?;
            if !members.contains(&id) {
                members.push(id);
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

/// Why a teacher's solve of a task was skipped: it has nothing to show.
const NOTHING_TO_SHOW: &str =
    "nothing to show a teacher: the task has no evidence passage and no hint";

fn skip(task: &Digest, why: &dyn std::fmt::Display) -> Skipped {
    Skipped {
        task: task.clone(),
        reason: why.to_string(),
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
