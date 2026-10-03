// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curricula that spend a learner's solver and
// training budget where it still learns, for its clients. If your team needs
// expertise in curriculum design or agent evaluation, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Frontier selection by pass@k, and by a teacher's verified answer.
//!
//! [`measure`] solves every task of a set k times with the policy (the
//! solve stage, [`PassAtK::k`] attempts each, sampling as
//! [`PassAtK::sampling`] says) and grades every attempt with its task
//! kind's own verifiers (the verify stage, no judge); a teacher solves,
//! open-book, each task no graded attempt solved, and is graded the same
//! way ([`super::teacher`]); [`select_frontier`] tallies each task's
//! graded attempts and the teacher's ([`splinter_eval::frontier`]) and
//! keeps the tasks worth training on: those the student fails at least
//! sometimes and that have a verified answer - a passing attempt of its
//! own (the frontier proper) or a teacher's (taught). Every attempt stays
//! in the store as an experience with its verdicts: a kept task's passing
//! attempts and verified teacher answers are training data, its failed
//! ones feed critique and preference views.
//!
//! The measurement is recorded under
//! `<root>/curriculum/measurements/<hex>.json`, content-addressed: per task
//! its kind, concepts, attempts, graded attempts, passes and rate (absent,
//! never `0`, with no graded attempt), the teacher's attempts, graded
//! attempts and passes, and its class, with the solver, the teacher, the
//! release the policy was, k and the sampling.

use std::collections::BTreeMap;
use std::time::Instant;

use serde::Serialize;
use splinter_core::digest::Digest;
use splinter_core::experience::ExperienceId;
use splinter_eval::frontier::{Distribution, FrontierClass, PassCount};
use splinter_knowledge::concepts::{Concept, ConceptResolver};
pub use splinter_model::Sampling;
use splinter_model::AGENT_SAMPLING;
use splinter_store::experiences::{ExperienceSet, SetId};
use splinter_store::tasks::{TaskSet, TaskSetId};
use sven_sdk::CancelToken;

use crate::context::Context;
use crate::curriculum::teacher::{teach, Taught, TeachRequest};
use crate::error::CampaignError;
use crate::learn::PolicyUsed;
use crate::model_ref::ModelRef;
use crate::solving::{solve_tasks, SamplingChoice, SolveRequest, Solved};
use crate::verify::{verify_set, Verified};

/// Attempts per task when a command names none: enough for a task solved
/// about half the time to show both a pass and a fail most of the time,
/// few enough that measuring costs a handful of solves per task.
pub const DEFAULT_K: usize = 4;

/// How the policy samples its pass@k attempts when a command names
/// nothing: the agent sampling with a higher temperature, so the k attempts
/// can differ where the policy is unsure (at the agent's own low
/// temperature they are nearly one attempt repeated).
pub const DEFAULT_SAMPLING: Sampling = Sampling {
    temperature: 0.8,
    ..AGENT_SAMPLING
};

/// The parameters of a pass@k measurement.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct PassAtK {
    /// Attempts per task; at least two, or no task could be on the
    /// frontier.
    pub k: usize,
    /// How the policy samples. `None`: [`DEFAULT_SAMPLING`] where its
    /// sampling can be set (a local model loaded here), as its provider
    /// does elsewhere. `Some`: exactly this, refused where it cannot be
    /// set.
    pub sampling: Option<Sampling>,
}

impl Default for PassAtK {
    fn default() -> Self {
        Self {
            k: DEFAULT_K,
            sampling: None,
        }
    }
}

impl PassAtK {
    /// k attempts per task. With a `temperature` or a `top_k` given, the
    /// policy samples exactly so (the other from [`DEFAULT_SAMPLING`]),
    /// refused where its sampling cannot be set; with neither, as
    /// [`Self::sampling`] `None` says.
    #[must_use]
    pub fn new(k: usize, temperature: Option<f32>, top_k: Option<u32>) -> Self {
        let sampling = (temperature.is_some() || top_k.is_some()).then(|| Sampling {
            temperature: temperature.unwrap_or(DEFAULT_SAMPLING.temperature),
            top_k: top_k.unwrap_or(DEFAULT_SAMPLING.top_k),
            ..DEFAULT_SAMPLING
        });
        Self { k, sampling }
    }

    /// Refuses a k below two, and a negative or non-finite temperature.
    pub fn validate(&self) -> Result<(), CampaignError> {
        if self.k < 2 {
            return Err(CampaignError::Refused(format!(
                "pass@k needs k of at least 2 to find a task sometimes solved; {} given",
                self.k
            )));
        }
        if let Some(sampling) = &self.sampling {
            if !(sampling.temperature.is_finite() && sampling.temperature >= 0.0) {
                return Err(CampaignError::Refused(format!(
                    "a temperature is finite and at least 0; {} given",
                    sampling.temperature
                )));
            }
        }
        Ok(())
    }

    /// How the solve stage samples.
    #[must_use]
    pub fn sampling_choice(&self) -> SamplingChoice {
        match self.sampling {
            Some(sampling) => SamplingChoice::Require(sampling),
            None => SamplingChoice::Prefer(DEFAULT_SAMPLING),
        }
    }
}

/// One task's pass@k.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TaskPassRate {
    /// The task.
    pub task: Digest,
    /// Its kind.
    pub kind: String,
    /// The concepts it exercises.
    pub concepts: Vec<Concept>,
    /// Attempts recorded.
    pub attempts: usize,
    /// Of those, attempts graded pass or fail.
    pub graded: usize,
    /// Of those, attempts graded pass.
    pub passes: usize,
    /// `passes / graded`; `None` with no graded attempt.
    pub rate: Option<f64>,
    /// The teacher's solves of it: none unless no graded attempt passed.
    pub teacher: PassCount,
    /// Where the rate and the teacher's verified answer place it.
    pub class: FrontierClass,
}

/// The record a measurement leaves under the state root.
#[derive(Clone, Debug, Serialize)]
pub struct Measurement {
    /// The task set measured.
    pub task_set: TaskSetId,
    /// The solver's identity.
    pub solver: String,
    /// The release a policy solver was; `None` for another solver.
    pub policy: Option<PolicyUsed>,
    /// Attempts asked per task.
    pub k: usize,
    /// How the solver sampled; `None`: as its provider does.
    pub sampling: Option<Sampling>,
    /// Every attempt, graded.
    pub experience_set: SetId,
    /// The teacher's identity.
    pub teacher: String,
    /// Every teacher's solve, graded.
    pub teacher_experience_set: SetId,
    /// Each task of the set, in its order.
    pub tasks: Vec<TaskPassRate>,
    /// Tasks by class.
    pub distribution: Distribution,
    /// The frontier's tasks.
    pub frontier_task_set: TaskSetId,
    /// The frontier's attempts.
    pub frontier_experience_set: SetId,
    /// When it was measured, from the injected clock.
    pub measured_at: String,
}

const MEASUREMENT: &str = "frontier_measurement";

/// What frontier selection reports.
#[derive(Clone, Debug, Serialize)]
pub struct Frontier {
    /// The recorded measurement's address.
    pub measurement: Digest,
    /// The solver's identity.
    pub solver: String,
    /// The release a policy solver was.
    pub policy: Option<PolicyUsed>,
    /// Attempts per task.
    pub k: usize,
    /// How the solver sampled; `None`: as its provider does.
    pub sampling: Option<Sampling>,
    /// The teacher's identity.
    pub teacher: String,
    /// Tasks by class: always solved, on the frontier, taught, never
    /// solved with no verified answer, unmeasured.
    pub distribution: Distribution,
    /// The tasks kept: on the frontier, or taught (`solve
    /// <frontier_task_set>`).
    pub frontier_task_set: TaskSetId,
    /// Their attempts and verified teacher answers (`critique`, `dataset
    /// build`).
    pub frontier_experience_set: SetId,
}

/// Each task's graded solves in `set`: how many, how many graded, how many
/// passed, and the experiences.
pub(crate) fn tally(
    ctx: &Context,
    set: &SetId,
) -> Result<BTreeMap<Digest, (PassCount, Vec<ExperienceId>)>, CampaignError> {
    let store = ctx.experiences();
    let mut tallied: BTreeMap<Digest, (PassCount, Vec<ExperienceId>)> = BTreeMap::new();
    let members = store.get_set(set)?.members;
    let decisions = store.decisions(&members)?;
    for id in members {
        let experience = store.get(&id)?;
        let outcome = decisions.get(&id).map(|d| d.passed);
        let (count, ids) = tallied.entry(experience.task.id).or_default();
        count.record(outcome);
        ids.push(id);
    }
    Ok(tallied)
}

/// Tallies the verified attempts `solved` recorded for `task_set` and the
/// teacher's `taught`, keeps the tasks worth training on, and records the
/// measurement.
pub fn select_frontier(
    ctx: &Context,
    task_set: &TaskSetId,
    solved: &Solved,
    taught: &Taught,
) -> Result<Frontier, CampaignError> {
    let set = ctx.tasks().get_set(task_set)?;
    let mut attempts = tally(ctx, &solved.experience_set)?;
    let mut teacher = tally(ctx, &taught.solve.experience_set)?;
    let mut resolver = ConceptResolver::new(ctx.sources());
    let mut tasks = Vec::with_capacity(set.members.len());
    let mut distribution = Distribution::default();
    let mut frontier_tasks = Vec::new();
    let mut frontier_attempts = Vec::new();
    for entry in &set.members {
        let task = ctx.tasks().get(&entry.task)?;
        let (count, ids) = attempts.remove(&entry.task).unwrap_or_default();
        let (taught_count, taught_ids) = teacher.remove(&entry.task).unwrap_or_default();
        let class = count.class(taught_count.passes > 0);
        distribution.add(class);
        if class.kept() {
            frontier_tasks.push(entry.clone());
            frontier_attempts.extend(ids);
            frontier_attempts.extend(taught_ids);
        }
        tasks.push(TaskPassRate {
            concepts: resolver.concepts(&task)?,
            task: entry.task.clone(),
            kind: task.task.kind,
            attempts: count.attempts,
            graded: count.graded,
            passes: count.passes,
            rate: count.rate(),
            teacher: taught_count,
            class,
        });
    }
    let frontier_task_set = ctx.tasks().put_set(&TaskSet {
        name: format!("frontier of {task_set} by pass@{}", solved.attempts),
        members: frontier_tasks,
    })?;
    let frontier_experience_set = ctx.experiences().put_set(&ExperienceSet {
        name: format!(
            "frontier attempts of {task_set} by pass@{}",
            solved.attempts
        ),
        members: frontier_attempts,
    })?;
    let measurement = Measurement {
        task_set: task_set.clone(),
        solver: solved.solver.clone(),
        policy: solved.policy.clone(),
        k: solved.attempts,
        sampling: solved.sampling,
        experience_set: solved.experience_set.clone(),
        teacher: taught.solve.solver.clone(),
        teacher_experience_set: taught.solve.experience_set.clone(),
        tasks,
        distribution,
        frontier_task_set: frontier_task_set.clone(),
        frontier_experience_set: frontier_experience_set.clone(),
        measured_at: ctx.clock().utc_now(),
    };
    let id = record(ctx, &measurement)?;
    Ok(Frontier {
        measurement: id,
        solver: measurement.solver,
        policy: measurement.policy,
        k: measurement.k,
        sampling: measurement.sampling,
        teacher: measurement.teacher,
        distribution,
        frontier_task_set,
        frontier_experience_set,
    })
}

/// Records `measurement` once under its address.
fn record(ctx: &Context, measurement: &Measurement) -> Result<Digest, CampaignError> {
    Ok(ctx.workspace().put_document(MEASUREMENT, measurement)?)
}

/// One pass@k measurement's inputs and bounds.
pub struct MeasureRequest<'a> {
    /// The tasks.
    pub task_set: &'a TaskSetId,
    /// The policy measured.
    pub solver: &'a ModelRef,
    /// The model that teaches what the policy never solves; `None`: the
    /// policy itself.
    pub teacher: Option<&'a ModelRef>,
    /// k and the sampling.
    pub pass_at_k: PassAtK,
    /// No solve starts after this, and none runs past it.
    pub deadline: Option<Instant>,
    /// Stops the measurement.
    pub cancel: CancelToken,
}

/// What `solve --frontier` reports: the solve, the grading, the teacher,
/// the frontier.
#[derive(Clone, Debug, Serialize)]
pub struct Measured {
    /// The k attempts per task.
    pub solve: Solved,
    /// Their verdicts.
    pub verify: Verified,
    /// The teacher's graded solves of the tasks never solved.
    pub teach: Taught,
    /// The tasks kept.
    pub frontier: Frontier,
}

/// Solves each task of `request.task_set` k times, grades every attempt,
/// has the teacher solve each task never solved, and keeps the tasks worth
/// training on; see the module documentation.
pub fn measure(ctx: &Context, request: &MeasureRequest<'_>) -> Result<Measured, CampaignError> {
    request.pass_at_k.validate()?;
    let solve = solve_tasks(
        ctx,
        &SolveRequest {
            task_set: request.task_set,
            solver: request.solver,
            attempts: request.pass_at_k.k,
            sampling: request.pass_at_k.sampling_choice(),
            teacher: false,
            deadline: request.deadline,
            cancel: request.cancel.clone(),
        },
    )?;
    let verify = verify_set(ctx, &solve.experience_set, None, &request.cancel)?;
    let teach = teach(
        ctx,
        &TeachRequest {
            task_set: request.task_set,
            attempts: Some(&solve.experience_set),
            teacher: request.teacher.unwrap_or(request.solver),
            deadline: request.deadline,
            cancel: request.cancel.clone(),
        },
    )?;
    let frontier = select_frontier(ctx, request.task_set, &solve, &teach)?;
    Ok(Measured {
        solve,
        verify,
        teach,
        frontier,
    })
}
