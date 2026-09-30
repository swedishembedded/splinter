// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curricula that spend a learner's solver and
// training budget where it still learns, for its clients. If your team needs
// expertise in curriculum design or agent evaluation, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Frontier selection by pass@k.
//!
//! [`measure`] solves every task of a set k times with the policy (the
//! solve stage, [`PassAtK::k`] attempts each, sampling as
//! [`PassAtK::sampling`] says), grades every attempt with its task kind's
//! own verifiers (the verify stage, no judge), and hands the verified set
//! to [`select_frontier`], which tallies each task's graded attempts
//! ([`splinter_lab::frontier`]) and keeps the tasks strictly between never
//! and always solved. Every attempt stays in the store as an experience
//! with its verdicts: a frontier task's passing attempts are training
//! data, its failed ones feed critique and preference views.
//!
//! The measurement is recorded under
//! `<root>/curriculum/measurements/<hex>.json`, content-addressed: per task
//! its kind, concepts, attempts, graded attempts, passes and rate (absent,
//! never `0`, with no graded attempt), with the solver, the release the
//! policy was, k and the sampling.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use serde::Serialize;
use splinter_knowledge::concepts::{Concept, ConceptResolver};
use splinter_lab::frontier::{Distribution, FrontierClass, PassCount};
pub use splinter_policy::Sampling;
use splinter_policy::AGENT_SAMPLING;
use splinter_store::annotation::decide;
use splinter_store::digest::{canonical_json, Digest};
use splinter_store::experience::ExperienceId;
use splinter_store::experiences::{ExperienceSet, SetId};
use splinter_store::tasks::{TaskSet, TaskSetId};
use splinter_store::write_once;
use sven_sdk::CancelToken;

use crate::context::Context;
use crate::error::{io, CampaignError};
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
    /// Where the rate places it.
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

/// What frontier selection reports.
#[derive(Clone, Debug, Serialize)]
pub struct Frontier {
    /// The recorded measurement's address.
    pub measurement: Digest,
    /// Where it is recorded.
    pub path: PathBuf,
    /// The solver's identity.
    pub solver: String,
    /// The release a policy solver was.
    pub policy: Option<PolicyUsed>,
    /// Attempts per task.
    pub k: usize,
    /// How the solver sampled; `None`: as its provider does.
    pub sampling: Option<Sampling>,
    /// Tasks by class: always solved, never solved, on the frontier,
    /// unmeasured.
    pub distribution: Distribution,
    /// The frontier's tasks (`solve <frontier_task_set>`).
    pub frontier_task_set: TaskSetId,
    /// The frontier's attempts (`critique`, `dataset build`).
    pub frontier_experience_set: SetId,
}

/// Tallies the verified attempts `solved` recorded for `task_set`, keeps
/// the frontier, and records the measurement.
pub fn select_frontier(
    ctx: &Context,
    task_set: &TaskSetId,
    solved: &Solved,
) -> Result<Frontier, CampaignError> {
    let set = ctx.tasks().get_set(task_set)?;
    let store = ctx.experiences();
    let mut attempts: BTreeMap<Digest, (PassCount, Vec<ExperienceId>)> = BTreeMap::new();
    for id in store.get_set(&solved.experience_set)?.members {
        let experience = store.get(&id)?;
        let outcome = decide(&store.annotations(&id)?.annotations).map(|d| d.passed);
        let (count, ids) = attempts.entry(experience.task.id).or_default();
        count.record(outcome);
        ids.push(id);
    }
    let mut resolver = ConceptResolver::new(ctx.sources());
    let mut tasks = Vec::with_capacity(set.members.len());
    let mut distribution = Distribution::default();
    let mut frontier_tasks = Vec::new();
    let mut frontier_attempts = Vec::new();
    for entry in &set.members {
        let task = ctx.tasks().get(&entry.task)?;
        let (count, ids) = attempts.remove(&entry.task).unwrap_or_default();
        let class = count.class();
        distribution.add(class);
        if class == FrontierClass::Frontier {
            frontier_tasks.push(entry.clone());
            frontier_attempts.extend(ids);
        }
        tasks.push(TaskPassRate {
            concepts: resolver.concepts(&task)?,
            task: entry.task.clone(),
            kind: task.task.kind,
            attempts: count.attempts,
            graded: count.graded,
            passes: count.passes,
            rate: count.rate(),
            class,
        });
    }
    let frontier_task_set = ctx.tasks().put_set(&TaskSet {
        name: format!("frontier of {task_set} by pass@{}", solved.attempts),
        members: frontier_tasks,
    })?;
    let frontier_experience_set = store.put_set(&ExperienceSet {
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
        tasks,
        distribution,
        frontier_task_set: frontier_task_set.clone(),
        frontier_experience_set: frontier_experience_set.clone(),
        measured_at: ctx.clock().utc_now(),
    };
    let (id, path) = record(ctx, &measurement)?;
    Ok(Frontier {
        measurement: id,
        path,
        solver: measurement.solver,
        policy: measurement.policy,
        k: measurement.k,
        sampling: measurement.sampling,
        distribution,
        frontier_task_set,
        frontier_experience_set,
    })
}

/// Writes `measurement` once under its address.
fn record(ctx: &Context, measurement: &Measurement) -> Result<(Digest, PathBuf), CampaignError> {
    let bytes = canonical_json(measurement).map_err(|source| CampaignError::Json {
        what: "pass@k measurement".into(),
        source,
    })?;
    let id = Digest::of(&bytes);
    let path = ctx
        .root()
        .curriculum()
        .join("measurements")
        .join(format!("{}.json", id.hex()));
    write_once(&path, &bytes).map_err(io(&path))?;
    Ok((id, path))
}

/// One pass@k measurement's inputs and bounds.
pub struct MeasureRequest<'a> {
    /// The tasks.
    pub task_set: &'a TaskSetId,
    /// The policy measured.
    pub solver: &'a ModelRef,
    /// k and the sampling.
    pub pass_at_k: PassAtK,
    /// No solve starts after this, and none runs past it.
    pub deadline: Option<Instant>,
    /// Stops the measurement.
    pub cancel: CancelToken,
}

/// What `solve --frontier` reports: the solve, the grading, the frontier.
#[derive(Clone, Debug, Serialize)]
pub struct Measured {
    /// The k attempts per task.
    pub solve: Solved,
    /// Their verdicts.
    pub verify: Verified,
    /// The frontier kept.
    pub frontier: Frontier,
}

/// Solves each task of `request.task_set` k times, grades every attempt,
/// and keeps the frontier; see the module documentation.
pub fn measure(ctx: &Context, request: &MeasureRequest<'_>) -> Result<Measured, CampaignError> {
    request.pass_at_k.validate()?;
    let solve = solve_tasks(
        ctx,
        &SolveRequest {
            task_set: request.task_set,
            solver: request.solver,
            attempts: request.pass_at_k.k,
            sampling: request.pass_at_k.sampling_choice(),
            deadline: request.deadline,
            cancel: request.cancel.clone(),
        },
    )?;
    let verify = verify_set(ctx, &solve.experience_set, None, &request.cancel)?;
    let frontier = select_frontier(ctx, request.task_set, &solve)?;
    Ok(Measured {
        solve,
        verify,
        frontier,
    })
}
