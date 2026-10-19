// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the choice between retraining, a better
// harness, more search or no change at all, on measured cost-adjusted gain,
// for its clients. If your team needs expertise in deciding how an agent
// should be improved before spending a training budget, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The arms of an improvement experiment, measured by solving the target
//! tasks: the incumbent once, every arm against it.
//!
//! Every arm attempts the same tasks as the incumbent and is graded by the
//! same verifiers and the same exposure audit (the `solve` and `verify`
//! stages); an attempt the audit failed or no verifier decided is in no
//! denominator. A family is the unit of evidence: the arm's pass rate in it
//! minus the incumbent's.
//!
//! * a **protocol** arm solves with the incumbent under another episode
//!   protocol (follow-up turns, tool-call caps);
//! * a **samples** arm solves each task `samples` times and counts a task
//!   solved when any attempt is verified: pass@k, which in use needs
//!   something that picks the verified attempt (a visible check, a search),
//!   and the arm says so;
//! * a **candidate** arm solves with a model trained by an intervention (a
//!   teacher's demonstrations and training, for one), its one-time cost
//!   given with it.

use std::collections::BTreeMap;

use splinter_agent::CancelToken;
use splinter_core::digest::Digest;
use splinter_eval::competence::outcome_of;
use splinter_store::decision::decide;
use splinter_store::experiences::SetId;

use super::{ArmRunner, ImprovementExperiment, InterventionSpec, Measured};
use crate::solving::{solve_tasks, SolveRequest};
use crate::verify::{exposure_failure, verify_set, Grading};
use splinter_core::model_ref::ModelRef;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// What one task's graded attempts came to.
struct TaskResult {
    family: String,
    /// Graded attempts, each passed or failed.
    graded: Vec<bool>,
}

/// An arm's attempts: results by task, and the seconds they took.
struct Attempts {
    tasks: BTreeMap<Digest, TaskResult>,
    attempts: usize,
    /// Wall-clock seconds of every attempt; `None` when one did not say.
    wall_seconds: Option<f64>,
}

fn read(ctx: &Context, set: &SetId) -> Result<Attempts, OrchestratorError> {
    let store = ctx.experiences();
    let mut read = Attempts {
        tasks: BTreeMap::new(),
        attempts: 0,
        wall_seconds: Some(0.0),
    };
    for id in store.get_set(set)?.members {
        let experience = store.get(&id)?;
        let notes = store.annotations(&id)?.annotations;
        read.attempts += 1;
        let wall = outcome_of(&experience, &notes)?.wall_seconds;
        read.wall_seconds = read.wall_seconds.zip(wall).map(|(sum, wall)| sum + wall);
        let family = experience
            .concepts
            .first()
            .cloned()
            .unwrap_or_else(|| experience.task.id.to_string());
        let result = read
            .tasks
            .entry(experience.task.id.clone())
            .or_insert_with(|| TaskResult {
                family,
                graded: Vec::new(),
            });
        if notes.iter().any(|note| exposure_failure(note).is_some()) {
            continue;
        }
        if let Some(decision) = decide(&notes) {
            result.graded.push(decision.passed);
        }
    }
    Ok(read)
}

/// How a task counts as solved in an arm.
#[derive(Clone, Copy)]
enum Counting {
    /// The share of its graded attempts that passed.
    Rate,
    /// One when any graded attempt passed.
    Any,
}

impl Counting {
    fn of(self, graded: &[bool]) -> Option<f64> {
        if graded.is_empty() {
            return None;
        }
        let passes = graded.iter().filter(|p| **p).count();
        Some(match self {
            Self::Rate => passes as f64 / graded.len() as f64,
            Self::Any => f64::from(passes > 0),
        })
    }
}

/// Per family, the arm's pass rate minus the incumbent's, over the tasks
/// both have a graded attempt at. A family with no such task is absent.
fn family_deltas(incumbent: &Attempts, arm: &Attempts, counting: Counting) -> Vec<f64> {
    let mut families: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for (task, ours) in &arm.tasks {
        let Some(theirs) = incumbent.tasks.get(task) else {
            continue;
        };
        if let (Some(a), Some(i)) = (counting.of(&ours.graded), Counting::Rate.of(&theirs.graded)) {
            families.entry(&ours.family).or_default().push(a - i);
        }
    }
    families
        .into_values()
        .map(|d| d.iter().sum::<f64>() / d.len() as f64)
        .collect()
}

/// Seconds one attempt takes on average; `NaN` (which makes an arm
/// ineligible) when attempts did not record their time.
fn seconds_per_attempt(attempts: &Attempts) -> f64 {
    match (attempts.wall_seconds, attempts.attempts) {
        (Some(total), n) if n > 0 => total / n as f64,
        _ => f64::NAN,
    }
}

/// The runner that measures arms by solving the experiment's target tasks.
pub struct SolveArms<'a> {
    ctx: &'a Context,
    cancel: CancelToken,
    incumbent: Option<(SetId, Attempts)>,
}

impl<'a> SolveArms<'a> {
    /// A runner over `ctx`, stopped by `cancel`.
    #[must_use]
    pub fn new(ctx: &'a Context, cancel: CancelToken) -> Self {
        Self {
            ctx,
            cancel,
            incumbent: None,
        }
    }

    fn solve(
        &self,
        experiment: &ImprovementExperiment,
        solver: &ModelRef,
        attempts: usize,
        work: splinter_agent::work::WorkProtocol,
    ) -> Result<SetId, OrchestratorError> {
        let pass_at_k = experiment.evaluation.pass_at_k;
        let solved = solve_tasks(
            self.ctx,
            &SolveRequest {
                task_set: &experiment.target_tasks,
                solver,
                attempts,
                sampling: pass_at_k.sampling_choice(),
                teacher: false,
                system: None,
                work,
                deadline: None,
                harness: None,
                resume: None,
                cancel: self.cancel.clone(),
            },
        )?;
        verify_set(
            self.ctx,
            &solved.experience_set,
            Grading::CodeOnly,
            &self.cancel,
        )?;
        Ok(solved.experience_set)
    }

    /// The incumbent's attempts under the experiment's own protocol,
    /// solved once and kept.
    fn incumbent(
        &mut self,
        experiment: &ImprovementExperiment,
    ) -> Result<&(SetId, Attempts), OrchestratorError> {
        if self.incumbent.is_none() {
            let set = self.solve(
                experiment,
                &experiment.incumbent,
                experiment.evaluation.pass_at_k.k,
                experiment.evaluation.protocol.into(),
            )?;
            let attempts = read(self.ctx, &set)?;
            self.incumbent = Some((set, attempts));
        }
        self.incumbent
            .as_ref()
            .ok_or_else(|| OrchestratorError::Refused("the incumbent was not measured".into()))
    }
}

impl ArmRunner for SolveArms<'_> {
    fn measure(
        &mut self,
        experiment: &ImprovementExperiment,
        spec: &InterventionSpec,
    ) -> Result<Measured, OrchestratorError> {
        let k = experiment.evaluation.pass_at_k.k;
        let protocol = experiment.evaluation.protocol;
        let (solver, attempts, work, counting, one_time, samples, note) = match spec {
            InterventionSpec::NoChange => {
                let (_, incumbent) = self.incumbent(experiment)?;
                let units = family_deltas(incumbent, incumbent, Counting::Rate).len();
                return Ok(Measured {
                    deltas: vec![0.0; units],
                    one_time_cost: 0.0,
                    per_use_cost: 0.0,
                    experience_set: None,
                    note: None,
                });
            }
            InterventionSpec::Protocol { protocol, .. } => (
                experiment.incumbent.clone(),
                k,
                (*protocol).into(),
                Counting::Rate,
                0.0,
                1,
                None,
            ),
            InterventionSpec::MoreSamples { samples } => (
                experiment.incumbent.clone(),
                *samples,
                protocol.into(),
                Counting::Any,
                0.0,
                *samples,
                Some(
                    "pass@k: counts a task solved when any sample is verified, which in use needs \
                     a selector that picks the verified one"
                        .to_string(),
                ),
            ),
            InterventionSpec::Candidate {
                model,
                one_time_seconds,
                ..
            } => (
                model.clone(),
                k,
                protocol.into(),
                Counting::Rate,
                *one_time_seconds,
                1,
                None,
            ),
            InterventionSpec::Unavailable { reason, .. } => {
                return Err(OrchestratorError::Refused(format!(
                    "an intervention that is not available cannot be measured: {reason}"
                )))
            }
        };
        let set = self.solve(experiment, &solver, attempts, work)?;
        let arm = read(self.ctx, &set)?;
        let (_, incumbent) = self.incumbent(experiment)?;
        // The incumbent's attempt is one use; the arm's use is `samples` of
        // its attempts.
        let per_use = samples as f64 * seconds_per_attempt(&arm) - seconds_per_attempt(incumbent);
        Ok(Measured {
            deltas: family_deltas(incumbent, &arm, counting),
            one_time_cost: one_time,
            per_use_cost: per_use,
            experience_set: Some(set),
            note,
        })
    }
}
