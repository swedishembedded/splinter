// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements paired evaluations of coding agents that
// leave out what the agent was never to see and what the harness could not
// carry out, for its clients. If your team needs expertise in measuring a
// code-editing agent against another on the same repository tasks, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Probing a model on tasks that are changes to a repository.
//!
//! A task is attempted as the `practice` solve stage attempts it: by the
//! worker, in its own copy of the task's snapshot ([`solve_attempt`]), with
//! the follow-up turns and bounds the suite's [`AgenticConfig`] names -
//! the same for the champion, the candidate and the served candidate, so a
//! difference between them is the weights'. Every attempt draws from a seed
//! that is a function of the task and the attempt alone
//! ([`attempt_seed`]), where the arm's sampling can be set here.
//!
//! An attempt's patch is graded by the task's verifiers (the workspace
//! verifier, which applies it to a fresh copy and runs the project's own
//! checks with the hidden tests laid over, and the exposure audit). It has
//! one of three outcomes:
//!
//! * **graded**: right or wrong, by the verdict the store's decision rule
//!   reaches over the verifiers' verdicts;
//! * **exposed**: its trajectory shows the solver reached for what only the
//!   grader holds. It is no one's win or loss, whatever its patch did, and
//!   is counted;
//! * **failed**: the harness could not carry it out - the environment is not
//!   the one the task records, its snapshot is not kept, a verifier could
//!   not run or declined to decide. It says nothing about the model and is
//!   counted.
//!
//! Exposed and failed attempts are in no denominator. A task's verdict is
//! what a strict majority of its graded attempts say; with no majority, or no
//! graded attempt, it has none, and the task is unpaired.
//!
//! Attempts are never stored as experiences: nothing a probe produces can
//! reach a training set.

use splinter_agent::solve::{Model, SolveError, SolveOptions};
use splinter_agent::work::WorkProtocol;
use splinter_agent::CancelToken;
use splinter_core::digest::Digest;
use splinter_core::experience::{Experience, Provenance, Task};
use splinter_eval::gate::{AgenticConfig, AgenticProbe, Withheld};
use splinter_model::local::GREEDY_SAMPLING;
use splinter_model::Sampling;
use splinter_sandbox::ResolvedEnvironment;
use splinter_store::decision::decide;

use crate::release::probe::{Arm, Probe, Suite};
use crate::solving::DEFAULT_SOLVE_DEADLINE;
use crate::verify::{exposure_failure, verifiers_for};
use crate::workspace_solve::{attempt_seed, solve_attempt, WorkspaceInputs};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// How one attempt ended.
enum Outcome {
    /// Graded: whether its patch passed.
    Graded(bool),
    /// Its trajectory shows the solver saw what only the grader holds.
    Exposed,
    /// The harness could not carry it out or grade it.
    Failed,
}

/// One attempt: its patch, when the run produced one, and how it ended.
struct Attempt {
    patch: Option<String>,
    outcome: Outcome,
}

impl Attempt {
    fn failed() -> Self {
        Self {
            patch: None,
            outcome: Outcome::Failed,
        }
    }
}

/// `arm`'s probe of every task of `tasks`, in order, under `config`.
pub(crate) fn grade(
    ctx: &Context,
    arm: &Arm<'_>,
    tasks: &[&Task],
    config: &AgenticConfig,
    cancel: &CancelToken,
) -> Result<Vec<Probe>, OrchestratorError> {
    tasks
        .iter()
        .map(|task| probe_task(ctx, arm, task, config, cancel))
        .collect()
}

/// `task` attempted [`AgenticConfig::attempts`] times.
fn probe_task(
    ctx: &Context,
    arm: &Arm<'_>,
    task: &Task,
    config: &AgenticConfig,
    cancel: &CancelToken,
) -> Result<Probe, OrchestratorError> {
    let mut graded = Vec::new();
    let mut withheld = Withheld::default();
    let mut answer = None;
    for attempt in 0..config.attempts.max(1) {
        let made = attempt_at(ctx, arm, task, attempt, config, cancel)?;
        answer = answer.or(made.patch);
        match made.outcome {
            Outcome::Graded(passed) => graded.push(passed),
            Outcome::Exposed => withheld.exposed += 1,
            Outcome::Failed => withheld.environment_failures += 1,
        }
    }
    Ok(Probe {
        answer,
        verdict: majority(&graded),
        withheld,
    })
}

/// What a strict majority of `verdicts` say; `None` with no majority.
fn majority(verdicts: &[bool]) -> Option<bool> {
    let passed = verdicts.iter().filter(|passed| **passed).count();
    let failed = verdicts.len() - passed;
    if passed > failed {
        Some(true)
    } else if failed > passed {
        Some(false)
    } else {
        None
    }
}

/// Attempt `attempt` at `task`.
fn attempt_at(
    ctx: &Context,
    arm: &Arm<'_>,
    task: &Task,
    attempt: u32,
    config: &AgenticConfig,
    cancel: &CancelToken,
) -> Result<Attempt, OrchestratorError> {
    if cancel.is_cancelled() {
        return Err(OrchestratorError::Cancelled);
    }
    let Ok(ResolvedEnvironment::Workspace(env)) = ctx.environments().for_record(&task.environment)
    else {
        return Ok(Attempt::failed());
    };
    let Ok(inputs) = WorkspaceInputs::of(ctx, &env, task) else {
        return Ok(Attempt::failed());
    };
    let model = arm.attempt_model(ctx, &task.task.id, attempt)?;
    let mut options = SolveOptions::new(DEFAULT_SOLVE_DEADLINE);
    options.cancel = Some(cancel.clone());
    let options = model.solving(options);
    let protocol = WorkProtocol {
        follow_ups: config.follow_ups,
        max_tool_calls: config.max_tool_calls,
        max_repeated_results: config.max_repeated_results,
    };
    let solved = ctx.block_on(solve_attempt(
        task, &env, &inputs, &model, options, &protocol,
    ));
    if cancel.is_cancelled() {
        return Err(OrchestratorError::Cancelled);
    }
    let solution = match solved {
        Ok(solution) => solution,
        Err(SolveError::EnvironmentMismatch { .. } | SolveError::Environment(_)) => {
            return Ok(Attempt::failed());
        }
        Err(e) => return Err(e.into()),
    };
    let experience = solution.into_experience(
        task.clone(),
        Provenance::new(model.identity.clone(), ctx.clock()),
    )?;
    grade_attempt(ctx, task, experience)
}

/// `experience`, an attempt at `task`, graded by the task's verifiers.
fn grade_attempt(
    ctx: &Context,
    task: &Task,
    experience: Experience,
) -> Result<Attempt, OrchestratorError> {
    let patch = experience.final_output.clone();
    let made = |outcome| Attempt {
        patch: patch.clone(),
        outcome,
    };
    // A verifier that cannot be built (the snapshot or the hidden tests are
    // not kept) is the harness failing, not the model.
    let Ok(verifiers) = verifiers_for(ctx, task, &[], None) else {
        return Ok(made(Outcome::Failed));
    };
    let verification = verifiers.run(task, &experience)?;
    if !verification.errors.is_empty() {
        return Ok(made(Outcome::Failed));
    }
    if verification
        .annotations
        .iter()
        .any(|note| exposure_failure(note).is_some())
    {
        return Ok(made(Outcome::Exposed));
    }
    // The workspace verifier abstains where the task names an environment it
    // was not offered: no decision is the harness's.
    Ok(made(match decide(&verification.annotations) {
        Some(decision) => Outcome::Graded(decision.passed),
        None => Outcome::Failed,
    }))
}

impl Arm<'_> {
    /// The model for attempt `attempt` at `task`: drawing from that attempt's
    /// seed where the arm names the reference it is loaded from and its
    /// sampling can be set here, as it was handed in otherwise (a scripted
    /// model, or one reached over an API, whose server samples).
    fn attempt_model(
        &self,
        ctx: &Context,
        task: &Digest,
        attempt: u32,
    ) -> Result<Model, OrchestratorError> {
        let Some(reference) = self.reference else {
            return Ok(self.model.clone());
        };
        let seeded = Sampling {
            thinking: ctx.config().thinking,
            seed: Some(attempt_seed(task, attempt)),
            ..GREEDY_SAMPLING
        };
        Ok(match ctx.resampled(reference, seeded)? {
            Some(mut model) => {
                model.system.clone_from(&self.model.system);
                model
            }
            None => self.model.clone(),
        })
    }
}

/// What the probes of the workspace tasks of `suite` set aside, for each
/// arm; `None` when it holds none.
#[must_use]
pub(crate) fn withheld(
    suite: &Suite,
    candidate: &[Probe],
    champion: &[Probe],
) -> Option<AgenticProbe> {
    if suite.workspace_tasks() == 0 {
        return None;
    }
    let sum = |probes: &[Probe]| {
        suite
            .tasks
            .iter()
            .zip(probes)
            .filter(|(task, _)| is_workspace(task))
            .fold(Withheld::default(), |sum, (_, probe)| {
                sum.plus(probe.withheld)
            })
    };
    Some(AgenticProbe {
        attempts: suite.work.attempts.max(1),
        candidate: sum(candidate),
        champion: sum(champion),
    })
}

/// Whether `task` is a change to a repository.
pub(crate) fn is_workspace(task: &Task) -> bool {
    task.environment.kind == splinter_core::experience::Environment::WORKSPACE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_task_has_the_verdict_a_strict_majority_of_its_graded_attempts_give() {
        assert_eq!(majority(&[true]), Some(true));
        assert_eq!(majority(&[false, true, false]), Some(false));
        assert_eq!(majority(&[true, true, false]), Some(true));
        assert_eq!(majority(&[true, false]), None, "a tie decides nothing");
        assert_eq!(
            majority(&[]),
            None,
            "no graded attempt is no verdict, not a fail"
        );
    }
}
