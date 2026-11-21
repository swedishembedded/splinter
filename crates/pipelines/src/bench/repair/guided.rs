// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements on-policy learning from privileged
// demonstrations, for its clients. If your team needs expertise in turning a
// guided or taught solution into training data the student could have
// produced itself, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Path-guided resampling: the student continues, unhinted, from the states
//! of a verified path that a privileged source reached.
//!
//! A privileged path is a verified pass of the task that the student could
//! not have produced as it stands: a person typed into its run (a
//! human-guided trial, `extra.sven.origin: human` on their turns), or the run
//! names it as its privileged source (a teacher's verified trace, by set). Its tokens
//! were produced with that guidance in context, so they are never trained on
//! ([`splinter_data::Exclusion::HumanContext`]). Its *states* are: the path
//! with the person's turns taken out ([`unguided`]) is a sequence of actions
//! and observations the student can be put in, and from a state after one of
//! its actions ([`fork_points`]) the student's own model continues in the
//! task's sandbox with nothing of the guidance in its context. A continuation
//! the task's verifier passes is the student's own success from that state;
//! it is kept with the path's steps as copied context, never supervised, and
//! the student's steps after them, which were sampled in a context the
//! student will also see.
//!
//! The intervention ([`PathGuided`]) forks a failed attempt's task at the
//! path's states, the latest first: its `n`-th sample forks after the
//! `n+1`-th latest action (never after the last), so a failure's chain of
//! samples moves the fork earlier until the student succeeds alone (the chain
//! then stops) or the samples run out. A continuation that calls no tool is no
//! success of the student's: the path had done the work, so it is refused and
//! the chain goes on. Which path a task uses is [`PrivilegedPaths::for_task`]: the store's
//! verified guided passes of the same bench task, in id order.

use std::collections::BTreeMap;

use atif::{StepOrigin, Trajectory};
use serde_json::json;
use splinter_agent::fork_at::POLICY;
use splinter_agent::solve::{Model, SolveOptions};
use splinter_agent::CancelToken;
use splinter_core::experience::{Environment, Experience, ExperienceId, Provenance};
use splinter_data::is_human_turn;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;
use splinter_store::experiences::SetId;

use super::super::practice::BenchCheckout;
use super::fork::{finish, fork_and_grade, Continuation, Source};
use super::{Attempted, Case, Intervention, InterventionKind, Notes};

/// `path` without the turns a person typed, its steps numbered again from 1:
/// the states the path reached, with nothing of the guidance in them. A reply
/// that called no tool just before such a turn goes with it: it ended the
/// model's turn ("done"), and without the person's message the conversation
/// would have ended there, so it is no state to continue from (kept, it would
/// also fold into the next action as one model response).
#[must_use]
pub fn unguided(path: &Trajectory) -> Trajectory {
    let mut kept = path.clone();
    let ended_by_person: Vec<bool> = (0..path.steps.len())
        .map(|at| {
            let step = &path.steps[at];
            is_human_turn(step)
                || (step.source == StepOrigin::Agent
                    && step.tool_calls.as_ref().is_none_or(Vec::is_empty)
                    && path.steps.get(at + 1).is_some_and(is_human_turn))
        })
        .collect();
    let mut index = 0;
    kept.steps.retain(|_| {
        index += 1;
        !ended_by_person[index - 1]
    });
    for (index, step) in kept.steps.iter_mut().enumerate() {
        step.step_id = u64::try_from(index + 1).unwrap_or(u64::MAX);
    }
    kept
}

/// The steps of `path` (numbered as given) after which the student can be
/// put to continue, the latest first: every agent step that called a tool
/// but the path's last action. After its last action the path has done its
/// work: a student put there can only end it, which passes without one
/// action of its own to learn.
#[must_use]
pub fn fork_points(path: &Trajectory) -> Vec<u64> {
    let mut points: Vec<u64> = path
        .steps
        .iter()
        .filter(|s| s.source == StepOrigin::Agent)
        .filter(|s| s.tool_calls.as_ref().is_some_and(|c| !c.is_empty()))
        .map(|s| s.step_id)
        .collect();
    points.pop();
    points.reverse();
    points
}

/// The store's privileged paths, by bench task: verified passes a person
/// guided.
#[derive(Debug, Default)]
pub struct PrivilegedPaths {
    by_task: BTreeMap<String, Vec<(ExperienceId, Experience)>>,
}

impl PrivilegedPaths {
    /// Every verified (passed) experience of a bench task that is privileged:
    /// its trajectory holds a turn a person typed, or it is a member of one of
    /// `sets`, the passes a run names as its privileged source (a teacher's
    /// verified traces, say). A pass by another model is never one unnamed.
    ///
    /// # Errors
    /// When the store cannot be read.
    pub fn load(ctx: &Context, sets: &[SetId]) -> Result<Self, OrchestratorError> {
        let store = ctx.experiences();
        let mut named = std::collections::BTreeSet::new();
        for set in sets {
            named.extend(store.get_set(set)?.members);
        }
        let mut guided = Vec::new();
        for id in store.list()? {
            let experience = store.get(&id)?;
            if experience.environment.kind == Environment::BENCH
                && (named.contains(&id) || experience.trajectory.steps.iter().any(is_human_turn))
            {
                guided.push((id, experience));
            }
        }
        let ids: Vec<ExperienceId> = guided.iter().map(|(id, _)| id.clone()).collect();
        let decided = store.decisions(&ids)?;
        let mut by_task: BTreeMap<String, Vec<(ExperienceId, Experience)>> = BTreeMap::new();
        for (id, experience) in guided {
            if !decided.get(&id).is_some_and(|d| d.passed) {
                continue;
            }
            let Some(task) = experience
                .environment
                .spec
                .pointer("/task/id")
                .and_then(|t| t.as_str())
                .map(str::to_string)
            else {
                continue;
            };
            by_task.entry(task).or_default().push((id, experience));
        }
        Ok(Self { by_task })
    }

    /// Whether `task` has a path.
    #[must_use]
    pub fn has(&self, task: &str) -> bool {
        self.by_task.contains_key(task)
    }

    /// The path `task` forks on: the first of its paths by id.
    #[must_use]
    pub fn for_task(&self, task: &str) -> Option<&(ExperienceId, Experience)> {
        self.by_task.get(task)?.first()
    }
}

/// Path-guided resampling of a failed attempt's task: the student's own
/// model continues from a state of the task's privileged path.
pub struct PathGuided<'a> {
    /// The examiner that lays out and grades the workspace.
    pub checkout: &'a BenchCheckout,
    /// The student's own model.
    pub student: Model,
    /// The paths the forks start from.
    pub paths: &'a PrivilegedPaths,
    /// The time one continuation may take.
    pub deadline: std::time::Duration,
    /// Stops a continuation.
    pub cancel: CancelToken,
}

impl Intervention for PathGuided<'_> {
    fn name(&self) -> &'static str {
        InterventionKind::PathGuided.as_str()
    }

    fn actor(&self) -> &str {
        &self.student.identity
    }

    fn applies(&self, task: &str) -> bool {
        self.paths.has(task)
    }

    fn attempt(&self, ctx: &Context, case: &Case<'_>) -> Result<Attempted, String> {
        let (path_id, path) = self
            .paths
            .for_task(&case.trial.task)
            .ok_or("the task has no verified privileged path")?;
        let removed = path
            .trajectory
            .steps
            .iter()
            .filter(|s| is_human_turn(s))
            .count();
        let stored = unguided(&path.trajectory);
        let (system, exported) = split_system(&stored);
        let points = fork_points(&exported);
        let keep_through = *points.get(case.sample).ok_or_else(|| {
            format!(
                "the path has {} state(s) to fork at; sample {} has none",
                points.len(),
                case.sample
            )
        })?;
        // Sampled as the practice run was, under a seed of the repair's own.
        let (params, sampling) =
            super::retry::sampling(case.run, super::retry::retry_seed(case.key));
        let mut options = SolveOptions::new(self.deadline);
        options.cancel = Some(self.cancel.clone());
        let mut options = self.student.solving(options);
        options.request_sampling = Some(params);
        let forked = fork_and_grade(
            ctx,
            self.checkout,
            case,
            Some(&Source {
                stored: &stored,
                exported: &exported,
                system: system.as_deref(),
                solver: &path.provenance.solver,
                keep_through,
            }),
            Continuation {
                model: &self.student,
                role: POLICY,
                options,
                briefing: None,
                turns: &[],
                after_turn: &mut |_, _| true,
            },
        )?;
        // A pass without an action of the student's own is the path's work,
        // not the student's: no repair, and the chain moves the fork earlier.
        if !forked.acted() {
            return Err(format!(
                "the student took no action of its own after step {keep_through}: the path had \
                 done the task's work there"
            ));
        }
        let notes = Notes {
            sampling: Some(sampling),
            privileged_path: Some(json!({
                "path": path_id,
                "fork_after": keep_through,
                "states": points.len(),
                "removed_turns": removed,
            })),
            ..Notes::default()
        };
        let provenance = Provenance::new(self.student.identity.clone(), ctx.clock());
        finish(case, self.name(), forked, notes, provenance)
    }
}

/// The system prompt a stored trajectory leads with, and the trajectory as
/// an export holds it: without that step, numbered from 1.
fn split_system(stored: &Trajectory) -> (Option<String>, Trajectory) {
    let mut exported = stored.clone();
    let system = match exported.steps.first() {
        Some(step) if step.source == StepOrigin::System => {
            let text = step.message.as_text().map(str::to_string);
            exported.steps.remove(0);
            for step in &mut exported.steps {
                step.step_id -= 1;
            }
            text
        }
        _ => None,
    };
    (system, exported)
}

#[cfg(test)]
#[path = "guided_tests.rs"]
mod tests;
