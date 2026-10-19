// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curricula that decide, from measured
// competence and measured GPU cost, which tasks a learner attempts first, for
// its clients. If your team needs expertise in budget-limited training of
// code-editing agents, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Which tasks are solved and taught first when the budget may not reach
//! them all.
//!
//! A run's solve and teach stages go through the admitted tasks in the order
//! of their task set and stop at the deadline. A [`Strategy`] decides that
//! order, and nothing else: the frontier (pass@k) classification that
//! follows keeps judging what was solved.
//!
//! * [`Strategy::Frontier`] keeps the set as it is: the order the tasks were
//!   admitted in.
//! * [`Strategy::LearningProgress`] draws the families, without replacement
//!   and in proportion to their learning progress per GPU second
//!   ([`super::progress`]), from a recorded seed.
//! * [`Strategy::GainRanked`] puts the families in order of that same signal,
//!   heaviest first, with no draw.
//!
//! A family's competence history is what [`super::mastery`] already reads
//! from the store: its verified pass rate under each release the policy was,
//! oldest first, with the device seconds those attempts took. A family no
//! release has attempted has no history and is ordered by its novelty bonus
//! alone, so on a repository none of whose files an earlier release met every
//! family looks alike and the order is the draw: the signal only
//! discriminates once a family has been measured under two releases, which
//! [`ScheduleReport::informative`] counts.
//!
//! Tasks of one family stay together in their set order; a task that declares
//! no family goes last.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use splinter_core::rng::Seed;
use splinter_eval::competence::outcome_of;
use splinter_knowledge::concepts::Concept;
use splinter_store::experiences::SetId;
use splinter_store::tasks::{TaskSet, TaskSetId};

use super::mastery::{mastery_costed, DEFAULT_MASTERY_WINDOW};
use super::progress::{self, History, Params, Round, Score};
use crate::family::family_of_task;
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// How the admitted tasks are ordered for the solve and teach stages.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Strategy {
    /// The order the tasks were admitted in.
    #[default]
    Frontier,
    /// Families drawn in proportion to learning progress per GPU second.
    LearningProgress,
    /// Families ranked by learning progress per GPU second.
    GainRanked,
}

impl Strategy {
    /// The name the command line and reports use.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Frontier => "frontier",
            Self::LearningProgress => "learning-progress",
            Self::GainRanked => "gain-ranked",
        }
    }
}

/// What a schedule did, for the run's report.
#[derive(Clone, Debug, Serialize)]
pub struct ScheduleReport {
    /// The strategy.
    pub strategy: Strategy,
    /// The parameters of the signal.
    pub params: Params,
    /// The seed of the draw; `None` for a strategy that draws nothing.
    pub seed: Option<Seed>,
    /// What each family's weight was made of.
    pub families: BTreeMap<String, Score>,
    /// Families measured under at least two releases: the ones whose
    /// progress is not zero by construction.
    pub informative: usize,
    /// The families in the order their tasks are attempted.
    pub order: Vec<String>,
}

/// The task set in its new order, and the report.
#[derive(Clone, Debug)]
pub struct Scheduled {
    /// The reordered set; the input set itself under
    /// [`Strategy::Frontier`].
    pub task_set: TaskSetId,
    /// What was decided.
    pub report: ScheduleReport,
}

/// Each declared family's competence history, from the store.
fn histories(ctx: &Context) -> Result<BTreeMap<String, History>, OrchestratorError> {
    let policy = ctx.policy_pin(splinter_core::model_ref::POLICY_DEFAULT)?;
    let current = policy.as_ref().map(|pin| &pin.release);
    let mut by_family = BTreeMap::new();
    for concept in mastery_costed(ctx, current, DEFAULT_MASTERY_WINDOW)? {
        let Concept::Declared { name } = concept.concept else {
            continue;
        };
        let rounds = concept
            .releases
            .iter()
            .map(|r| Round {
                attempts: u32::try_from(r.graded).unwrap_or(u32::MAX),
                passes: u32::try_from(r.passes).unwrap_or(u32::MAX),
                gpu_seconds: r.gpu_seconds,
            })
            .collect();
        by_family.insert(name, History { rounds });
    }
    Ok(by_family)
}

/// `tasks` grouped by family, families in `order`, each family's tasks in
/// their set order; tasks of no family last.
fn arranged<T>(tasks: Vec<(Option<String>, T)>, order: &[String]) -> Vec<T> {
    let position: BTreeMap<&String, usize> =
        order.iter().enumerate().map(|(i, f)| (f, i)).collect();
    let mut keyed: Vec<(usize, usize, T)> = tasks
        .into_iter()
        .enumerate()
        .map(|(at, (family, task))| {
            let rank = family
                .as_ref()
                .and_then(|f| position.get(f))
                .copied()
                .unwrap_or(usize::MAX);
            (rank, at, task)
        })
        .collect();
    keyed.sort_by_key(|(rank, at, _)| (*rank, *at));
    keyed.into_iter().map(|(_, _, task)| task).collect()
}

/// Orders `set` by `strategy`; `seed` is only used by
/// [`Strategy::LearningProgress`] and is required there.
pub fn schedule(
    ctx: &Context,
    set: &TaskSetId,
    strategy: Strategy,
    params: &Params,
    seed: Option<Seed>,
) -> Result<Scheduled, OrchestratorError> {
    params.validate()?;
    let members = ctx.tasks().get_set(set)?;
    let mut by_family = histories(ctx)?;
    let mut declared = Vec::with_capacity(members.members.len());
    for entry in members.members {
        let family = family_of_task(&ctx.tasks().get(&entry.task)?).map(str::to_string);
        if let Some(family) = &family {
            by_family.entry(family.clone()).or_default();
        }
        declared.push((family, entry));
    }
    let present: Vec<&String> = declared.iter().filter_map(|(f, _)| f.as_ref()).collect();
    by_family.retain(|family, _| present.contains(&family));
    let informative = by_family
        .values()
        .filter(|h| h.rounds.iter().filter(|r| r.attempts > 0).count() >= 2)
        .count();
    let scores = progress::weights(&by_family, params);
    let order = match (strategy, seed) {
        (Strategy::Frontier, _) => Vec::new(),
        (Strategy::GainRanked, _) => progress::ranked(&scores),
        (Strategy::LearningProgress, Some(seed)) => progress::order(&scores, seed),
        (Strategy::LearningProgress, None) => {
            return Err(OrchestratorError::Refused(
                "the learning-progress strategy draws its order from a seed".into(),
            ))
        }
    };
    let task_set = match strategy {
        Strategy::Frontier => set.clone(),
        _ => ctx.tasks().put_set(&TaskSet {
            name: format!("{set} in {} order", strategy.as_str()),
            members: arranged(declared, &order),
        })?,
    };
    Ok(Scheduled {
        task_set,
        report: ScheduleReport {
            strategy,
            params: *params,
            seed: (strategy == Strategy::LearningProgress)
                .then_some(seed)
                .flatten(),
            families: scores,
            informative,
            order,
        },
    })
}

/// What a set of attempts cost, and what the run kept for it: the yield a
/// strategy is compared by.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Yield {
    /// Device-busy seconds of the attempts that recorded them.
    pub gpu_seconds: f64,
    /// Attempts that recorded none: the total is a lower bound while this is
    /// not zero.
    pub unmeasured: usize,
    /// Tasks kept for training.
    pub kept: usize,
    /// `kept` per GPU hour; `None` when no attempt recorded device time.
    pub kept_per_gpu_hour: Option<f64>,
}

/// The device time of every attempt in `sets` (the student's and the
/// teacher's), against the `kept` tasks they produced.
pub fn yield_of(ctx: &Context, sets: &[&SetId], kept: usize) -> Result<Yield, OrchestratorError> {
    let store = ctx.experiences();
    let (mut gpu_seconds, mut unmeasured) = (0.0, 0);
    for set in sets {
        for id in store.get_set(set)?.members {
            let experience = store.get(&id)?;
            let notes = store.annotations(&id)?;
            let outcome = outcome_of(&experience, &notes.annotations)
                .map_err(|e| OrchestratorError::Refused(format!("experience {id}: {e}")))?;
            match outcome.gpu_seconds {
                Some(seconds) => gpu_seconds += seconds,
                None => unmeasured += 1,
            }
        }
    }
    Ok(Yield {
        gpu_seconds,
        unmeasured,
        kept,
        kept_per_gpu_hour: (gpu_seconds > 0.0).then(|| kept as f64 / (gpu_seconds / 3600.0)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn families_are_attempted_in_order_tasks_stay_together_and_unfamilied_ones_come_last() {
        let tasks = vec![
            (Some("b".to_string()), "b1"),
            (None, "loose"),
            (Some("a".to_string()), "a1"),
            (Some("b".to_string()), "b2"),
            (Some("a".to_string()), "a2"),
        ];
        let order = ["a".to_string(), "b".to_string()];
        assert_eq!(arranged(tasks, &order), ["a1", "a2", "b1", "b2", "loose"]);
    }
}
