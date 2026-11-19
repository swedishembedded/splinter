// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements curricula that spend a reinforcement-learning
// budget on the tasks that teach the most per token, for its clients. If your
// team needs expertise in curricula for policy-gradient training on agent
// rollouts, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The variance-per-cost curriculum: practice where a group-relative policy
//! gradient learns the most per token.
//!
//! Every earlier attempt at a practice task is rewarded as the campaign's
//! dataset rewards it (the strict pass, or `shaped-partial` above the task's
//! naive floor), and each task's attempts are measured
//! ([`splinter_eval::campaign::learning_signal::TaskSignal`]). A task is
//! drawn without replacement in proportion to `sqrt(V) / mean cost` (its
//! shrunk reward variance over the tokens an attempt costs) from a recorded
//! seed, up to a budget; a task never attempted weighs as the prior and the
//! mean cost say. Never drawn: a task whose two or more graded attempts all
//! earned the same (no advantage to learn from), a task the campaign retired
//! as mastered, and a task the infrastructure stopped every time. The same
//! seed and history draw the same tasks.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use splinter_core::rng::Seed;
use splinter_data::PartialCredit;
use splinter_eval::campaign::learning_signal::{
    fallback_cost, selection_weight, Attempt, TaskSignal,
};
use splinter_eval::campaign::pass_rate::{Band, PassRates};

use super::progress::weighted_order;
use crate::rl_build::RewardPreset;

/// One earlier attempt at a practice task.
#[derive(Clone, Debug, PartialEq)]
pub struct Scored {
    /// The task, by the name the practice material stages it under.
    pub task: String,
    /// Whether it passed, when graded; `None` when the infrastructure
    /// stopped it.
    pub passed: Option<bool>,
    /// The share of the task's weighted checks it passed, when measured.
    pub correctness: Option<f64>,
    /// Completion plus uncached prompt tokens, when counted.
    pub cost: Option<u64>,
    /// Completion tokens, when counted.
    pub completion: Option<u64>,
}

/// How the attempts are rewarded: as the campaign's dataset rewards them.
#[derive(Clone, Debug)]
pub struct Rewarding {
    /// The named reward.
    pub preset: RewardPreset,
    /// Under `shaped-partial`: the credit and each task's floor, by task name.
    pub credit: PartialCredit,
}

impl Rewarding {
    /// The reward of `attempt`; `None` for one the infrastructure stopped, or
    /// a failure `shaped-partial` cannot credit because it measured nothing.
    fn reward(&self, attempt: &Scored) -> Option<f64> {
        match (attempt.passed?, self.preset) {
            (true, _) => Some(1.0),
            (false, RewardPreset::Verdict) => Some(0.0),
            (false, RewardPreset::ShapedPartial) => Some(
                self.credit
                    .earned(attempt.correctness?, self.credit.floor(Some(&attempt.task))),
            ),
        }
    }
}

/// What the variance-per-cost selection decided, for the record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SignalSelection {
    /// The tasks to practise, in the order drawn.
    pub tasks: Vec<String>,
    /// Each candidate task's weight; a task of weight zero is never drawn.
    pub weights: BTreeMap<String, f64>,
    /// Tasks never drawn because their graded attempts all earned the same.
    pub uninformative: Vec<String>,
    /// Tasks left out because every attempt at them so far was stopped by
    /// the infrastructure.
    pub infrastructure: Vec<String>,
    /// The reward the attempts were measured under.
    pub reward: RewardPreset,
    /// The seed of the draw.
    pub seed: Seed,
}

/// What [`select`] chooses from and by.
#[derive(Clone, Copy, Debug)]
pub struct SignalRequest<'a> {
    /// The practice tasks, by name: the only tasks ever selected.
    pub pool: &'a [String],
    /// Every earlier attempt.
    pub history: &'a [Scored],
    /// Each task's family (its lineage), by name; a task absent is its own.
    pub families: &'a BTreeMap<String, String>,
    /// How the attempts are rewarded.
    pub rewarding: &'a Rewarding,
    /// The campaign's per-task pass rates: a retired task is never drawn.
    pub rates: &'a PassRates,
    /// The most tasks selected; `None` selects every task of positive weight.
    pub budget: Option<usize>,
    /// The seed of the draw.
    pub seed: Seed,
}

/// The tasks of `request.pool` drawn by variance per cost (module
/// documentation).
#[must_use]
pub fn select(request: &SignalRequest<'_>) -> SignalSelection {
    let mut attempts: BTreeMap<&str, Vec<Attempt>> = BTreeMap::new();
    let mut visited: BTreeMap<&str, bool> = BTreeMap::new();
    for scored in request.history {
        let graded = visited.entry(&scored.task).or_insert(false);
        *graded |= scored.passed.is_some();
        if let Some(reward) = request.rewarding.reward(scored) {
            attempts.entry(&scored.task).or_default().push(Attempt {
                reward,
                cost: scored.cost,
                completion: scored.completion,
            });
        }
    }
    let signals: Vec<TaskSignal> = request
        .pool
        .iter()
        .map(|task| {
            let family = request.families.get(task).unwrap_or(task);
            TaskSignal::of(
                task,
                family,
                attempts.get(task.as_str()).map_or(&[][..], Vec::as_slice),
            )
        })
        .collect();
    let fallback = fallback_cost(&signals);
    let mut weights = BTreeMap::new();
    let mut uninformative = Vec::new();
    let mut infrastructure = Vec::new();
    for signal in &signals {
        let task = &signal.task;
        if visited.get(task.as_str()) == Some(&false) {
            infrastructure.push(task.clone());
            continue;
        }
        if signal.measured() && !signal.informative {
            uninformative.push(task.clone());
        }
        let weight = if request.rates.band(task) == Band::Retired {
            0.0
        } else {
            selection_weight(signal, fallback)
        };
        weights.insert(task.clone(), weight);
    }
    let mut tasks = weighted_order(
        weights
            .iter()
            .filter(|(_, w)| **w > 0.0)
            .map(|(task, w)| (task, *w)),
        request.seed,
    );
    if let Some(budget) = request.budget {
        tasks.truncate(budget);
    }
    SignalSelection {
        tasks,
        weights,
        uninformative,
        infrastructure,
        reward: request.rewarding.preset,
        seed: request.seed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{BuildHasher, RandomState};

    /// A seed drawn for this call: the specs hold for every seed.
    fn any_seed() -> u64 {
        RandomState::new().hash_one(0_u8)
    }

    fn scored(task: &str, passed: bool, cost: u64) -> Scored {
        Scored {
            task: task.into(),
            passed: Some(passed),
            correctness: Some(if passed { 1.0 } else { 0.5 }),
            cost: Some(cost),
            completion: Some(cost / 10),
        }
    }

    fn rewarding(preset: RewardPreset) -> Rewarding {
        Rewarding {
            preset,
            credit: PartialCredit {
                metric: "functional_correctness".into(),
                credit: 0.5,
                floors: BTreeMap::new(),
                sources: Vec::new(),
                groups_without_floor: 0,
            },
        }
    }

    #[test]
    fn flat_groups_are_never_drawn_and_a_cheap_spread_task_outweighs_a_dear_one() {
        let pool: Vec<String> = ["flat", "cheap", "dear", "new"].map(String::from).to_vec();
        let mut history = Vec::new();
        for pass in [true, true, true, true] {
            history.push(scored("flat", pass, 100));
        }
        for pass in [true, false, true, false] {
            history.push(scored("cheap", pass, 100));
            history.push(scored("dear", pass, 1000));
        }
        let rewarding = rewarding(RewardPreset::Verdict);
        let selection = select(&SignalRequest {
            pool: &pool,
            history: &history,
            families: &BTreeMap::new(),
            rewarding: &rewarding,
            rates: &PassRates::default(),
            budget: None,
            seed: Seed::new(any_seed()),
        });
        assert_eq!(selection.uninformative, ["flat"]);
        assert_eq!(selection.weights["flat"], 0.0);
        assert!((selection.weights["cheap"] / selection.weights["dear"] - 10.0).abs() < 1e-9);
        assert!(selection.weights["new"] > 0.0, "an unseen task is drawn");
        assert!(!selection.tasks.contains(&"flat".to_string()));
        assert_eq!(selection.tasks.len(), 3);
    }

    #[test]
    fn under_shaped_partial_a_task_that_only_fails_can_still_teach() {
        let pool = vec!["stuck".to_string()];
        let mut history = Vec::new();
        for f in [0.2, 0.8, 0.5] {
            history.push(Scored {
                correctness: Some(f),
                ..scored("stuck", false, 100)
            });
        }
        let verdict = rewarding(RewardPreset::Verdict);
        let shaped = rewarding(RewardPreset::ShapedPartial);
        let (families, rates) = (BTreeMap::new(), PassRates::default());
        let request = |rewarding| SignalRequest {
            pool: &pool,
            history: &history,
            families: &families,
            rewarding,
            rates: &rates,
            budget: None,
            seed: Seed::new(any_seed()),
        };
        assert!(select(&request(&verdict)).tasks.is_empty());
        assert_eq!(select(&request(&shaped)).tasks, ["stuck"]);
    }
}
