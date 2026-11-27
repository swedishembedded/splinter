// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for spending a reinforcement-learning
// budget only on tasks that teach, for its clients. If your team needs
// expertise in curricula for policy-gradient training you can procure our
// services by sending an email to info@swedishembedded.com.

//! The frontier selector: each group's task drawn where a group can teach.
//!
//! A group teaches only when its rewards differ, which needs a task the
//! policy passes sometimes, not always and not never. Each task's pass rate
//! has a Beta posterior (a uniform prior; each scored rollout's reward,
//! in `[0, 1]`, adds to its passes and the rest to its failures). It follows
//! a moving policy: evidence ages by [`DECAY`] for every policy version
//! published since it was counted (by group, for a caller that names no
//! version). A task is drawn with probability proportional to its
//! learning signal, the posterior's expected reward variance
//! `E[p (1 - p)] = a b / ((a + b) (a + b + 1))`, which is largest at the
//! frontier and small at either end. A task never passed keeps at least
//! [`EXPLORATION_FLOOR`] of the largest weight, so a task the policy cannot
//! yet solve is still visited. A task whose pass rate reaches the retirement
//! rate on enough scored rollouts is retired for good
//! ([`splinter_eval::campaign::pass_rate`]): the policy has learned it.
//!
//! The draws are a function of the run's seed and the group's place alone,
//! and the posterior a function of the groups the controller reports
//! ([`TaskSelector::observe`]), which a resumed controller replays from the
//! ledger, for its own epoch only: the selector resumes with no state of its
//! own, and a rolled-back epoch's statistics are reset. The selector never
//! sees anything but the run's own rollouts.

use std::collections::BTreeMap;

use serde::Serialize;
use splinter_core::digest::digest_of;
use splinter_eval::campaign::pass_rate::{PassRates, RETIRE_AT};

use super::seams::TaskSelector;
use super::SettingsError;

/// How much evidence weighs for every policy version published since it was
/// counted (or, for a caller that names no version, every group since).
pub const DECAY: f64 = 0.9;

/// The share of the largest learning signal a task never passed keeps.
pub const EXPLORATION_FLOOR: f64 = 0.1;

/// Scored rollouts a task needs before it can be retired.
pub const RETIRE_AFTER: usize = 8;

/// One task's posterior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Posterior {
    /// The policy version its evidence was last counted at.
    pub version: Option<u64>,
    /// Weighted passes.
    pub passes: f64,
    /// Weighted failures.
    pub failures: f64,
    /// Groups observed.
    pub groups: u64,
}

impl Posterior {
    /// The Beta posterior's expected reward variance under a uniform prior.
    #[must_use]
    pub fn signal(&self) -> f64 {
        let (a, b) = (1.0 + self.passes, 1.0 + self.failures);
        a * b / ((a + b) * (a + b + 1.0))
    }

    /// Whether any rollout of the task passed.
    #[must_use]
    pub fn ever_passed(&self) -> bool {
        self.passes > 0.0
    }
}

/// The frontier selector over a run's tasks, `groups` groups in all.
#[derive(Clone, Debug)]
pub struct Frontier {
    tasks: Vec<String>,
    groups: u64,
    seed: u64,
    posteriors: BTreeMap<String, Posterior>,
    retirement: PassRates,
}

impl Frontier {
    /// `groups` groups drawn over `tasks`, the draws derived from `seed`.
    ///
    /// # Errors
    /// No task is given.
    pub fn new(tasks: Vec<String>, groups: u64, seed: u64) -> Result<Self, SettingsError> {
        if tasks.is_empty() {
            return Err(SettingsError("a frontier selector needs a task".into()));
        }
        let retirement =
            PassRates::new(RETIRE_AT, RETIRE_AFTER).map_err(|e| SettingsError(e.to_string()))?;
        Ok(Self {
            tasks,
            groups,
            seed,
            posteriors: BTreeMap::new(),
            retirement,
        })
    }

    /// Each task's posterior, by task.
    #[must_use]
    pub fn posteriors(&self) -> &BTreeMap<String, Posterior> {
        &self.posteriors
    }

    /// Whether `task` is retired.
    #[must_use]
    pub fn is_retired(&self, task: &str) -> bool {
        self.retirement.is_retired(task)
    }

    /// Each task still drawn with its weight.
    fn weights(&self) -> Vec<(&String, f64)> {
        let live: Vec<(&String, Posterior)> = self
            .tasks
            .iter()
            .filter(|t| !self.retirement.is_retired(t))
            .map(|t| (t, self.posteriors.get(t).copied().unwrap_or_default()))
            .collect();
        let top = live.iter().map(|(_, p)| p.signal()).fold(0.0_f64, f64::max);
        live.into_iter()
            .map(|(t, p)| {
                let w = p.signal();
                (
                    t,
                    if p.ever_passed() {
                        w
                    } else {
                        w.max(EXPLORATION_FLOOR * top)
                    },
                )
            })
            .collect()
    }

    /// A uniform draw in `[0, 1)` for the run's `seq`-th group.
    fn uniform(&self, seq: u64) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let u = (digest_of(&("frontier", self.seed, seq)).leading_u64() >> 11) as f64;
        u / (1u64 << 53) as f64
    }
}

impl TaskSelector for Frontier {
    fn select(&mut self, seq: u64) -> Option<String> {
        if seq >= self.groups {
            return None;
        }
        let weights = self.weights();
        let total: f64 = weights.iter().map(|(_, w)| w).sum();
        if weights.is_empty() || total <= 0.0 {
            return None;
        }
        let mut at = self.uniform(seq) * total;
        for (task, weight) in &weights {
            if at < *weight {
                return Some((*task).clone());
            }
            at -= weight;
        }
        weights.last().map(|(t, _)| (*t).clone())
    }

    fn observe(&mut self, task: &str, rewards: &[f64]) {
        let at = self
            .posteriors
            .get(task)
            .and_then(|p| p.version)
            .map_or(0, |v| v + 1);
        self.observe_at(task, rewards, at);
    }

    fn observe_at(&mut self, task: &str, rewards: &[f64], version: u64) {
        let posterior = self.posteriors.entry(task.to_string()).or_default();
        // Evidence counted under an older policy weighs less for every
        // version published since; a group of an older version than the
        // newest counted ages nothing.
        let elapsed = posterior
            .version
            .map_or(0, |last| version.saturating_sub(last));
        let aged = DECAY.powi(i32::try_from(elapsed).unwrap_or(i32::MAX));
        posterior.passes *= aged;
        posterior.failures *= aged;
        posterior.version = Some(posterior.version.map_or(version, |last| last.max(version)));
        for reward in rewards {
            let r = reward.clamp(0.0, 1.0);
            posterior.passes += r;
            posterior.failures += 1.0 - r;
        }
        posterior.groups += 1;
        let outcomes: Vec<Option<bool>> = rewards.iter().map(|r| Some(*r >= 1.0)).collect();
        self.retirement.observe(task, 0, &outcomes);
    }
}

/// Which selector draws an online run's tasks.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, Serialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum SelectorKind {
    /// The tasks in turn.
    #[default]
    RoundRobin,
    /// Where a group can teach ([`Frontier`]).
    Frontier,
}

impl SelectorKind {
    /// Whether it is the default, which a recipe leaves out.
    #[must_use]
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// A run's selector, of either kind.
#[derive(Clone, Debug)]
pub enum Selector {
    /// The tasks in turn.
    RoundRobin(super::seams::RoundRobin),
    /// Where a group can teach.
    Frontier(Frontier),
}

impl Selector {
    /// The selector of `kind` over `tasks`, `groups` groups in all, its
    /// draws derived from `seed`.
    ///
    /// # Errors
    /// No task is given.
    pub fn new(
        kind: SelectorKind,
        tasks: Vec<String>,
        groups: u64,
        seed: u64,
    ) -> Result<Self, SettingsError> {
        Ok(match kind {
            SelectorKind::RoundRobin => {
                Self::RoundRobin(super::seams::RoundRobin::new(tasks, groups)?)
            }
            SelectorKind::Frontier => Self::Frontier(Frontier::new(tasks, groups, seed)?),
        })
    }

    /// What it knows, for a run's report: each task's posterior and whether
    /// it retired, for the frontier selector; nothing for the round robin.
    #[must_use]
    pub fn report(&self) -> serde_json::Value {
        match self {
            Self::RoundRobin(_) => serde_json::json!({"kind": SelectorKind::RoundRobin}),
            Self::Frontier(f) => {
                let tasks: BTreeMap<&String, serde_json::Value> = f
                    .posteriors()
                    .iter()
                    .map(|(t, p)| {
                        (
                            t,
                            serde_json::json!({"posterior": p, "signal": p.signal(),
                                               "retired": f.is_retired(t)}),
                        )
                    })
                    .collect();
                serde_json::json!({"kind": SelectorKind::Frontier, "tasks": tasks})
            }
        }
    }
}

impl TaskSelector for Selector {
    fn select(&mut self, seq: u64) -> Option<String> {
        match self {
            Self::RoundRobin(s) => s.select(seq),
            Self::Frontier(s) => s.select(seq),
        }
    }

    fn observe(&mut self, task: &str, rewards: &[f64]) {
        match self {
            Self::RoundRobin(s) => s.observe(task, rewards),
            Self::Frontier(s) => s.observe(task, rewards),
        }
    }

    fn observe_at(&mut self, task: &str, rewards: &[f64], version: u64) {
        match self {
            Self::RoundRobin(s) => s.observe_at(task, rewards, version),
            Self::Frontier(s) => s.observe_at(task, rewards, version),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tasks() -> Vec<String> {
        ["easy", "frontier", "hard"].map(String::from).to_vec()
    }

    #[test]
    fn groups_go_where_rewards_differ_never_passed_tasks_are_still_visited_and_learned_ones_retire()
    {
        // seed: the draws are checked by their shares over many groups; any seed serves.
        let mut selector = Frontier::new(tasks(), 4_000, 0x51).unwrap();
        for _ in 0..3 {
            selector.observe("easy", &[1.0; 4]);
            selector.observe("frontier", &[1.0, 0.0, 1.0, 0.0]);
            selector.observe("hard", &[0.0; 4]);
        }
        assert!(
            selector.is_retired("easy"),
            "passed every one of 12 rollouts"
        );
        let mut drawn: BTreeMap<String, u64> = BTreeMap::new();
        for seq in 0..4_000 {
            *drawn.entry(selector.select(seq).unwrap()).or_default() += 1;
        }
        assert_eq!(drawn.get("easy"), None, "a retired task is never drawn");
        let (frontier, hard) = (drawn["frontier"], drawn["hard"]);
        assert!(frontier > 3 * hard, "{drawn:?}");
        assert!(
            hard > 0,
            "a never-passed task keeps the exploration floor: {drawn:?}"
        );
        assert_eq!(selector.select(4_000), None, "the run's groups are spent");
    }

    #[test]
    fn the_draws_are_a_function_of_the_seed_and_the_observations_alone() {
        let run = |seed| {
            let mut s = Frontier::new(tasks(), 50, seed).unwrap();
            s.observe("hard", &[0.0, 1.0]);
            (0..50)
                .map(|seq| s.select(seq).unwrap())
                .collect::<Vec<_>>()
        };
        // seed: two fixed seeds compare a draw with itself and with another.
        assert_eq!(run(7), run(7));
        assert_ne!(run(7), run(8));
        let all_retired = {
            let mut s = Frontier::new(vec!["only".into()], 10, 1).unwrap();
            for _ in 0..3 {
                s.observe("only", &[1.0; 4]);
            }
            s.select(0)
        };
        assert_eq!(all_retired, None, "nothing left to teach ends the run");
    }

    #[test]
    fn evidence_ages_with_every_version_published_since_it_was_counted() {
        // seed: no draw is read.
        let mut selector = Frontier::new(tasks(), 10, 1).unwrap();
        selector.observe_at("hard", &[0.0, 0.0], 3);
        selector.observe_at("hard", &[1.0, 1.0], 3);
        let same = selector.posteriors()["hard"];
        assert!(
            (same.failures - 2.0).abs() < 1e-12,
            "the same version ages nothing: {same:?}"
        );
        selector.observe_at("hard", &[1.0, 1.0], 6);
        let later = selector.posteriors()["hard"];
        assert!(
            (later.failures - 2.0 * DECAY.powi(3)).abs() < 1e-12,
            "{later:?}"
        );
        assert!(
            (later.passes - (2.0 * DECAY.powi(3) + 2.0)).abs() < 1e-12,
            "{later:?}"
        );
        selector.observe_at("hard", &[0.0], 5);
        assert_eq!(
            selector.posteriors()["hard"].version,
            Some(6),
            "an older group ages nothing back"
        );
    }
}
