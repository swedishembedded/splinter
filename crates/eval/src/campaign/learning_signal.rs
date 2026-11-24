// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for spending a reinforcement-learning
// budget on the tasks that teach the most per token, for its clients. If your
// team needs expertise in curricula and the economics of policy-gradient
// training on agent rollouts, you can procure our services by sending an email
// to info@swedishembedded.com.

//! How much a task's attempts can teach a group-relative policy gradient, per
//! token they cost.
//!
//! A group's advantage is its rewards' spread: a task whose attempts all earn
//! the same teaches nothing, and the variance of the rewards bounds what it
//! can teach. So the **learning signal** of task `i` is
//!
//! `LS_i = Var_j(r_ij) / mean_j(cost_ij)`
//!
//! over its attempts `j`, the cost of an attempt being its completion plus its
//! uncached prompt tokens ([`splinter_core::usage::TokenUsage::cost`]).
//!
//! **Shrinkage.** Eight attempts estimate a variance poorly, and a task seen
//! once or never has none. The variance is shrunk towards what a Beta
//! posterior over the task's success rate expects:
//!
//! `V_i = (n * s2 + k * v0) / (n + k)`, with `v0 = a*b / ((a + b) * (a + b + 1))`,
//!
//! where `s2` is the population variance of the `n` rewards, `a = 1 + sum r`
//! and `b = 1 + sum (1 - r)` the Beta(1, 1) prior updated by the rewards as
//! fractional successes (clamped to `[0, 1]`), so `v0` is the posterior
//! expectation of `p * (1 - p)`, and `k` = [`PRIOR_ATTEMPTS`] the prior's
//! weight in attempts. A task never attempted has `V = 1/6`, the prior's.
//! For pass/fail rewards `s2 = r_bar * (1 - r_bar)`, so `V` is close to the
//! learnability `p * (1 - p)` with `p` smoothed.
//!
//! **Informative.** A group of two or more attempts whose rewards differ
//! ([`splinter_core::rewards::differ`]) carries an advantage; one whose rewards are all alike
//! carries none, whatever its shrunk variance says.
//!
//! **Families.** Tasks generated from one lineage (seeds of one template) are
//! not independent evidence: a report counts them by family, never by seed.
//!
//! **Selection.** Under the variance-per-cost curriculum a task is drawn in
//! proportion to `sqrt(V_i) / mean cost_i` ([`selection_weight`]), and a group
//! measured uninformative is never drawn. [`SignalReport::selected`]
//! summarises the measured groups as that draw would weigh them, so the same
//! rollouts say what the selection gains before any is spent.
//!
//! An unmeasured number (no usage recorded) is absent, never `0`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The weight, in attempts, of the Beta prior the variance is shrunk towards.
pub const PRIOR_ATTEMPTS: f64 = 2.0;

/// The prior's own expected `p * (1 - p)`, a Beta(1, 1)'s: what a task never
/// attempted is credited with.
pub const PRIOR_VARIANCE: f64 = 1.0 / 6.0;

/// One attempt: its reward and what it cost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Attempt {
    /// The reward it earned.
    pub reward: f64,
    /// Completion plus uncached prompt tokens; `None` when not counted.
    pub cost: Option<u64>,
    /// Completion tokens; `None` when not counted.
    pub completion: Option<u64>,
}

/// One task's attempts, measured.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TaskSignal {
    /// The task.
    pub task: String,
    /// Its family: the lineage it was generated from, else the task itself.
    pub family: String,
    /// Attempts measured.
    pub attempts: usize,
    /// Their mean reward; `None` with none.
    pub mean_reward: Option<f64>,
    /// The population variance of their rewards; `None` with none.
    pub variance: Option<f64>,
    /// The shrunk variance `V` (module documentation).
    pub shrunk_variance: f64,
    /// The mean cost of the attempts that counted one; `None` when none did.
    pub mean_cost: Option<f64>,
    /// Completion tokens summed over the attempts, when every one counted
    /// them.
    pub completion_tokens: Option<u64>,
    /// Two or more attempts whose rewards differ.
    pub informative: bool,
    /// `V / mean cost`, per token; `None` without a cost.
    pub learning_signal: Option<f64>,
}

impl TaskSignal {
    /// The measurement of `attempts` at `task`, of `family`.
    #[must_use]
    pub fn of(task: &str, family: &str, attempts: &[Attempt]) -> Self {
        let n = attempts.len();
        let rewards: Vec<f64> = attempts.iter().map(|a| a.reward).collect();
        let mean = (n > 0).then(|| rewards.iter().sum::<f64>() / n as f64);
        let variance =
            mean.map(|m| rewards.iter().map(|r| (r - m).powi(2)).sum::<f64>() / n as f64);
        let informative = splinter_core::rewards::differ(rewards.iter().copied());
        let successes: f64 = rewards.iter().map(|r| r.clamp(0.0, 1.0)).sum();
        let (a, b) = (1.0 + successes, 1.0 + (n as f64 - successes));
        let prior = a * b / ((a + b) * (a + b + 1.0));
        let shrunk = (n as f64 * variance.unwrap_or(0.0) + PRIOR_ATTEMPTS * prior)
            / (n as f64 + PRIOR_ATTEMPTS);
        let costs: Vec<u64> = attempts.iter().filter_map(|a| a.cost).collect();
        let mean_cost = (!costs.is_empty())
            .then(|| costs.iter().sum::<u64>() as f64 / costs.len() as f64)
            .filter(|c| *c > 0.0);
        let completion_tokens = (n > 0)
            .then(|| attempts.iter().map(|a| a.completion).sum::<Option<u64>>())
            .flatten();
        Self {
            task: task.to_string(),
            family: family.to_string(),
            attempts: n,
            mean_reward: mean,
            variance,
            shrunk_variance: shrunk,
            mean_cost,
            completion_tokens,
            informative,
            learning_signal: mean_cost.map(|c| shrunk / c),
        }
    }

    /// Measured enough to say whether it is informative: two attempts.
    #[must_use]
    pub fn measured(&self) -> bool {
        self.attempts >= 2
    }
}

/// The weight the variance-per-cost curriculum draws `task` by:
/// `sqrt(V) / mean cost`, the mean cost `fallback_cost` when the task counted
/// none; zero for a task measured uninformative.
#[must_use]
pub fn selection_weight(task: &TaskSignal, fallback_cost: f64) -> f64 {
    if task.measured() && !task.informative {
        return 0.0;
    }
    let cost = task.mean_cost.unwrap_or(fallback_cost);
    if cost > 0.0 {
        task.shrunk_variance.sqrt() / cost
    } else {
        0.0
    }
}

/// The cost a task that counted none is taken to have: the mean of the
/// tasks' mean costs, else one (every task alike).
#[must_use]
pub fn fallback_cost(tasks: &[TaskSignal]) -> f64 {
    let costs: Vec<f64> = tasks.iter().filter_map(|t| t.mean_cost).collect();
    if costs.is_empty() {
        1.0
    } else {
        costs.iter().sum::<f64>() / costs.len() as f64
    }
}

/// One family's tasks, measured.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FamilySignal {
    /// The family.
    pub family: String,
    /// Its tasks (seeds) measured.
    pub tasks: usize,
    /// Of those, informative.
    pub informative_tasks: usize,
    /// Attempts over its tasks.
    pub attempts: usize,
    /// The mean of its tasks' learning signal, over those with one.
    pub mean_learning_signal: Option<f64>,
}

/// The signal of a set of groups.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SignalSummary {
    /// Groups of two or more attempts, by weight (a count when unweighted).
    pub groups: f64,
    /// Of those, informative.
    pub informative: f64,
    /// `informative / groups`; `None` with no group.
    pub informative_share: Option<f64>,
    /// Families (lineages) with a measured group.
    pub families: usize,
    /// Of those, with an informative group.
    pub informative_families: usize,
    /// The variance of the rewards summed over the groups.
    pub variance_sum: f64,
    /// Completion tokens over the groups that counted them.
    pub completion_tokens: f64,
    /// `variance_sum` over the groups that counted completion tokens, per
    /// million of them; `None` when none did.
    pub variance_per_million_completion: Option<f64>,
}

impl SignalSummary {
    /// The summary of the measured groups of `tasks`, each weighted by
    /// `weight`.
    fn of(tasks: &[TaskSignal], weight: impl Fn(&TaskSignal) -> f64) -> Self {
        let mut summary = Self::default();
        let mut families: BTreeMap<&str, bool> = BTreeMap::new();
        let mut counted_variance = 0.0;
        for task in tasks.iter().filter(|t| t.measured()) {
            let w = weight(task);
            let informative = families.entry(&task.family).or_insert(false);
            *informative |= task.informative;
            summary.groups += w;
            if task.informative {
                summary.informative += w;
            }
            let variance = task.variance.unwrap_or(0.0);
            summary.variance_sum += w * variance;
            if let Some(tokens) = task.completion_tokens {
                summary.completion_tokens += w * tokens as f64;
                counted_variance += w * variance;
            }
        }
        summary.informative_share =
            (summary.groups > 0.0).then(|| summary.informative / summary.groups);
        summary.families = families.len();
        summary.informative_families = families.values().filter(|i| **i).count();
        summary.variance_per_million_completion = (summary.completion_tokens > 0.0)
            .then(|| counted_variance / summary.completion_tokens * 1e6);
        summary
    }
}

/// The learning signal of some tasks: per task, per family, and summed with
/// and without the variance-per-cost selection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SignalReport {
    /// Per task, by task.
    pub tasks: Vec<TaskSignal>,
    /// Per family, by family.
    pub families: Vec<FamilySignal>,
    /// The measured groups, each counted once.
    pub uniform: SignalSummary,
    /// The measured groups as the variance-per-cost selection would draw
    /// them ([`selection_weight`]): a weight sums to one over them.
    pub selected: SignalSummary,
}

impl SignalReport {
    /// The report of `tasks`.
    #[must_use]
    pub fn of(mut tasks: Vec<TaskSignal>) -> Self {
        tasks.sort_by(|a, b| a.task.cmp(&b.task));
        let mut by_family: BTreeMap<&str, Vec<&TaskSignal>> = BTreeMap::new();
        for task in &tasks {
            by_family.entry(&task.family).or_default().push(task);
        }
        let families = by_family
            .into_iter()
            .map(|(family, members)| {
                let signals: Vec<f64> = members.iter().filter_map(|t| t.learning_signal).collect();
                FamilySignal {
                    family: family.to_string(),
                    tasks: members.len(),
                    informative_tasks: members.iter().filter(|t| t.informative).count(),
                    attempts: members.iter().map(|t| t.attempts).sum(),
                    mean_learning_signal: (!signals.is_empty())
                        .then(|| signals.iter().sum::<f64>() / signals.len() as f64),
                }
            })
            .collect();
        let fallback = fallback_cost(&tasks);
        let total: f64 = tasks
            .iter()
            .filter(|t| t.measured())
            .map(|t| selection_weight(t, fallback))
            .sum();
        let selected = SignalSummary::of(&tasks, |t| {
            if total > 0.0 {
                selection_weight(t, fallback) / total
            } else {
                0.0
            }
        });
        Self {
            uniform: SignalSummary::of(&tasks, |_| 1.0),
            selected,
            families,
            tasks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attempts(rewards: &[f64], cost: u64) -> Vec<Attempt> {
        rewards
            .iter()
            .map(|r| Attempt {
                reward: *r,
                cost: Some(cost),
                completion: Some(cost / 2),
            })
            .collect()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    /// The formula of the module documentation, by hand: four passes in
    /// eight, `s2 = 0.25`, `a = b = 5`, `v0 = 25 / 110`.
    #[test]
    fn the_shrunk_variance_is_the_sample_pulled_towards_the_beta_posterior() {
        let t = TaskSignal::of("t", "f", &attempts(&[1., 1., 1., 1., 0., 0., 0., 0.], 100));
        let expected = (8.0 * 0.25 + 2.0 * 25.0 / 110.0) / 10.0;
        assert!(close(t.shrunk_variance, expected), "{}", t.shrunk_variance);
        assert!(close(t.learning_signal.unwrap(), expected / 100.0));
        assert!(t.informative);
        let never = TaskSignal::of("u", "f", &[]);
        assert!(close(never.shrunk_variance, PRIOR_VARIANCE));
        assert_eq!(never.learning_signal, None, "no cost counted, no signal");
    }

    #[test]
    fn an_all_alike_group_is_never_drawn_and_a_cheaper_spread_one_is_drawn_more() {
        let flat = TaskSignal::of("flat", "f", &attempts(&[0.; 8], 100));
        let cheap = TaskSignal::of("cheap", "f", &attempts(&[1., 0., 1., 0.], 100));
        let dear = TaskSignal::of("dear", "g", &attempts(&[1., 0., 1., 0.], 400));
        assert!(!flat.informative);
        assert_eq!(selection_weight(&flat, 1.0), 0.0);
        assert!(close(
            selection_weight(&cheap, 1.0),
            4.0 * selection_weight(&dear, 1.0)
        ));
        let unseen = TaskSignal::of("new", "h", &[]);
        assert!(close(
            selection_weight(&unseen, 50.0),
            PRIOR_VARIANCE.sqrt() / 50.0
        ));
    }

    #[test]
    fn the_report_counts_families_not_seeds_and_the_selection_drops_the_flat_groups() {
        let report = SignalReport::of(vec![
            TaskSignal::of("a1", "lineage-a", &attempts(&[0.; 4], 100)),
            TaskSignal::of("a2", "lineage-a", &attempts(&[1., 0., 1., 0.], 100)),
            TaskSignal::of("b1", "lineage-b", &attempts(&[1.; 4], 100)),
        ]);
        assert_eq!(report.families.len(), 2);
        assert_eq!(report.uniform.families, 2);
        assert_eq!(report.uniform.informative_families, 1);
        assert!(close(report.uniform.informative_share.unwrap(), 1.0 / 3.0));
        // Variance 0.25 of one group over 3 * 200 completion tokens.
        assert!(close(
            report.uniform.variance_per_million_completion.unwrap(),
            0.25 / 600.0 * 1e6
        ));
        assert!(close(report.selected.informative_share.unwrap(), 1.0));
        assert!(close(
            report.selected.variance_per_million_completion.unwrap(),
            0.25 / 200.0 * 1e6
        ));
    }
}
