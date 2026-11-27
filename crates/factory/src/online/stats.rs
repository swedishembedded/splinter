// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement-learning controllers
// that never train a group twice or lose one they produced, for its clients.
// If your team needs expertise in asynchronous policy optimisation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What an online run reports: enough to say whether it is generation- or
//! training-bound, how much it produced was used, and what reward it earns.
//! A statistic with nothing to measure is `None`, never zero.

use std::collections::BTreeMap;

use serde::Serialize;

/// Running mean and population variance (Welford).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct Moments {
    count: u64,
    mean: f64,
    m2: f64,
}

impl Moments {
    /// Adds one value.
    pub fn add(&mut self, value: f64) {
        self.count += 1;
        #[allow(clippy::cast_precision_loss)]
        let n = self.count as f64;
        let delta = value - self.mean;
        self.mean += delta / n;
        self.m2 += delta * (value - self.mean);
    }

    /// How many values were added.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Their mean; `None` without any.
    #[must_use]
    pub fn mean(&self) -> Option<f64> {
        (self.count > 0).then_some(self.mean)
    }

    /// Their population standard deviation; `None` without any.
    #[must_use]
    pub fn std(&self) -> Option<f64> {
        #[allow(clippy::cast_precision_loss)]
        (self.count > 0).then(|| (self.m2 / self.count as f64).max(0.0).sqrt())
    }
}

/// The counters and gauges of an online run.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Stats {
    /// Groups created.
    pub groups_started: u64,
    /// Groups whose every rollout ended.
    pub groups_scored: u64,
    /// Scored groups whose rewards do not differ: counted in the reward
    /// statistics, never trained.
    pub filtered_zero_variance: u64,
    /// Groups whose rewards differ but whose exact rollouts' do not (or
    /// that have fewer than two): no training material.
    pub filtered_not_exact: u64,
    /// Groups with no scored rollout at all.
    pub filtered_unscored: u64,
    /// Groups dropped here as more than `max_staleness` versions behind.
    pub dropped_stale: u64,
    /// Groups the learner dropped as too old.
    pub dropped_by_learner: u64,
    /// Groups sent and accepted (or already held) by the learner.
    pub groups_sent: u64,
    /// Sends of a group after a resume (the learner may hold it already).
    pub resends: u64,
    /// Groups applied by the learner.
    pub groups_applied: u64,
    /// Groups in the learner's durable checkpoints.
    pub groups_committed: u64,
    /// Sends the learner answered with a full queue.
    pub learner_full: u64,
    /// The learner's queue capacity from its last full answer.
    pub learner_allowance: Option<u64>,
    /// Rollouts that finished with an outcome.
    pub rollouts_finished: u64,
    /// Finished rollouts with no reward (left out of their baseline).
    pub rollouts_unscored: u64,
    /// Finished rollouts without exact ids and behaviour log-probabilities:
    /// never sent.
    pub rollouts_not_exact: u64,
    /// Finished rollouts that hit a length limit: counted in the reward
    /// statistics, never sent (a learner trains none of them).
    pub rollouts_truncated: u64,
    /// Finished rollouts a limit ended before their agent finished, by cause
    /// (scored failures, never retried).
    pub truncations: BTreeMap<String, u64>,
    /// Infrastructure failures of rollouts (each retried or abandoned).
    pub rollout_failures: u64,
    /// Rollouts abandoned after their retry budget.
    pub rollouts_abandoned: u64,
    /// Policy tokens of all finished rollouts.
    pub tokens_generated: u64,
    /// Rewards of every scored rollout, zero-variance groups included.
    pub reward: Moments,
    /// The mean reward of each zero-variance group, over those groups.
    pub zero_std_reward: Moments,
    /// The staleness of each group when sent (versions behind -> groups).
    pub staleness: BTreeMap<u64, u64>,
    /// Rollouts started and not ended.
    pub in_flight: u64,
    /// Tokens produced and not yet applied, filtered or dropped.
    pub queue_tokens: u64,
    /// How long admission was held back by the token allowance with a
    /// rollout slot free, in milliseconds (this process's lifetime).
    pub backpressure_ms: u64,
    /// The learner's published version as last seen.
    pub learner_version: u64,
    /// The current estimate of tokens consumed per optimiser step.
    pub tokens_per_step: u64,
    /// The token allowance in force.
    pub allowance: u64,
    /// Learner acknowledgements of groups this run does not hold live.
    pub stray_acks: u64,
}

impl Stats {
    /// The share of scored groups (with at least one reward) whose rewards
    /// do not differ; `None` with none scored.
    #[must_use]
    pub fn frac_zero_std(&self) -> Option<f64> {
        let measured = self.groups_scored - self.filtered_unscored;
        #[allow(clippy::cast_precision_loss)]
        (measured > 0).then(|| self.filtered_zero_variance as f64 / measured as f64)
    }
}
