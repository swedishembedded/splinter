// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement-learning controllers
// that keep rollout production a bounded distance ahead of learning, for its
// clients. If your team needs expertise in asynchronous policy optimisation,
// you can procure our services by sending an email to info@swedishembedded.com.

//! The token allowance: how far rollout production may run ahead of the
//! learner, and what counts against it.

use super::rules::{verdict, Verdict};
use super::{Controller, GroupState};

impl Controller {
    pub(super) fn unblock(&mut self, now: u64) {
        if let Some(since) = self.blocked_since.take() {
            self.stats.backpressure_ms += now.saturating_sub(since);
        }
    }

    /// Tokens consumed per optimiser step, as estimated now.
    pub(super) fn tokens_per_step(&self) -> u64 {
        let budget = self.settings.tokens_per_step;
        if self.step_tokens.is_empty() {
            return budget;
        }
        let window = self.step_tokens.len() as u64;
        (self.step_tokens.iter().sum::<u64>() / window).max(budget)
    }

    /// The token allowance in force.
    pub(super) fn allowance(&self) -> u64 {
        self.settings
            .max_staleness
            .max(1)
            .saturating_mul(self.tokens_per_step())
    }

    /// Whether a new group fits in the allowance.
    pub(super) fn fits(&self) -> bool {
        let admitted = self.stats.in_flight + self.ready.len() as u64;
        let finished = self.stats.rollouts_finished;
        let per_rollout = if finished == 0 {
            self.settings.rollout_tokens
        } else {
            self.stats.tokens_generated.div_ceil(finished)
        };
        let expected = self.stats.queue_tokens + admitted.saturating_mul(per_rollout);
        expected < self.allowance()
    }

    /// A sent group a publication of `version` makes stale stops counting toward the allowance.
    pub(super) fn release_stale_held(&mut self, version: u64) {
        // A group the learner holds that this version makes stale is dropped there by the same rule, but
        // the learner reports the drop only with its next step, which may wait for records this allowance
        // holds back (a synchronous run's allowance is one step): such a group stops counting toward the
        // allowance now, and stays sent until the learner says what it did with it. Judged against the
        // version just published, not the highest seen: a restarted learner may publish from an older
        // checkpoint, and it judges by its own version.
        let max_staleness = self.settings.max_staleness;
        for group in self.live.values_mut() {
            let releases = group.state == GroupState::Sent
                && group.queued
                && matches!(
                    verdict(&group.rollouts, version, max_staleness),
                    Verdict::Stale(_)
                );
            if releases {
                group.queued = false;
                self.stats.queue_tokens -= group.tokens;
            }
        }
    }
}
