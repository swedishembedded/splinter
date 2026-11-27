// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement-learning controllers
// that never train a group twice or lose one they produced, for its clients.
// If your team needs expertise in asynchronous policy optimisation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Online policy-gradient runs: groups of rollouts produced continuously
//! against the policy the learner serves, each scored, filtered or dropped
//! as too old, and streamed to the learner as soon as it is complete, with
//! no barrier between groups.
//!
//! * [`control`] - the controller core: admission under the token
//!   allowance, scoring, staleness, what is sent; pure, the time an argument.
//! * [`seams`] - the rollout executor, the learner, the task selector and
//!   the durable group ledger the controller is carried out with.
//! * [`driver`] - the loop over those seams, every change durable before
//!   it takes effect, resumable after a crash at any point.
//! * [`stats`] - what a run reports.
//! * [`group_log`] - `groups.jsonl`, the live log of every group's
//!   transitions.
//! * [`health`] - the run's training health, per step and per version.
//! * [`frontier`] - the frontier selector: groups drawn where they teach.
//! * [`watchdog`] - detects training that moves the policy the wrong way,
//!   and rolls it back on evidence.
//! * [`brain`] - the learner on brain's streaming policy-gradient job.
//! * [`rollout`] - a finished rollout from its projected model calls, the
//!   same for every executor.

pub mod brain;
pub mod control;
pub mod driver;
pub mod frontier;
pub mod group_log;
pub mod health;
pub mod rollout;
pub mod seams;
pub mod stats;
pub mod watchdog;

use splinter_core::factory::group::GroupKey;

/// Invalid settings for an online run.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("online settings: {0}")]
pub struct SettingsError(pub String);

/// How an online run is controlled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Settings {
    /// The prefix of every group key (`<namespace>:<seq>`).
    pub namespace: String,
    /// Rollouts per group; at least two, or no reward could differ.
    pub group_size: u32,
    /// How many policy versions behind the learner a group's oldest request
    /// may be.
    pub max_staleness: u64,
    /// Rollouts run at once at most: what the environment can hold.
    pub rollout_slots: u32,
    /// The learner's step budget in policy tokens: the first estimate of
    /// tokens consumed per step, and its floor.
    pub tokens_per_step: u64,
    /// The policy tokens a rollout is assumed to sample until one has
    /// finished (a pessimistic bound such as the sampling budget); then the
    /// mean of the finished ones is used.
    pub rollout_tokens: u64,
    /// How many step reports the moving estimate averages.
    pub step_window: usize,
    /// Infrastructure failures a rollout is retried after; one more and it
    /// is abandoned.
    pub max_failures: u32,
    /// How long a full learner queue holds sends back, in milliseconds.
    pub send_retry_ms: u64,
    /// How long the driver waits for a rollout before polling the learner
    /// again, in milliseconds.
    pub poll_ms: u64,
    /// The run's epoch: every group it creates belongs to it, and a group
    /// of another epoch (one a rollback ended) is never sent. 0 for a run
    /// that never rolled back.
    pub epoch: u64,
    /// Train only rollouts whose every request one policy version decoded
    /// (a span of width 0); a rollout that straddles a publication is
    /// scored and counted, never trained. Off, a rollout's oldest version
    /// bounds its staleness.
    pub pin_versions: bool,
    /// Run every request of an episode on the version it started on: each
    /// rollout asks the id of the learner's newest version when it starts
    /// (when the learner names one), so its span is one version.
    pub pin_episodes: bool,
}

impl Settings {
    /// Checks every setting.
    ///
    /// # Errors
    /// A setting is out of range.
    pub fn validate(&self) -> Result<(), SettingsError> {
        let fail = |what: &str| Err(SettingsError(what.to_string()));
        if GroupKey::numbered(&self.namespace, 0).is_err() {
            return fail("the namespace makes no valid group key");
        }
        if self.group_size < 2 {
            return fail("group_size must be at least 2");
        }
        if self.rollout_slots == 0 {
            return fail("rollout_slots must be at least 1");
        }
        if self.tokens_per_step == 0 {
            return fail("tokens_per_step must be at least 1");
        }
        if self.rollout_tokens == 0 {
            return fail("rollout_tokens must be at least 1");
        }
        if self.step_window == 0 {
            return fail("step_window must be at least 1");
        }
        if self.poll_ms == 0 {
            return fail("poll_ms must be at least 1");
        }
        Ok(())
    }
}
