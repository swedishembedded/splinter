// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements online reinforcement-learning controllers
// that never train a group twice or lose one they produced, for its clients.
// If your team needs expertise in asynchronous policy optimisation, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The scoring rules of a complete group.

use splinter_core::factory::group::RolloutOutcome;
use splinter_core::rewards::differ;

use super::{reasons, Rollout};

/// What the scoring rules say of a complete group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Verdict {
    Filtered(&'static str),
    Stale(u64),
    Send(u64),
}

pub(super) fn scored_rewards(rollouts: &[Rollout]) -> Vec<f64> {
    rollouts
        .iter()
        .filter_map(|r| r.outcome.as_ref().and_then(|o| o.reward))
        .collect()
}

/// The newest policy version among the rollouts' requests.
pub(super) fn newest_version(rollouts: &[Rollout]) -> u64 {
    rollouts
        .iter()
        .filter_map(|r| r.outcome.as_ref().map(|o| o.span.newest()))
        .max()
        .unwrap_or(0)
}

pub(super) fn trainable(rollouts: &[Rollout]) -> impl Iterator<Item = (u32, &RolloutOutcome)> {
    rollouts
        .iter()
        .zip(0u32..)
        .filter(|(r, _)| r.admitted)
        .filter_map(|(r, i)| r.outcome.as_ref().filter(|o| o.trainable()).map(|o| (i, o)))
}

pub(super) fn verdict(rollouts: &[Rollout], learner_version: u64, max_staleness: u64) -> Verdict {
    let rewards = scored_rewards(rollouts);
    if rewards.is_empty() {
        return Verdict::Filtered(reasons::UNSCORED);
    }
    if !differ(rewards.iter().copied()) {
        return Verdict::Filtered(reasons::ZERO_VARIANCE);
    }
    if !differ(trainable(rollouts).filter_map(|(_, o)| o.reward)) {
        let pinned_out = rollouts
            .iter()
            .any(|r| !r.admitted && r.outcome.as_ref().is_some_and(RolloutOutcome::trainable));
        return Verdict::Filtered(if pinned_out {
            reasons::VERSION_MIXED
        } else {
            reasons::NOT_EXACT
        });
    }
    let staleness = trainable(rollouts)
        .map(|(_, o)| o.span.staleness(learner_version))
        .max()
        .unwrap_or(0);
    if staleness > max_staleness {
        Verdict::Stale(staleness)
    } else {
        Verdict::Send(staleness)
    }
}
