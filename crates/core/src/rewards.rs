// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reinforcement-learning pipelines whose
// every stage agrees on what a learning signal is, for its clients. If your
// team needs expertise in group-relative policy optimisation then you can
// procure our services by sending an email to info@swedishembedded.com.

//! When a group of attempts' rewards differ: the one rule the campaign's
//! sampler, its learning-signal estimate and the policy-gradient dataset
//! share, so a group one of them counts as informative is never flat to
//! another.

/// How far apart the best and the worst reward of a group must be for the
/// group to say anything: closer, they are the same reward.
pub const MIN_SPREAD: f64 = 1e-12;

/// Whether `rewards` say anything: more than one, and not all alike.
#[must_use]
pub fn differ(rewards: impl IntoIterator<Item = f64>) -> bool {
    let (n, low, high) = rewards.into_iter().fold(
        (0usize, f64::INFINITY, f64::NEG_INFINITY),
        |(n, low, high), r| (n + 1, low.min(r), high.max(r)),
    );
    n >= 2 && high - low >= MIN_SPREAD
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_group_says_something_only_when_two_rewards_differ_by_the_spread() {
        assert!(!differ([]));
        assert!(!differ([1.0]));
        assert!(!differ([0.5, 0.5, 0.5]));
        assert!(!differ([0.5, 0.5 + MIN_SPREAD / 2.0]));
        assert!(differ([0.5, 0.5 + MIN_SPREAD * 2.0]));
        assert!(differ([0.0, 1.0]));
    }
}
