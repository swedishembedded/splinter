// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reward signals for speech that keep
// expressiveness from costing clarity, for its clients. If your team needs
// expertise in reinforcement from feedback for voice models, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The reward a spoken take earns, and how takes of one sentence are compared.
//!
//! One sentence can be said in many good ways. Training on one recording of it
//! teaches the average of them; sampling several takes, scoring each, and
//! making the better ones likelier teaches the consistent good delivery. The
//! score has two parts that guard each other: how natural the take sounds
//! (from a judge, or from the user's own feedback) and how many words survive
//! being heard again, so a take cannot earn naturalness by mumbling nor
//! clarity by reading flat.
//!
//! Takes of one sentence are compared with each other, not with a fixed bar:
//! a long explanation scores lower than "Sure." however well it is said, so
//! each take's advantage is measured against the mean of its own group.

/// One spoken take of a sentence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Take {
    /// How natural it sounds, from 0 (stilted) to 1 (natural), as judged.
    pub naturalness: f32,
    /// The share of its words lost when it is heard again, from 0 to 1.
    pub word_error_rate: f32,
}

/// How much clarity weighs against naturalness.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RewardWeights {
    /// Reward lost per unit of word error rate.
    pub clarity: f32,
}

impl Default for RewardWeights {
    /// Losing a fifth of the words costs more than the most natural delivery
    /// can earn back.
    fn default() -> Self {
        Self { clarity: 5.0 }
    }
}

/// The reward of `take`: naturalness less the weighted word error rate. A
/// take with no judged naturalness has no reward, not a zero one.
#[must_use]
pub fn reward(take: Take, weights: RewardWeights) -> f32 {
    take.naturalness.clamp(0.0, 1.0) - weights.clarity * take.word_error_rate.clamp(0.0, 1.0)
}

/// How far each reward stands above its group's mean, in units of the
/// group's spread. A group whose rewards do not differ carries no signal and
/// advantages are all zero, so nothing is learned from a tie.
#[must_use]
pub fn group_advantages(rewards: &[f32]) -> Vec<f32> {
    if rewards.len() < 2 {
        return vec![0.0; rewards.len()];
    }
    let n = rewards.len() as f32;
    let mean = rewards.iter().sum::<f32>() / n;
    let variance = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / n;
    let spread = variance.sqrt();
    if spread < 1e-6 {
        return vec![0.0; rewards.len()];
    }
    rewards.iter().map(|r| (r - mean) / spread).collect()
}

/// The indices of the best and the worst take of a group, as a preference pair
/// to learn from; `None` when the group has fewer than two takes or its best
/// and worst earn the same.
#[must_use]
pub fn preference_pair(rewards: &[f32]) -> Option<(usize, usize)> {
    let best = rewards
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))?
        .0;
    let worst = rewards
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))?
        .0;
    (rewards[best] > rewards[worst]).then_some((best, worst))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn take(naturalness: f32, word_error_rate: f32) -> Take {
        Take {
            naturalness,
            word_error_rate,
        }
    }

    #[test]
    fn naturalness_cannot_buy_back_lost_words() {
        let weights = RewardWeights::default();
        let mumbled = reward(take(1.0, 0.25), weights);
        let flat = reward(take(0.2, 0.0), weights);
        assert!(flat > mumbled, "{flat} vs {mumbled}");
        assert!(reward(take(0.9, 0.0), weights) > reward(take(0.4, 0.0), weights));
    }

    #[test]
    fn a_take_is_measured_against_the_other_takes_of_its_own_sentence() {
        let advantages = group_advantages(&[0.2, 0.5, 0.8]);
        assert!(advantages[0] < 0.0 && advantages[2] > 0.0);
        assert!(
            advantages[1].abs() < 1e-6,
            "the middle take sits at the mean"
        );
        let sum: f32 = advantages.iter().sum();
        assert!(sum.abs() < 1e-5, "advantages are centred: {sum}");
        // The same spread at a different level teaches the same thing.
        let shifted = group_advantages(&[10.2, 10.5, 10.8]);
        for (a, b) in advantages.iter().zip(&shifted) {
            assert!((a - b).abs() < 1e-3);
        }
    }

    #[test]
    fn a_tie_teaches_nothing() {
        assert_eq!(group_advantages(&[0.5, 0.5, 0.5]), [0.0, 0.0, 0.0]);
        assert_eq!(group_advantages(&[0.5]), [0.0]);
        assert!(group_advantages(&[]).is_empty());
        assert_eq!(preference_pair(&[0.5, 0.5]), None);
        assert_eq!(preference_pair(&[0.5]), None);
    }

    #[test]
    fn the_best_and_worst_takes_form_the_preference_pair() {
        assert_eq!(preference_pair(&[0.3, 0.9, 0.1, 0.5]), Some((1, 2)));
    }
}
