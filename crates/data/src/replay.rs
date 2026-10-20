// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements continual fine-tuning that replays earlier
// verified experience so a model keeps what it learned, for its clients. If
// your team needs expertise in continual learning, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Which earlier records are replayed beside new ones: a seeded sample of
//! a fixed size, the same every time for the same records, fraction and
//! seed.
//!
//! Each record is drawn by a hash of the seed and the record's own bytes,
//! so the choice depends on what a record is, not on where it sits; the
//! `ceil(fraction * n)` records with the lowest draws are kept, in the
//! order they were given. Mixing the sample into training is the trainer's
//! business: brain replays exactly the files it is handed.
//!
//! [`select_replay`] chooses by a [`ReplayPolicy`]: what to replay is a
//! decision of the caller's, so that the policies a continual-learning
//! experiment compares are one function over the same records. A record's
//! capability is its metadata's `group` (the family it was projected from).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;

use crate::strip::draw;
use crate::Fraction;

/// The domain of the replay draw, apart from every other seeded draw.
// The name is part of the draw: renaming it would change which earlier
// records a seed replays.
const REPLAY_DOMAIN: &str = "splinter-views/replay";

/// The indices of the records of `records` replayed at `fraction` under
/// `seed`, in ascending order.
#[must_use]
pub fn replay_sample<T: AsRef<str>>(records: &[T], fraction: Fraction, seed: u64) -> Vec<usize> {
    let wanted = (fraction.get() * records.len() as f64).ceil() as usize;
    let mut drawn: Vec<(f64, usize)> = records
        .iter()
        .enumerate()
        .map(|(index, record)| {
            let subject = Digest::of(record.as_ref().as_bytes());
            (draw(REPLAY_DOMAIN, seed, &subject), index)
        })
        .collect();
    // Draws are finite, so the order is total; equal draws (equal records)
    // fall back to position.
    drawn.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut kept: Vec<usize> = drawn
        .into_iter()
        .take(wanted.min(records.len()))
        .map(|(_, i)| i)
        .collect();
    kept.sort_unstable();
    kept
}

/// Which earlier records are replayed, and how the replay is spread over
/// capabilities (the groups of the records).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "policy", rename_all = "snake_case")]
pub enum ReplayPolicy {
    /// Nothing is replayed.
    None,
    /// Records are drawn without regard to their capability: a capability
    /// with more records gets more of the replay.
    #[default]
    Uniform,
    /// Every capability gets the same number of records, as far as it has
    /// them; what one has too few of goes to the others.
    CapabilityBalanced,
    /// Capabilities get replay in proportion to their weight, typically how
    /// often the incumbent fails them; a capability of weight zero gets
    /// none. All weights zero is no failure to target: the replay is
    /// balanced instead.
    FailureTargeted {
        /// Weight by capability; a capability not named weighs zero.
        weights: BTreeMap<String, f64>,
    },
    /// Capabilities get replay in proportion to their weight, here how fast
    /// the incumbent's pass rate on each has been moving across releases per
    /// unit of cost (the caller computes it). Allocated exactly as
    /// [`ReplayPolicy::FailureTargeted`] is; kept apart so that a recorded
    /// run says which signal weighted it.
    LearningProgress {
        /// Weight by capability; a capability not named weighs zero.
        weights: BTreeMap<String, f64>,
    },
}

/// The group a record belongs to: its metadata's `group`, or the empty
/// group when it names none.
fn group_of(record: &str) -> String {
    serde_json::from_str::<serde_json::Value>(record)
        .ok()
        .and_then(|v| v["metadata"]["group"].as_str().map(str::to_string))
        .unwrap_or_default()
}

/// Splits `wanted` over groups of the sizes `capacity`, in proportion to
/// `weights`, giving no group more than it has: the surplus of a group that
/// has fewer than its share goes to the others by the same proportion. Whole
/// units go by largest remainder, ties to the earlier group. At most the
/// sum of the capacities of the groups with a positive weight is given.
fn allocate(wanted: usize, capacity: &[usize], weights: &[f64]) -> Vec<usize> {
    let mut given = vec![0usize; capacity.len()];
    let mut remaining = wanted;
    loop {
        let active: Vec<usize> = (0..capacity.len())
            .filter(|&i| given[i] < capacity[i] && weights[i] > 0.0)
            .collect();
        let total: f64 = active.iter().map(|&i| weights[i]).sum();
        if remaining == 0 || active.is_empty() || total <= 0.0 {
            return given;
        }
        let ideal: Vec<f64> = active
            .iter()
            .map(|&i| remaining as f64 * weights[i] / total)
            .collect();
        let mut handed = 0;
        for (slot, &i) in active.iter().enumerate() {
            let whole = (ideal[slot].floor() as usize).min(capacity[i] - given[i]);
            given[i] += whole;
            handed += whole;
        }
        let mut by_remainder: Vec<usize> = (0..active.len()).collect();
        by_remainder.sort_by(|&a, &b| {
            (ideal[b] - ideal[b].floor())
                .total_cmp(&(ideal[a] - ideal[a].floor()))
                .then(a.cmp(&b))
        });
        let mut left = remaining - handed;
        for slot in by_remainder {
            if left == 0 {
                break;
            }
            let i = active[slot];
            if given[i] < capacity[i] && ideal[slot].fract() > 0.0 {
                given[i] += 1;
                left -= 1;
                handed += 1;
            }
        }
        if handed == 0 {
            return given;
        }
        remaining -= handed;
    }
}

/// The indices of the records of `records` replayed under `policy` at
/// `fraction` under `seed`, in ascending order. `fraction` is the share of
/// the records replayed, as for [`replay_sample`]; the policy decides which.
#[must_use]
pub fn select_replay<T: AsRef<str>>(
    records: &[T],
    policy: &ReplayPolicy,
    fraction: Fraction,
    seed: u64,
) -> Vec<usize> {
    let wanted = (fraction.get() * records.len() as f64).ceil() as usize;
    let weights_of = |groups: &[String]| -> Vec<f64> {
        match policy {
            ReplayPolicy::None | ReplayPolicy::Uniform => Vec::new(),
            ReplayPolicy::CapabilityBalanced => vec![1.0; groups.len()],
            ReplayPolicy::FailureTargeted { weights }
            | ReplayPolicy::LearningProgress { weights } => {
                let w: Vec<f64> = groups
                    .iter()
                    .map(|g| weights.get(g).copied().unwrap_or(0.0).max(0.0))
                    .map(|w| if w.is_finite() { w } else { 0.0 })
                    .collect();
                if w.iter().all(|w| *w == 0.0) {
                    vec![1.0; groups.len()]
                } else {
                    w
                }
            }
        }
    };
    match policy {
        ReplayPolicy::None => return Vec::new(),
        ReplayPolicy::Uniform => return replay_sample(records, fraction, seed),
        _ => {}
    }
    // Records by group, each group's records in the order a hash of the
    // seed and the record draws them.
    let mut by_group: BTreeMap<String, Vec<(f64, usize)>> = BTreeMap::new();
    for (index, record) in records.iter().enumerate() {
        let subject = Digest::of(record.as_ref().as_bytes());
        by_group
            .entry(group_of(record.as_ref()))
            .or_default()
            .push((draw(REPLAY_DOMAIN, seed, &subject), index));
    }
    let groups: Vec<String> = by_group.keys().cloned().collect();
    let capacity: Vec<usize> = by_group.values().map(Vec::len).collect();
    let share = allocate(wanted, &capacity, &weights_of(&groups));
    let mut kept = Vec::new();
    for (drawn, take) in by_group.values_mut().zip(share) {
        drawn.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        kept.extend(drawn.iter().take(take).map(|(_, i)| *i));
    }
    kept.sort_unstable();
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fraction(value: f64) -> Fraction {
        Fraction::new(value).unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn the_sample_has_a_fixed_size_and_is_reproducible() {
        let records: Vec<String> = (0..40).map(|i| format!("{{\"n\":{i}}}")).collect();
        let a = replay_sample(&records, fraction(0.25), 7);
        assert_eq!(a.len(), 10);
        assert_eq!(a, replay_sample(&records, fraction(0.25), 7));
        assert_ne!(
            a,
            replay_sample(&records, fraction(0.25), 8),
            "the seed matters"
        );
        assert!(a.windows(2).all(|w| w[0] < w[1]), "given order: {a:?}");
        assert_eq!(
            replay_sample(&records, fraction(0.01), 7).len(),
            1,
            "rounded up"
        );
        assert!(replay_sample(&records, fraction(0.0), 7).is_empty());
        assert_eq!(replay_sample(&records, fraction(1.0), 7).len(), 40);
    }
    fn record(group: &str, n: usize) -> String {
        format!("{{\"metadata\":{{\"group\":\"{group}\"}},\"n\":{n}}}")
    }

    /// `counts` records of each group, named by their position.
    fn corpus(counts: &[(&str, usize)]) -> Vec<String> {
        counts
            .iter()
            .flat_map(|(group, count)| (0..*count).map(move |n| record(group, n)))
            .collect()
    }

    fn per_group(records: &[String], picked: &[usize]) -> BTreeMap<String, usize> {
        let mut tally = BTreeMap::new();
        for &i in picked {
            *tally.entry(group_of(&records[i])).or_default() += 1;
        }
        tally
    }

    fn tally(pairs: &[(&str, usize)]) -> BTreeMap<String, usize> {
        pairs.iter().map(|(g, n)| ((*g).to_string(), *n)).collect()
    }

    #[test]
    fn none_replays_nothing_and_uniform_is_the_plain_sample() {
        let records = corpus(&[("a", 30), ("b", 10)]);
        assert!(select_replay(&records, &ReplayPolicy::None, fraction(0.5), 7).is_empty());
        assert_eq!(
            select_replay(&records, &ReplayPolicy::Uniform, fraction(0.25), 7),
            replay_sample(&records, fraction(0.25), 7)
        );
    }

    #[test]
    fn a_balanced_replay_gives_every_capability_the_same_share_as_far_as_it_has_records() {
        // Eight of 64 records over three groups: the two extra go to the first two.
        let records = corpus(&[("a", 22), ("b", 21), ("c", 21)]);
        let picked = select_replay(
            &records,
            &ReplayPolicy::CapabilityBalanced,
            fraction(0.125),
            7,
        );
        assert_eq!(
            per_group(&records, &picked),
            tally(&[("a", 3), ("b", 3), ("c", 2)])
        );

        // A group with two records gives what it has; the rest share the remainder.
        let records = corpus(&[("a", 2), ("b", 31), ("c", 31)]);
        let picked = select_replay(
            &records,
            &ReplayPolicy::CapabilityBalanced,
            fraction(0.25),
            7,
        );
        assert_eq!(
            per_group(&records, &picked),
            tally(&[("a", 2), ("b", 7), ("c", 7)])
        );
        assert_eq!(
            picked,
            select_replay(
                &records,
                &ReplayPolicy::CapabilityBalanced,
                fraction(0.25),
                7
            ),
            "the same records, fraction and seed choose the same"
        );
    }

    #[test]
    fn a_failure_targeted_replay_follows_the_weights_and_spares_capabilities_that_do_not_fail() {
        let records = corpus(&[("a", 22), ("b", 21), ("c", 21)]);
        let weights = |pairs: &[(&str, f64)]| ReplayPolicy::FailureTargeted {
            weights: pairs.iter().map(|(g, w)| ((*g).to_string(), *w)).collect(),
        };
        let picked = select_replay(
            &records,
            &weights(&[("a", 3.0), ("b", 1.0)]),
            fraction(0.125),
            7,
        );
        assert_eq!(per_group(&records, &picked), tally(&[("a", 6), ("b", 2)]));

        // No failure anywhere is nothing to target: balanced, not empty.
        let picked = select_replay(&records, &weights(&[]), fraction(0.125), 7);
        assert_eq!(
            per_group(&records, &picked),
            tally(&[("a", 3), ("b", 3), ("c", 2)])
        );

        // Never more than a group has: its surplus goes to the others.
        let few = corpus(&[("a", 3), ("b", 61)]);
        let picked = select_replay(&few, &weights(&[("a", 9.0), ("b", 1.0)]), fraction(0.5), 7);
        assert_eq!(per_group(&few, &picked), tally(&[("a", 3), ("b", 29)]));
    }

    #[test]
    fn a_learning_progress_replay_follows_its_weights_like_any_weighted_replay() {
        let records = corpus(&[("a", 22), ("b", 21), ("c", 21)]);
        let weighted = |weights: &[(&str, f64)]| ReplayPolicy::LearningProgress {
            weights: weights
                .iter()
                .map(|(g, w)| ((*g).to_string(), *w))
                .collect(),
        };
        let picked = select_replay(
            &records,
            &weighted(&[("a", 3.0), ("b", 1.0)]),
            fraction(0.125),
            7,
        );
        assert_eq!(per_group(&records, &picked), tally(&[("a", 6), ("b", 2)]));
        // No signal anywhere is nothing to weight by: balanced, not empty.
        let picked = select_replay(&records, &weighted(&[]), fraction(0.125), 7);
        assert_eq!(
            per_group(&records, &picked),
            tally(&[("a", 3), ("b", 3), ("c", 2)])
        );
    }
}
