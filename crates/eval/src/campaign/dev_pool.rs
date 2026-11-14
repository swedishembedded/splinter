// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for self-improvement loops whose
// validation data stays honest while it is reused, for its clients. If your
// team needs expertise in validation-set design and adaptive data reuse you
// can procure our services by sending an email to info@swedishembedded.com.

//! The learner-side dev pool of the evidence gate: each round draws its dev
//! items uniformly without replacement from a pool larger than the draw, and
//! every item a round grades is remembered as used, so the next round knows
//! how exposed each of its draws is.
//!
//! The exposure probability of draw `i` is the share of the items it could be
//! that earlier rounds already used, `zeta_i = |intersection(I, R_{i-1})| / |R_{i-1}|`
//! (equation 9 of TRUST, arXiv 2611.10787v1), where `R_{i-1}` is the pool
//! minus the round's first `i - 1` draws and `I` the used set. It is fixed
//! before the item is seen, as the e-process requires. Items are returned to
//! the pool after the round; only the used set grows.
//!
//! The pool holds learner data only: ids of the campaign's own dev tasks,
//! never an examiner's. This module draws and accounts; the caller keeps the
//! used set durably (it is the union of the items its decision records say
//! were spent).

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use splinter_core::rng::SplitMix64;

use super::evidence::EvidenceError;

/// One dev item a round drew, in draw order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Drawn {
    /// The dev task id.
    pub task: String,
    /// The probability, before the draw, that it would be an item an earlier
    /// round used.
    pub zeta: f64,
    /// Whether it is one.
    pub reused: bool,
}

/// The pool and what earlier rounds used of it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DevPool {
    pool: Vec<String>,
    used: BTreeSet<String>,
}

impl DevPool {
    /// A pool of `tasks` (deduplicated, in a canonical order so a seed draws
    /// the same items however the caller listed them) of which `used` were
    /// graded by earlier rounds; used ids outside the pool are dropped.
    #[must_use]
    pub fn new(tasks: impl IntoIterator<Item = String>, used: &BTreeSet<String>) -> Self {
        let pool: BTreeSet<String> = tasks.into_iter().collect();
        let used = used.intersection(&pool).cloned().collect();
        Self {
            pool: pool.into_iter().collect(),
            used,
        }
    }

    /// Items in the pool.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pool.len()
    }

    /// Whether the pool is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pool.is_empty()
    }

    /// Items earlier rounds used.
    #[must_use]
    pub fn used(&self) -> &BTreeSet<String> {
        &self.used
    }

    /// Draws `n` items uniformly without replacement, each with its exposure
    /// probability.
    ///
    /// # Errors
    /// The draw is empty or not smaller than the pool: a round that draws
    /// the whole pool has no fresh item from its second round on.
    pub fn draw(&self, n: usize, rng: &mut SplitMix64) -> Result<Vec<Drawn>, EvidenceError> {
        if n == 0 || n >= self.pool.len() {
            return Err(EvidenceError::Parameter(format!(
                "a round draws {n} of a dev pool of {}; it must draw at least one and fewer than the pool holds",
                self.pool.len()
            )));
        }
        let mut order: Vec<usize> = (0..self.pool.len()).collect();
        let mut used_left = self.used.len();
        let mut drawn = Vec::with_capacity(n);
        for i in 0..n {
            let remaining = self.pool.len() - i;
            let zeta = used_left as f64 / remaining as f64;
            let j = i + rng.below(remaining);
            order.swap(i, j);
            let task = &self.pool[order[i]];
            let reused = self.used.contains(task);
            used_left -= usize::from(reused);
            drawn.push(Drawn {
                task: task.clone(),
                zeta,
                reused,
            });
        }
        Ok(drawn)
    }

    /// Marks `tasks` used: what a round graded, whether or not its evidence
    /// stopped before them (grading exposes an item).
    pub fn spend<'a>(&mut self, tasks: impl IntoIterator<Item = &'a String>) {
        for task in tasks {
            if self.pool.binary_search(task).is_ok() {
                self.used.insert(task.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A seed no test depends on, printed so a failure can be replayed.
    fn drawn_seed() -> u64 {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);
        println!("seed {seed}");
        seed
    }

    fn ids(range: std::ops::Range<usize>) -> Vec<String> {
        range.map(|i| format!("dev-{i:03}")).collect()
    }

    #[test]
    fn a_draw_is_without_replacement_from_the_pool_and_fresh_pools_are_unexposed() {
        let pool = DevPool::new(ids(0..50), &BTreeSet::new());
        let drawn = pool.draw(20, &mut SplitMix64::new(drawn_seed())).unwrap();
        let distinct: BTreeSet<&String> = drawn.iter().map(|d| &d.task).collect();
        assert_eq!(distinct.len(), 20);
        assert!(drawn.iter().all(|d| d.zeta == 0.0 && !d.reused));
        assert!(pool.draw(50, &mut SplitMix64::new(drawn_seed())).is_err());
        assert!(pool.draw(0, &mut SplitMix64::new(drawn_seed())).is_err());
    }

    #[test]
    fn zeta_is_the_used_share_of_what_the_draw_could_still_be() {
        let used: BTreeSet<String> = ids(0..10).into_iter().collect();
        let pool = DevPool::new(ids(0..40), &used);
        let drawn = pool.draw(39, &mut SplitMix64::new(drawn_seed())).unwrap();
        assert!((drawn[0].zeta - 10.0 / 40.0).abs() < 1e-12);
        let mut used_left = 10usize;
        for (i, d) in drawn.iter().enumerate() {
            assert!((d.zeta - used_left as f64 / (40 - i) as f64).abs() < 1e-12);
            assert_eq!(d.reused, used.contains(&d.task));
            used_left -= usize::from(d.reused);
        }
    }

    #[test]
    fn spending_grows_the_used_set_only_with_pool_items_and_the_order_is_canonical() {
        let mut pool = DevPool::new(ids(0..30), &BTreeSet::new());
        let seed = drawn_seed();
        let drawn = pool.draw(10, &mut SplitMix64::new(seed)).unwrap();
        let tasks: Vec<String> = drawn.iter().map(|d| d.task.clone()).collect();
        pool.spend(tasks.iter().take(6));
        pool.spend(&["elsewhere".to_string()]);
        assert_eq!(pool.used().len(), 6);
        let next = pool.draw(29, &mut SplitMix64::new(seed)).unwrap();
        assert!((next[0].zeta - 6.0 / 30.0).abs() < 1e-12);
        let mut reversed = ids(0..30);
        reversed.reverse();
        let same = DevPool::new(reversed, &BTreeSet::new());
        assert_eq!(
            same.draw(10, &mut SplitMix64::new(seed)).unwrap(),
            DevPool::new(ids(0..30), &BTreeSet::new())
                .draw(10, &mut SplitMix64::new(seed))
                .unwrap(),
            "a seed draws the same items however the pool was listed"
        );
    }
}
