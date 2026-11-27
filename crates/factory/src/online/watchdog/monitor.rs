// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training loops that roll a model back only on
// evidence, for its clients. If your team needs expertise in sequential
// testing for continuously trained models, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The held-out tasks the watchdog's evaluations read.
//!
//! The **monitor pool** is held out from training: each look draws a fresh
//! random subsample of it, and a task is drawn at most `max_uses` times per
//! epoch, so no look re-reads a few tasks until their noise looks like a
//! trend. The **confirm set** is sealed: it is read only when a candidate is
//! about to be promoted. Neither reaches the task selector: the selector sees
//! training groups alone.

use std::collections::BTreeMap;

use splinter_core::digest::digest_of;

/// A run's monitor pool.
#[derive(Clone, Debug, PartialEq)]
pub struct MonitorPool {
    tasks: Vec<String>,
    max_uses: u32,
    seed: u64,
    epoch: u64,
    uses: BTreeMap<String, u32>,
    looks: u64,
}

impl MonitorPool {
    /// The pool of `tasks`, each drawn at most `max_uses` times per epoch,
    /// its draws derived from `seed`.
    #[must_use]
    pub fn new(tasks: Vec<String>, max_uses: u32, seed: u64) -> Self {
        Self {
            tasks,
            max_uses,
            seed,
            epoch: 0,
            uses: BTreeMap::new(),
            looks: 0,
        }
    }

    /// A new epoch: every task may be drawn `max_uses` times again.
    pub fn epoch(&mut self, epoch: u64) {
        self.epoch = epoch;
        self.uses.clear();
    }

    /// A fresh subsample of at most `size` tasks with uses left, each
    /// counted as used; fewer (or none) when the pool is spent for the epoch.
    pub fn draw(&mut self, size: usize) -> Vec<String> {
        let look = self.looks;
        self.looks += 1;
        let mut candidates: Vec<(u64, &String)> = self
            .tasks
            .iter()
            .filter(|t| self.uses.get(*t).copied().unwrap_or(0) < self.max_uses)
            .map(|t| {
                (
                    digest_of(&("monitor", self.seed, self.epoch, look, t)).leading_u64(),
                    t,
                )
            })
            .collect();
        candidates.sort();
        let drawn: Vec<String> = candidates
            .into_iter()
            .take(size)
            .map(|(_, t)| t.clone())
            .collect();
        for task in &drawn {
            *self.uses.entry(task.clone()).or_default() += 1;
        }
        drawn
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_look_is_a_fresh_subsample_and_no_task_is_read_past_its_uses_in_an_epoch() {
        let tasks: Vec<String> = (0..10).map(|i| format!("t{i}")).collect();
        // seed: the draws are checked for their counts and caps; any seed serves.
        let mut pool = MonitorPool::new(tasks, 2, 3);
        let first = pool.draw(4);
        let second = pool.draw(4);
        assert_eq!((first.len(), second.len()), (4, 4));
        assert_ne!(first, second, "a fresh subsample per look");
        let mut drawn = first.len() + second.len();
        loop {
            let look = pool.draw(4);
            if look.is_empty() {
                break;
            }
            drawn += look.len();
        }
        assert_eq!(drawn, 20, "ten tasks, two uses each per epoch");
        pool.epoch(1);
        assert_eq!(pool.draw(4).len(), 4, "a new epoch reads them again");
    }
}
