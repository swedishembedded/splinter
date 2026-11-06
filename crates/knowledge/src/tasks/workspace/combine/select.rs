// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements executable evaluation of code changes
// against a project's own test suite, for its clients. If your team needs
// expertise in test-driven benchmarks for code-editing agents, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Which single tasks are tried together, in which order.
//!
//! The choice is a function of the seed and the tasks alone. It spreads the
//! combined tasks evenly: the next task is anchored on the file that takes
//! part in the fewest admitted tasks so far, and filled with the files that
//! have been paired with the chosen ones least, so every file is covered
//! before any is repeated and every pair of files is met about as often as
//! any other. Of a file's single tasks, the one used least so far is taken.

use std::collections::{BTreeMap, BTreeSet};

use splinter_core::family::rank;
use splinter_core::rng::Seed;

use super::single::Single;

/// How many times in a row a tuple already tried may be drawn before the
/// choice gives up: the tasks of the eligible files are exhausted.
const MAX_REDRAWS: usize = 32;

/// The state of the choice.
pub(super) struct Chooser<'a> {
    singles: &'a [Single],
    by_family: BTreeMap<&'a str, Vec<usize>>,
    faults: usize,
    per_family: usize,
    seed: Seed,
    draws: usize,
    taking_part: BTreeMap<&'a str, usize>,
    pairs: BTreeMap<(&'a str, &'a str), usize>,
    used: Vec<usize>,
    tried: BTreeSet<Vec<usize>>,
}

impl<'a> Chooser<'a> {
    /// A choice among `singles`, `faults` of different files at a time, in
    /// which no file takes part in more than `per_family` admitted tasks.
    pub(super) fn new(singles: &'a [Single], faults: usize, per_family: usize, seed: Seed) -> Self {
        let mut by_family: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for (at, single) in singles.iter().enumerate() {
            by_family
                .entry(single.family.as_str())
                .or_default()
                .push(at);
        }
        Self {
            taking_part: by_family.keys().map(|family| (*family, 0)).collect(),
            by_family,
            singles,
            faults,
            per_family,
            seed,
            draws: 0,
            pairs: BTreeMap::new(),
            used: vec![0; singles.len()],
            tried: BTreeSet::new(),
        }
    }

    /// How many families there are to choose among.
    pub(super) fn families(&self) -> usize {
        self.by_family.len()
    }

    /// The indices of the next single tasks to try together, sorted; `None`
    /// when too few files can still take part, or every tuple they offer has
    /// been tried.
    pub(super) fn next(&mut self) -> Option<Vec<usize>> {
        for _ in 0..MAX_REDRAWS {
            self.draws += 1;
            let chosen = self.families_to_try()?;
            let mut tuple: Vec<usize> = chosen
                .iter()
                .map(|family| self.least_used(family))
                .collect();
            tuple.sort_unstable();
            if self.tried.insert(tuple.clone()) {
                for at in &tuple {
                    self.used[*at] += 1;
                }
                return Some(tuple);
            }
        }
        None
    }

    /// Records that the tasks at `tuple` made an admitted task.
    pub(super) fn admitted(&mut self, tuple: &[usize]) {
        let families: Vec<&'a str> = tuple
            .iter()
            .map(|at| self.singles[*at].family.as_str())
            .collect();
        for family in &families {
            *self.taking_part.entry(family).or_default() += 1;
        }
        for (at, first) in families.iter().enumerate() {
            for second in &families[at + 1..] {
                *self.pairs.entry(pair(first, second)).or_default() += 1;
            }
        }
    }

    /// The files to try together: the least used as the anchor, then those
    /// least paired with the ones chosen so far.
    fn families_to_try(&self) -> Option<Vec<&'a str>> {
        let eligible: Vec<&'a str> = self
            .taking_part
            .iter()
            .filter(|(_, n)| **n < self.per_family)
            .map(|(family, _)| *family)
            .collect();
        if eligible.len() < self.faults {
            return None;
        }
        let anchor = eligible
            .iter()
            .min_by_key(|family| (self.taking_part[**family], self.rank("anchor", family)))?;
        let mut chosen = vec![*anchor];
        while chosen.len() < self.faults {
            let next = eligible
                .iter()
                .filter(|f| !chosen.contains(f))
                .min_by_key(|family| {
                    let met: usize = chosen
                        .iter()
                        .map(|c| self.pairs.get(&pair(c, family)).copied().unwrap_or(0))
                        .sum();
                    (met, self.taking_part[**family], self.rank("fill", family))
                })?;
            chosen.push(*next);
        }
        Some(chosen)
    }

    /// Of `family`'s single tasks, the one used least so far.
    fn least_used(&self, family: &str) -> usize {
        self.by_family[family]
            .iter()
            .copied()
            .min_by_key(|at| {
                (
                    self.used[*at],
                    self.rank("single", self.singles[*at].id.hex()),
                )
            })
            .unwrap_or(0)
    }

    fn rank(&self, what: &str, name: &str) -> String {
        rank(&format!("{what}|{}|{name}", self.draws), self.seed)
    }
}

/// The two names in order, as the key of a pair.
fn pair<'a>(first: &'a str, second: &'a str) -> (&'a str, &'a str) {
    if first <= second {
        (first, second)
    } else {
        (second, first)
    }
}
