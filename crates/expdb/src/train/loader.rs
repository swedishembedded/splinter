// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Streaming a plan to a trainer: a seeded random order, fetched a window at
//! a time so shuffled samples do not mean scattered reads.

use std::collections::VecDeque;

use super::export::Materialized;
use super::plan::TrainingPlan;
use super::rng::Rng;
use crate::error::{Error, Result};
use crate::manifest::Snapshot;

/// How a loader walks a plan.
#[derive(Debug, Clone, Copy)]
pub struct LoaderConfig {
    /// Samples fetched together. Within a window the blocks holding them are
    /// read once, in storage order, then the samples are handed out in the
    /// logical order.
    pub window: usize,
    /// Fixes the order.
    pub seed: u64,
    /// Which pass over the plan; each epoch has its own order.
    pub epoch: u64,
}

/// An iterator over a plan's samples, materialised.
pub struct Loader<'a> {
    snapshot: &'a Snapshot,
    plan: &'a TrainingPlan,
    order: Vec<usize>,
    window: usize,
    next: usize,
    ready: VecDeque<Result<Materialized>>,
    refused: bool,
}

impl TrainingPlan {
    /// A loader over the plan, reading from `snapshot`.
    pub fn loader<'a>(&'a self, snapshot: &'a Snapshot, config: LoaderConfig) -> Loader<'a> {
        let mut order: Vec<usize> = (0..self.samples.len()).collect();
        Rng::new(config.seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ config.epoch)
            .shuffle(&mut order);
        Loader {
            snapshot,
            plan: self,
            order,
            window: config.window,
            next: 0,
            ready: VecDeque::new(),
            refused: false,
        }
    }
}

impl Iterator for Loader<'_> {
    type Item = Result<Materialized>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.window == 0 {
            return (!std::mem::replace(&mut self.refused, true)).then(|| {
                Err(Error::invalid(
                    "loader window",
                    "it must be at least one sample",
                ))
            });
        }
        if self.ready.is_empty() && self.next < self.order.len() {
            let end = (self.next + self.window).min(self.order.len());
            let window = &self.order[self.next..end];
            self.next = end;
            let provenance: Vec<_> = window
                .iter()
                .map(|i| self.plan.samples[*i].provenance)
                .collect();
            if let Err(e) = self.snapshot.prefetch(&provenance) {
                self.ready.push_back(Err(e));
            }
            for i in window {
                self.ready
                    .push_back(self.snapshot.materialize(&self.plan.samples[*i]));
            }
        }
        self.ready.pop_front()
    }
}
