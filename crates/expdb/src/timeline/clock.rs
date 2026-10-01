// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Resolving one clock to another through whatever mappings are known.

use std::collections::{HashMap, VecDeque};

use crate::error::Result;
use crate::id::ContentId;
use crate::manifest::Snapshot;

/// A conversion between two clocks, as one straight line with an
/// uncertainty: `destination = dst_ref + slope * (source - src_ref)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClockTransform {
    /// A source time the line passes through.
    pub src_ref_ns: i64,
    /// The destination time at that source time.
    pub dst_ref_ns: i64,
    /// Destination seconds per source second.
    pub slope: f64,
    /// How far the conversion may be off, in nanoseconds.
    pub uncertainty_ns: u64,
}

impl ClockTransform {
    /// The conversion from a clock to itself.
    pub fn identity() -> Self {
        Self {
            src_ref_ns: 0,
            dst_ref_ns: 0,
            slope: 1.0,
            uncertainty_ns: 0,
        }
    }

    /// A source time on the destination clock.
    pub fn apply(&self, source_ns: i64) -> i64 {
        self.dst_ref_ns + (self.slope * (source_ns - self.src_ref_ns) as f64).round() as i64
    }

    /// The conversion the other way.
    pub fn invert(&self) -> Self {
        Self {
            src_ref_ns: self.dst_ref_ns,
            dst_ref_ns: self.src_ref_ns,
            slope: 1.0 / self.slope,
            uncertainty_ns: self.uncertainty_ns,
        }
    }

    /// This conversion followed by `next`; uncertainties add.
    pub fn then(&self, next: &ClockTransform) -> Self {
        Self {
            src_ref_ns: self.src_ref_ns,
            dst_ref_ns: next.apply(self.dst_ref_ns),
            slope: self.slope * next.slope,
            uncertainty_ns: self.uncertainty_ns.saturating_add(next.uncertainty_ns),
        }
    }
}

impl Snapshot {
    /// How to read a time on clock `from` as a time on clock `to`, through the
    /// shortest chain of recorded mappings, using the most certain mapping
    /// between each pair. `None` if the clocks are not connected.
    pub fn resolve_clock(&self, from: ContentId, to: ContentId) -> Result<Option<ClockTransform>> {
        if from == to {
            return Ok(Some(ClockTransform::identity()));
        }
        let data = self.timeline()?;
        let mut edges: HashMap<(ContentId, ContentId), ClockTransform> = HashMap::new();
        for m in &data.mappings {
            let forward = ClockTransform {
                src_ref_ns: m.src_ref_ns,
                dst_ref_ns: m.dst_ref_ns,
                slope: m.slope,
                uncertainty_ns: m.uncertainty_ns,
            };
            for (pair, transform) in [
                ((m.source, m.destination), forward),
                ((m.destination, m.source), forward.invert()),
            ] {
                let better = edges
                    .get(&pair)
                    .is_none_or(|known| transform.uncertainty_ns < known.uncertainty_ns);
                if better {
                    edges.insert(pair, transform);
                }
            }
        }
        let mut reached: HashMap<ContentId, ClockTransform> =
            HashMap::from([(from, ClockTransform::identity())]);
        let mut queue = VecDeque::from([from]);
        while let Some(at) = queue.pop_front() {
            let so_far = reached[&at];
            let mut next: Vec<_> = edges.iter().filter(|((a, _), _)| *a == at).collect();
            next.sort_by_key(|((_, b), _)| *b);
            for ((_, b), step) in next {
                if !reached.contains_key(b) {
                    reached.insert(*b, so_far.then(step));
                    queue.push_back(*b);
                }
            }
        }
        Ok(reached.get(&to).copied())
    }
}
