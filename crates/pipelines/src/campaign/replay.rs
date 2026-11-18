// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that keep a continuously trained
// model from forgetting what it learned in earlier generations, for its
// clients. If your team needs expertise in replay buffers for continual
// fine-tuning you can procure our services by sending an email to
// info@swedishembedded.com.

//! The bounded replay of earlier arms' verified data.
//!
//! An arm trains on fresh prompts, and what it learns from them can cost what
//! earlier arms taught (a generation lost 66 and 50 items its own probes could
//! not see). The buffer holds the verified completions each accepted arm
//! contributed, bounded in records: past the capacity the oldest arm's go
//! first. A draw takes only from arms before the current one, uniformly,
//! so replay is always of earlier generations' data and never an echo of the
//! batch it accompanies.

use serde::{Deserialize, Serialize};
use splinter_core::rng::SplitMix64;
use splinter_data::Record;

/// One accepted arm's contribution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Contribution {
    generation: u32,
    records: Vec<Record>,
}

/// The records of earlier arms, newest last.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReplayBuffer {
    capacity: usize,
    items: Vec<Contribution>,
}

impl ReplayBuffer {
    /// An empty buffer holding at most `capacity` records.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            items: Vec::new(),
        }
    }

    /// Records held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.iter().map(|c| c.records.len()).sum()
    }

    /// Whether nothing is held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Adds arm `generation`'s records, then drops the oldest arms' (and, if
    /// one arm alone exceeds the capacity, the oldest of its records) until
    /// the buffer fits.
    pub fn push(&mut self, generation: u32, records: Vec<Record>) {
        if records.is_empty() || self.capacity == 0 {
            return;
        }
        self.items.push(Contribution {
            generation,
            records,
        });
        while self.len() > self.capacity {
            let over = self.len() - self.capacity;
            let oldest = &mut self.items[0];
            if oldest.records.len() <= over {
                self.items.remove(0);
            } else {
                oldest.records.drain(..over);
            }
        }
    }

    /// Up to `count` records drawn uniformly, without replacement, from arms
    /// before `current`.
    #[must_use]
    pub fn sample(&self, count: usize, current: u32, rng: &mut SplitMix64) -> Vec<Record> {
        let mut pool: Vec<&Record> = self
            .items
            .iter()
            .filter(|c| c.generation < current)
            .flat_map(|c| c.records.iter())
            .collect();
        rng.shuffle(&mut pool);
        pool.into_iter().take(count).cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use splinter_data::{Objective, RecordBody, RecordMetadata};

    fn record(tag: &str) -> Record {
        Record {
            body: RecordBody::Text { text: tag.into() },
            metadata: RecordMetadata {
                experiences: Vec::new(),
                task: None,
                sources: Vec::new(),
                group: None,
                split: None,
                split_draw: None,
                view: tag.into(),
                objective: Objective::Cpt,
            },
        }
    }

    fn records(tag: &str, n: usize) -> Vec<Record> {
        (0..n).map(|i| record(&format!("{tag}{i}"))).collect()
    }

    fn views(records: &[Record]) -> Vec<String> {
        records.iter().map(|r| r.metadata.view.clone()).collect()
    }

    fn drawn_seed() -> u64 {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);
        println!("seed {seed}");
        seed
    }

    #[test]
    fn the_oldest_arms_go_first_when_the_buffer_is_full() {
        let mut buffer = ReplayBuffer::new(5);
        buffer.push(1, records("a", 3));
        buffer.push(2, records("b", 3));
        assert_eq!(buffer.len(), 5);
        let all = buffer.sample(10, 9, &mut SplitMix64::new(drawn_seed()));
        let mut seen = views(&all);
        seen.sort();
        assert_eq!(seen, ["a1", "a2", "b0", "b1", "b2"], "a0 was dropped");
        buffer.push(3, records("c", 8));
        assert_eq!(buffer.len(), 5, "one arm alone is bounded too");
        assert!(buffer
            .sample(10, 9, &mut SplitMix64::new(drawn_seed()))
            .iter()
            .all(|r| r.metadata.view.starts_with('c')));
    }

    #[test]
    fn a_draw_comes_only_from_earlier_arms_without_repeats() {
        let mut buffer = ReplayBuffer::new(100);
        buffer.push(1, records("a", 4));
        buffer.push(2, records("b", 4));
        let drawn = buffer.sample(6, 2, &mut SplitMix64::new(drawn_seed()));
        assert_eq!(drawn.len(), 4, "only arm 1 is earlier than arm 2");
        assert!(views(&drawn).iter().all(|v| v.starts_with('a')));
        let unique: std::collections::BTreeSet<String> = views(&drawn).into_iter().collect();
        assert_eq!(unique.len(), 4, "drawn without replacement");
        assert!(buffer
            .sample(3, 1, &mut SplitMix64::new(drawn_seed()))
            .is_empty());
    }

    #[test]
    fn a_zero_capacity_buffer_holds_nothing_and_round_trips_through_json() {
        let mut none = ReplayBuffer::new(0);
        none.push(1, records("a", 2));
        assert!(none.is_empty());
        let mut buffer = ReplayBuffer::new(4);
        buffer.push(1, records("a", 2));
        let text = serde_json::to_string(&buffer).unwrap();
        assert_eq!(serde_json::from_str::<ReplayBuffer>(&text).unwrap(), buffer);
    }
}
