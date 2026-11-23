// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training pipelines that read any dataset
// the same way, for its clients. If your team needs expertise in loading,
// sampling and versioning datasets for agent training, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Samplers and the loader that walks a [`Dataset`] by them.
//!
//! The order of an epoch is a function of the sampler's seed and the epoch
//! number and of nothing else (no clock, no thread, no machine), so two
//! processes that name the same seed walk a dataset in the same order and a
//! resumed run continues the order it was in.

use super::{Dataset, Record};

/// An order over the records of a dataset, one per epoch.
pub trait Sampler {
    /// Every index of `0..len` once, in the order of epoch `epoch`.
    fn indices(&self, len: usize, epoch: u64) -> Vec<usize>;
}

/// The records in the order they are stored.
#[derive(Clone, Copy, Debug, Default)]
pub struct SequentialSampler;

impl Sampler for SequentialSampler {
    fn indices(&self, len: usize, _epoch: u64) -> Vec<usize> {
        (0..len).collect()
    }
}

/// The records in a random order that depends on the seed and the epoch.
#[derive(Clone, Copy, Debug)]
pub struct RandomSampler {
    seed: u64,
}

impl RandomSampler {
    /// A sampler drawing by `seed`. A seed is always named: a run states it.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }
}

/// The next value of a SplitMix64 stream.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Sampler for RandomSampler {
    fn indices(&self, len: usize, epoch: u64) -> Vec<usize> {
        let mut order: Vec<usize> = (0..len).collect();
        let mut state = self.seed ^ epoch.wrapping_mul(0xD1B5_4A32_D192_ED03);
        for i in (1..len).rev() {
            // A draw in 0..=i, without modulo bias worth the name for a
            // dataset's length: the 64-bit draw is far larger than `i`.
            let j = (splitmix(&mut state) % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        order
    }
}

/// An order cut into batches of indices.
#[derive(Clone, Copy, Debug)]
pub struct BatchSampler<S> {
    sampler: S,
    batch_size: usize,
    drop_last: bool,
}

impl<S: Sampler> BatchSampler<S> {
    /// Batches of `batch_size` (at least 1) from `sampler`'s order; the last
    /// one, shorter when the length does not divide, is dropped when
    /// `drop_last`.
    #[must_use]
    pub fn new(sampler: S, batch_size: usize, drop_last: bool) -> Self {
        Self {
            sampler,
            batch_size: batch_size.max(1),
            drop_last,
        }
    }

    /// The batches of epoch `epoch` over `len` records.
    #[must_use]
    pub fn batches(&self, len: usize, epoch: u64) -> Vec<Vec<usize>> {
        let order = self.sampler.indices(len, epoch);
        let chunks = order.chunks(self.batch_size);
        if self.drop_last {
            chunks
                .filter(|c| c.len() == self.batch_size)
                .map(<[usize]>::to_vec)
                .collect()
        } else {
            chunks.map(<[usize]>::to_vec).collect()
        }
    }

    /// How many batches an epoch over `len` records has.
    #[must_use]
    pub fn batch_count(&self, len: usize) -> usize {
        if self.drop_last {
            len / self.batch_size
        } else {
            len.div_ceil(self.batch_size)
        }
    }
}

/// Batches of the records of a dataset, in a sampler's order.
#[derive(Debug)]
pub struct DataLoader<'a, R, S> {
    dataset: &'a Dataset<R>,
    sampler: BatchSampler<S>,
}

impl<'a, R: Record, S: Sampler> DataLoader<'a, R, S> {
    /// A loader over `dataset`.
    #[must_use]
    pub fn new(dataset: &'a Dataset<R>, sampler: BatchSampler<S>) -> Self {
        Self { dataset, sampler }
    }

    /// How many batches an epoch has.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sampler.batch_count(self.dataset.len())
    }

    /// Whether an epoch has no batch.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The batches of epoch `epoch`, each the records at that batch's indices.
    pub fn epoch(&self, epoch: u64) -> impl Iterator<Item = Vec<&'a R>> + '_ {
        let dataset = self.dataset;
        self.sampler
            .batches(dataset.len(), epoch)
            .into_iter()
            .map(move |batch| batch.into_iter().filter_map(|i| dataset.get(i)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::Provenance;

    #[derive(Clone)]
    struct Line(String);
    impl Record for Line {
        fn parse(line: &str) -> Result<Self, String> {
            Ok(Self(line.to_owned()))
        }
        fn line(&self) -> &str {
            &self.0
        }
    }

    fn dataset(n: usize) -> Dataset<Line> {
        let provenance = Provenance {
            reference: "memory".into(),
            config: None,
            revision: None,
            file: String::new(),
        };
        Dataset::new(
            "train",
            provenance,
            (0..n).map(|i| Line(i.to_string())).collect(),
        )
    }

    #[test]
    fn a_random_order_is_a_permutation_fixed_by_seed_and_epoch() {
        let sampler = RandomSampler::new(7);
        let first = sampler.indices(50, 0);
        let mut sorted = first.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..50).collect::<Vec<_>>(), "every index once");
        assert_eq!(
            first,
            RandomSampler::new(7).indices(50, 0),
            "same seed, same order"
        );
        assert_ne!(
            first,
            sampler.indices(50, 1),
            "another epoch, another order"
        );
        assert_ne!(
            first,
            RandomSampler::new(8).indices(50, 0),
            "another seed, another order"
        );
        assert_ne!(first, (0..50).collect::<Vec<_>>(), "not the stored order");
    }

    #[test]
    fn batches_cover_the_dataset_once_and_the_last_may_be_dropped() {
        let keep = BatchSampler::new(RandomSampler::new(1), 4, false);
        let batches = keep.batches(10, 0);
        assert_eq!(batches.iter().map(Vec::len).collect::<Vec<_>>(), [4, 4, 2]);
        assert_eq!(keep.batch_count(10), 3);
        let mut all: Vec<usize> = batches.concat();
        all.sort_unstable();
        assert_eq!(all, (0..10).collect::<Vec<_>>());
        let drop = BatchSampler::new(SequentialSampler, 4, true);
        assert_eq!(
            drop.batches(10, 0),
            vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]]
        );
        assert_eq!(drop.batch_count(10), 2);
    }

    #[test]
    fn a_loader_gives_the_records_of_each_batch_in_the_samplers_order() {
        let data = dataset(7);
        let loader = DataLoader::new(&data, BatchSampler::new(SequentialSampler, 3, false));
        assert_eq!(loader.len(), 3);
        let lines: Vec<Vec<&str>> = loader
            .epoch(0)
            .map(|b| b.into_iter().map(Line::line).collect())
            .collect();
        assert_eq!(lines, [vec!["0", "1", "2"], vec!["3", "4", "5"], vec!["6"]]);
        let shuffled = data.shuffle(3);
        assert_eq!(shuffled.len(), 7);
        assert_ne!(shuffled.lines(), data.lines());
        assert_eq!(shuffled.lines(), data.shuffle(3).lines());
    }
}
