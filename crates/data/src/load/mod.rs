// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training pipelines that read any dataset
// the same way, for its clients. If your team needs expertise in loading,
// sampling and versioning datasets for agent training, you can procure our
// services by sending an email to info@swedishembedded.com.

//! One way to load a dataset, whatever it is and wherever it lives.
//!
//! ```ignore
//! let dataset = load_dataset::<ChatRecord>("swedishembedded/vim-headless-mastery-sft", None, &options)?;
//! let train = dataset.get("train").unwrap();
//! let sampler = BatchSampler::new(RandomSampler::new(17), 32, false);
//! for batch in DataLoader::new(train, sampler).epoch(0) { /* one Vec<&ChatRecord> */ }
//! ```
//!
//! [`load_dataset`] takes a reference and an optional configuration and gives
//! a [`DatasetDict`]: the dataset's splits by name, each a [`Dataset`] of
//! typed records with random access. The reference is a directory of
//! `SPLIT.jsonl` files (at its root, or under `data/`, or under a
//! configuration's directory), a dataset product a factory run wrote, or a
//! repository of the Hugging Face Hub (`owner/name`, or `hf:owner/name@rev`),
//! which is resolved to a commit, downloaded into a cache and read from
//! there. Every [`Dataset`] says where it came from and, for a hub dataset,
//! the commit, so what a run trained on is never a moving name.
//!
//! The record type is the caller's: anything that can be read from one JSON
//! line ([`Record`]). [`ChatRecord`] is the supervised chat record the
//! trainers read. [`DataLoader`], [`BatchSampler`], [`RandomSampler`] and
//! [`SequentialSampler`] give batches of records in an order that is a
//! function of the seed and the epoch, as the loaders of the Python `datasets`
//! and `torch` libraries do, so the same dataset is walked the same way by
//! anything that names the same seed.

mod chat;
mod sampler;
mod source;

pub use chat::ChatRecord;
pub use sampler::{BatchSampler, DataLoader, RandomSampler, Sampler, SequentialSampler};
pub use source::{load_dataset, read_records, LoadError, LoadOptions, Reference};

use std::collections::BTreeMap;

/// What a dataset's lines are read as.
pub trait Record: Sized {
    /// The record of one line of a split file.
    ///
    /// # Errors
    /// Why the line is not a record of this kind.
    fn parse(line: &str) -> Result<Self, String>;

    /// The line the record was read from, byte for byte.
    fn line(&self) -> &str;
}

/// Where a split came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Provenance {
    /// The reference the caller gave.
    pub reference: String,
    /// The configuration asked for, if any.
    pub config: Option<String>,
    /// The commit a hub dataset was read at: never a branch or a tag.
    pub revision: Option<String>,
    /// The file the records were read from.
    pub file: String,
}

/// One split of a dataset: records with random access.
#[derive(Clone, Debug)]
pub struct Dataset<R> {
    split: String,
    provenance: Provenance,
    records: Vec<R>,
}

impl<R: Record> Dataset<R> {
    /// A split of `records`.
    #[must_use]
    pub fn new(split: impl Into<String>, provenance: Provenance, records: Vec<R>) -> Self {
        Self {
            split: split.into(),
            provenance,
            records,
        }
    }

    /// The split's name (`train`, `validation`).
    #[must_use]
    pub fn split(&self) -> &str {
        &self.split
    }

    /// Where the records came from.
    #[must_use]
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    /// How many records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether there is no record.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// The record at `index`.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&R> {
        self.records.get(index)
    }

    /// Every record, in order.
    pub fn iter(&self) -> std::slice::Iter<'_, R> {
        self.records.iter()
    }

    /// The records at `indices`, as a dataset of their own.
    #[must_use]
    pub fn select(&self, indices: &[usize]) -> Self
    where
        R: Clone,
    {
        Self {
            split: self.split.clone(),
            provenance: self.provenance.clone(),
            records: indices
                .iter()
                .filter_map(|&i| self.records.get(i).cloned())
                .collect(),
        }
    }

    /// The records in an order that is a function of `seed` alone.
    #[must_use]
    pub fn shuffle(&self, seed: u64) -> Self
    where
        R: Clone,
    {
        self.select(&RandomSampler::new(seed).indices(self.len(), 0))
    }

    /// The records' lines, in order, byte for byte.
    #[must_use]
    pub fn lines(&self) -> Vec<&str> {
        self.records.iter().map(Record::line).collect()
    }
}

impl<'a, R: Record> IntoIterator for &'a Dataset<R> {
    type Item = &'a R;
    type IntoIter = std::slice::Iter<'a, R>;
    fn into_iter(self) -> Self::IntoIter {
        self.records.iter()
    }
}

/// The splits of a dataset, by name.
#[derive(Clone, Debug)]
pub struct DatasetDict<R> {
    splits: BTreeMap<String, Dataset<R>>,
}

impl<R: Record> DatasetDict<R> {
    /// A dict of `splits`.
    #[must_use]
    pub fn new(splits: BTreeMap<String, Dataset<R>>) -> Self {
        Self { splits }
    }

    /// The names of the splits.
    pub fn split_names(&self) -> impl Iterator<Item = &str> {
        self.splits.keys().map(String::as_str)
    }

    /// The split called `name`; `eval`, `val` and `dev` name `validation`.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Dataset<R>> {
        self.splits.get(canonical_split(name))
    }

    /// The split called `name`, taken out of the dict.
    pub fn take(&mut self, name: &str) -> Option<Dataset<R>> {
        self.splits.remove(canonical_split(name))
    }
}

/// The name a split is kept under: the names a validation split goes by are
/// one.
#[must_use]
pub fn canonical_split(name: &str) -> &str {
    match name {
        "eval" | "val" | "dev" => "validation",
        other => other,
    }
}
