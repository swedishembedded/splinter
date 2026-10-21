// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for leakage-free evaluation of
// models trained on longitudinal data, for its clients. If your team needs
// expertise in train/test hygiene, temporal validation or external
// validation designs, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Content overlap between the parts of a split.
//!
//! Records that share no group can still share content: a change and its
//! revert, a cherry-pick, a copy. The caller extracts, from each record, a set
//! of features normalised so that such records share them; [`content_overlaps`]
//! finds the pairs in different parts whose Jaccard similarity reaches an
//! [`OverlapRule`], [`DataSplit::verify_no_overlap`] refuses a split that has
//! any, and [`DataSplit::without_overlaps`] leaves out the record of each pair
//! that is nearer the training side.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::split::{DataSplit, LeakageError, Part};

/// How alike two records of different parts must be to count as repeating
/// each other.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OverlapRule {
    /// The Jaccard similarity of two records' features at or above which they
    /// overlap: 1.0 is an exact duplicate.
    pub min_similarity: f64,
    /// A record with fewer features than this is too small to be compared
    /// (a handful of common lines would match anything).
    pub min_features: usize,
}

/// Two records of different parts whose content overlaps.
#[derive(Clone, Debug, PartialEq)]
pub struct Overlap {
    /// The record of the part nearer the training side (train, then
    /// validation).
    pub a: String,
    /// The record of the part further from it.
    pub b: String,
    /// The part of `a`.
    pub part_a: Part,
    /// The part of `b`.
    pub part_b: Part,
    /// Their Jaccard similarity.
    pub similarity: f64,
}

/// The pairs of records in different parts of `split` whose `features` (what
/// the caller extracts from the record's content, normalised so that a change
/// and its revert or copy share features) are at least as alike as `rule`
/// says. A record with no entry in `features` is not compared. The result is
/// ordered by record ids.
#[must_use]
pub fn content_overlaps(
    split: &DataSplit,
    features: &BTreeMap<String, BTreeSet<String>>,
    rule: &OverlapRule,
) -> Vec<Overlap> {
    let part_of = split.part_of();
    let comparable: Vec<(&str, &BTreeSet<String>)> = features
        .iter()
        .filter(|(id, set)| part_of.contains_key(id.as_str()) && set.len() >= rule.min_features)
        .map(|(id, set)| (id.as_str(), set))
        .collect();
    // Only records that share a feature can be alike: index them by it.
    let mut by_feature: HashMap<&str, Vec<usize>> = HashMap::new();
    for (at, (_, set)) in comparable.iter().enumerate() {
        for feature in *set {
            by_feature.entry(feature.as_str()).or_default().push(at);
        }
    }
    let mut shared: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for holders in by_feature.values() {
        for (i, a) in holders.iter().enumerate() {
            for b in &holders[i + 1..] {
                *shared.entry((*a, *b)).or_default() += 1;
            }
        }
    }
    let mut found = Vec::new();
    for ((i, j), common) in shared {
        let (id_i, set_i) = comparable[i];
        let (id_j, set_j) = comparable[j];
        let (part_i, part_j) = (part_of[id_i], part_of[id_j]);
        if part_i == part_j {
            continue;
        }
        let similarity = common as f64 / (set_i.len() + set_j.len() - common) as f64;
        if similarity < rule.min_similarity {
            continue;
        }
        let ((a, part_a), (b, part_b)) = if part_i < part_j {
            ((id_i, part_i), (id_j, part_j))
        } else {
            ((id_j, part_j), (id_i, part_i))
        };
        found.push(Overlap {
            a: a.to_owned(),
            b: b.to_owned(),
            part_a,
            part_b,
            similarity,
        });
    }
    found.sort_by(|x, y| (&x.a, &x.b).cmp(&(&y.a, &y.b)));
    found
}

impl DataSplit {
    /// Refuses a split in which records of different parts overlap in
    /// content under `rule`; the pairs are [`content_overlaps`].
    pub fn verify_no_overlap(
        &self,
        features: &BTreeMap<String, BTreeSet<String>>,
        rule: &OverlapRule,
    ) -> Result<(), LeakageError> {
        match content_overlaps(self, features, rule).len() {
            0 => Ok(()),
            pairs => Err(LeakageError::ContentOverlap { pairs }),
        }
    }

    /// This split without the record of each of `overlaps` that is nearer the
    /// training side (the held-out record is the one that stays), counted
    /// under `content_overlap`.
    #[must_use]
    pub fn without_overlaps(&self, overlaps: &[Overlap]) -> DataSplit {
        let dropped: HashSet<&str> = overlaps.iter().map(|o| o.a.as_str()).collect();
        let mut split = self.clone();
        for ids in split.parts.values_mut() {
            ids.retain(|id| !dropped.contains(id.as_str()));
        }
        if !dropped.is_empty() {
            *split
                .excluded
                .entry("content_overlap".to_owned())
                .or_default() += dropped.len();
        }
        split
    }
}
