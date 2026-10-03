// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements held-out measurement that cannot leak, for
// its clients. If your team needs expertise in evaluating a model on
// documents it has not seen when its sources overlap, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Which texts are one text printed more than once.
//!
//! Sources overlap: two editions print the same letter, a manual is quoted in
//! its errata. Holding out one print and training on the other is not holding
//! anything out. [`overlap_groups`] puts texts that share enough of the same
//! passages in one group so that a split can keep a group whole.
//!
//! Two texts are the same print when they share at least [`SHARED_TO_MERGE`]
//! of the sampled runs of [`RUN`] words (every [`SAMPLE_EVERY`]th run by hash,
//! over the first [`WINDOW`] words), and a run held by more than
//! [`MAX_HOLDERS`] texts is boilerplate and counts for nothing. Merging is
//! transitive.

use std::collections::{HashMap, HashSet};

use splinter_record::digest::Digest;

use crate::verifiers::quotation::words;

/// Words in a run.
const RUN: usize = 8;

/// One run in this many is sampled, by its hash.
const SAMPLE_EVERY: u64 = 8;

/// Words of a text the runs are taken from.
const WINDOW: usize = 700;

/// Sampled runs two texts must share to be one text.
const SHARED_TO_MERGE: usize = 3;

/// A run held by more texts than this is shared boilerplate, not a print of
/// one text.
const MAX_HOLDERS: usize = 6;

/// For each of `texts`, the index of the first text of its group.
#[must_use]
pub fn overlap_groups(texts: &[&str]) -> Vec<usize> {
    let mut holders: HashMap<u64, Vec<usize>> = HashMap::new();
    for (n, text) in texts.iter().enumerate() {
        let w = words(text);
        let limit = w.len().saturating_sub(RUN).min(WINDOW);
        let mut seen = HashSet::new();
        for at in 0..limit {
            let h = hash_run(&w[at..at + RUN]);
            if h.is_multiple_of(SAMPLE_EVERY) && seen.insert(h) {
                holders.entry(h).or_default().push(n);
            }
        }
    }

    let mut shared: HashMap<(usize, usize), usize> = HashMap::new();
    for group in holders.values().filter(|g| g.len() <= MAX_HOLDERS) {
        for (i, &a) in group.iter().enumerate() {
            for &b in &group[i + 1..] {
                *shared.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
    }

    let mut parent: Vec<usize> = (0..texts.len()).collect();
    for (&(a, b), &count) in &shared {
        if count >= SHARED_TO_MERGE {
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            if ra != rb {
                parent[ra.max(rb)] = ra.min(rb);
            }
        }
    }
    (0..texts.len()).map(|n| find(&mut parent, n)).collect()
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// A hash of `run` that does not change between runs, platforms or versions.
fn hash_run(run: &[String]) -> u64 {
    let digest = Digest::of(run.join(" ").as_bytes());
    let hex = digest.as_str().rsplit(':').next().unwrap_or_default();
    u64::from_str_radix(&hex[..16.min(hex.len())], 16).unwrap_or_default()
}
