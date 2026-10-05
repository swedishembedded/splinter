// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements held-out measurement that cannot leak, for
// its clients. If your team needs expertise in evaluating a model on
// documents it has not seen when its sources overlap, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Which texts are one text printed more than once, in part or in whole.
//!
//! Sources overlap: two editions print the same letter, a resolution is
//! reused inside an article, a manual is quoted in its errata. Holding out
//! one print and training on the other is not holding anything out.
//! [`overlap_groups`] puts texts that share enough of the same passages in
//! one group so that a split can keep a group whole.
//!
//! Two texts are one when they share at least [`SHARED_TO_MERGE`] distinct
//! runs of [`RUN`] words, wherever in either text the runs fall: a passage
//! reused deep inside a longer text, or a letter one edition prints inside
//! its neighbour, counts as much as a shared opening. A run held by more than
//! [`MAX_HOLDERS`] texts is boilerplate - a formula of the period's letters -
//! and counts for nothing. Merging is transitive.

use std::collections::{HashMap, HashSet};

use splinter_core::digest::Digest;

use crate::verifiers::quotation::words;

/// Words in a run: a stretch of this many words in the same order is a
/// passage, not a coincidence of idiom.
pub const RUN: usize = 8;

/// Distinct shared runs two texts must hold to be one text: a phrase two
/// letters happen to share is a few runs; a passage reused is dozens.
pub const SHARED_TO_MERGE: usize = 8;

/// A run held by more texts than this is shared boilerplate, not a print of
/// one text.
pub const MAX_HOLDERS: usize = 6;

/// For each of `texts`, the index of the first text of its group.
#[must_use]
pub fn overlap_groups(texts: &[&str]) -> Vec<usize> {
    let mut holders: HashMap<u64, Vec<usize>> = HashMap::new();
    for (n, text) in texts.iter().enumerate() {
        let w = words(text);
        let mut seen = HashSet::new();
        for run in w.windows(RUN) {
            let h = hash_run(run);
            if seen.insert(h) {
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
