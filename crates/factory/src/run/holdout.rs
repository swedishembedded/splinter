// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements held-out splits that keep whole tasks out of
// training, reproducible from a seed, for its clients. If your team needs
// expertise in evaluation design for learning systems, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Which groups (tasks, families, documents) a split holds out.

use std::collections::BTreeSet;

use splinter_core::digest::digest_of;

/// The groups held out: `percent` of them, drawn by `seed`, never all.
pub(crate) fn held_out(groups: &BTreeSet<String>, percent: u32, seed: u64) -> BTreeSet<String> {
    let mut drawn: Vec<(u64, &String)> = groups
        .iter()
        .map(|g| (digest_of(&(seed, g)).leading_u64(), g))
        .collect();
    drawn.sort();
    let wanted = (groups.len() * percent as usize)
        .div_ceil(100)
        .min(groups.len().saturating_sub(1));
    drawn
        .into_iter()
        .take(wanted)
        .map(|(_, g)| g.clone())
        .collect()
}

#[cfg(test)]
mod held_out_spec {
    use super::*;

    fn groups(n: usize) -> BTreeSet<String> {
        (0..n).map(|i| format!("g{i}")).collect()
    }

    /// `percent` of the groups, rounded up, are held out, but never every
    /// one: a split with no training group trains on nothing.
    #[test]
    fn the_share_is_rounded_up_and_never_takes_every_group() {
        assert_eq!(held_out(&groups(10), 25, 1).len(), 3);
        assert_eq!(held_out(&groups(2), 99, 1).len(), 1);
        assert!(held_out(&groups(1), 50, 1).is_empty());
        assert_eq!(held_out(&groups(10), 25, 1), held_out(&groups(10), 25, 1));
    }
}
