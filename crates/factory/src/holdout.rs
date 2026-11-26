// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements held-out splits that keep whole tasks out of
// training, reproducible from a seed, for its clients. If your team needs
// expertise in evaluation design for learning systems, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Which groups (tasks, families, documents) a split holds out: a share of a
//! set drawn by rank ([`held_out`]), or each item by its own draw
//! ([`held_out_by_id`]), which no other item changes.

use std::collections::BTreeSet;

use splinter_core::digest::{digest_of, Digest};

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

/// Whether `item` is held out at `percent` by its own draw, a hash of
/// `seed` and `item`: the same seed and percent put an item on the same side
/// whatever else is drawn beside it, so two stages that name them take
/// complementary sides of one split.
pub(crate) fn held_out_by_id(seed: u64, item: &str, percent: u32) -> bool {
    let digest = Digest::of(format!("{seed}\0{item}").as_bytes());
    let draw = u64::from_str_radix(&digest.hex()[..16], 16).unwrap_or(0);
    draw % 100 < u64::from(percent)
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
