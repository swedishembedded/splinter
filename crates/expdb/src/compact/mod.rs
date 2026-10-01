// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Background compaction: many small immutable files become few large ones.
//!
//! Compaction only ever adds the merged file and publishes a manifest that
//! removes its inputs. Record ids never change, so everything that points at
//! a record keeps working; a reader on an older snapshot keeps its inputs for
//! as long as it is pinned; and compactors, writers and other compactors need
//! no coordination. If two compactors overlap, a record may briefly sit in two
//! files, which readers fold together and the next compaction removes.

mod blobs;
mod segments;

/// What a compaction did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompactReport {
    /// Groups of small files that were merged.
    pub groups: usize,
    /// Files consumed.
    pub inputs: usize,
    /// The files written, by content id.
    pub outputs: Vec<crate::id::ContentId>,
}

/// Groups files, smallest first, into runs whose total stays within `target`.
/// Files already at or above the target are not touched, and a run of one is
/// not worth a rewrite.
pub(crate) fn group_by_size<T: Clone>(items: &[(u64, T)], target: u64) -> Vec<Vec<T>> {
    let mut sorted: Vec<&(u64, T)> = items.iter().filter(|(bytes, _)| *bytes < target).collect();
    sorted.sort_by_key(|(bytes, _)| *bytes);
    let mut groups: Vec<Vec<T>> = Vec::new();
    let (mut current, mut total) = (Vec::new(), 0u64);
    for (bytes, item) in sorted {
        if !current.is_empty() && total + bytes > target {
            groups.push(std::mem::take(&mut current));
            total = 0;
        }
        current.push(item.clone());
        total += bytes;
    }
    groups.push(current);
    groups.retain(|g| g.len() >= 2);
    groups
}
