// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: a seed means the same stream on every platform and in every build,
//! because task generators, partitions and synthetic authors are all drawn
//! from it and what they produce is pinned by digests and by tests. The stream
//! is the published SplitMix64 one.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use splinter_core::rng::SplitMix64;

#[test]
fn the_stream_is_splitmix64_as_published() {
    let mut rng = SplitMix64::new(0);
    assert_eq!(rng.next_u64(), 0xE220_A839_7B1D_CDAF);
    assert_eq!(rng.next_u64(), 0x6E78_9E6A_A1B9_65F4);
    assert_eq!(rng.next_u64(), 0x06C4_5D18_8009_454F);
}

#[test]
fn a_bounded_draw_stays_in_its_range() {
    let mut rng = SplitMix64::new(11);
    for n in 1..50 {
        assert!(rng.below(n) < n);
    }
    assert_eq!(rng.below(0), 0, "an empty range is treated as one choice");
    for _ in 0..200 {
        let v = rng.between(6, 24);
        assert!((6..=24).contains(&v));
    }
    assert_eq!(rng.between(5, 5), 5);
}

#[test]
fn a_shuffle_keeps_the_items_and_depends_on_the_seed() {
    let items: Vec<u32> = (0..30).collect();
    let (mut a, mut b, mut c) = (items.clone(), items.clone(), items.clone());
    SplitMix64::new(1).shuffle(&mut a);
    SplitMix64::new(1).shuffle(&mut b);
    SplitMix64::new(2).shuffle(&mut c);
    assert_eq!(a, b);
    assert_ne!(a, c);
    let mut sorted = a.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, items);
}

#[test]
fn a_pick_is_one_of_the_items() {
    let items = ["a", "b", "c"];
    let mut rng = SplitMix64::new(3);
    for _ in 0..20 {
        assert!(items.contains(rng.pick(&items).unwrap()));
    }
    assert!(rng.pick::<u8>(&[]).is_none());
}
