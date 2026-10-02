// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements storage engines for machine-learning
// experience data for its clients. If your team needs expertise in
// versioned, content-addressed databases or parallel-filesystem I/O, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: what a learner currently thinks is important changes constantly and
//! is kept apart from the experience, which never changes. Derived caches are
//! keyed by what they were derived from.
#![allow(clippy::unwrap_used)]

mod common;

use common::{attempt, rid, Scratch};
use splinter_expdb::cache::DerivedCache;
use splinter_expdb::model::Outcome;
use splinter_expdb::overlay::ReplayOverlay;
use splinter_expdb::{ContentId, WriterIdentity};

fn segment_bytes(root: &std::path::Path) -> Vec<(std::path::PathBuf, Vec<u8>)> {
    let mut found = Vec::new();
    for dir in std::fs::read_dir(root.join("segments")).unwrap() {
        for file in std::fs::read_dir(dir.unwrap().path()).unwrap() {
            let path = file.unwrap().path();
            found.push((path.clone(), std::fs::read(path).unwrap()));
        }
    }
    found.sort();
    found
}

#[test]
fn priorities_change_without_touching_a_single_byte_of_the_experience() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut c = db
        .collector(&WriterIdentity::new("exp", "job", "node", 0))
        .unwrap();
    let (decisions, _) = attempt(&mut c, 1, "p", 3, Outcome::Pass);
    c.flush().unwrap();
    let before = segment_bytes(scratch.dir.path());

    let mut overlay = ReplayOverlay::load(&db, "trainer-a").unwrap();
    for (n, d) in decisions.iter().enumerate() {
        overlay.set_priority(d.id, 1.0 + n as f64).unwrap();
        overlay.record_sample(d.id, 0.25);
    }
    overlay.save(&db).unwrap();

    assert_eq!(segment_bytes(scratch.dir.path()), before);
}

#[test]
fn an_overlay_survives_a_reload_and_each_trainer_has_its_own() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut a = ReplayOverlay::load(&db, "trainer-a").unwrap();
    a.set_priority(rid(1, 1), 4.0).unwrap();
    a.record_sample(rid(1, 1), 0.5);
    a.save(&db).unwrap();

    let again = ReplayOverlay::load(&db, "trainer-a").unwrap();
    assert_eq!(again.priority(rid(1, 1)), Some(4.0));
    assert_eq!(again.times_sampled(rid(1, 1)), 1);
    assert_eq!(again.td_error(rid(1, 1)), Some(0.5));
    assert_eq!(
        ReplayOverlay::load(&db, "trainer-b")
            .unwrap()
            .priority(rid(1, 1)),
        None
    );
}

#[test]
fn prioritised_sampling_favours_high_priority_and_is_fixed_by_the_seed() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let mut overlay = ReplayOverlay::load(&db, "t").unwrap();
    let ids: Vec<_> = (0..10).map(|n| rid(1, n)).collect();
    overlay.set_priority(ids[0], 1000.0).unwrap();
    for id in &ids[1..] {
        overlay.set_priority(*id, 1.0).unwrap();
    }
    let picks: Vec<_> = (0..50)
        .map(|seed| overlay.sample(&ids, 1, seed)[0])
        .collect();
    assert!(picks.iter().filter(|id| **id == ids[0]).count() > 40);
    assert_eq!(overlay.sample(&ids, 3, 7), overlay.sample(&ids, 3, 7));
    assert_eq!(overlay.sample(&ids, 3, 7).len(), 3);
    // An item nobody has prioritised is still sampled, at the default weight.
    let fresh = [rid(9, 9), rid(9, 8)];
    assert_eq!(overlay.sample(&fresh, 2, 1).len(), 2);
}

#[test]
fn a_derived_cache_is_keyed_by_what_it_was_derived_from_so_a_new_tokenizer_starts_clean() {
    let scratch = Scratch::new();
    let db = scratch.open();
    let cache = DerivedCache::new(&db, "tokens");
    let source = ContentId::of(b"the raw text");
    let key_v4 = DerivedCache::key("qwen-v4", "normalise-1", &source);
    let key_v5 = DerivedCache::key("qwen-v5", "normalise-1", &source);
    assert_ne!(key_v4, key_v5);

    assert_eq!(cache.get(&key_v4).unwrap(), None);
    cache.put(&key_v4, b"1 2 3").unwrap();
    assert_eq!(cache.get(&key_v4).unwrap().as_deref(), Some(&b"1 2 3"[..]));
    assert_eq!(
        cache.get(&key_v5).unwrap(),
        None,
        "the old cache is neither used nor damaged"
    );
    // Putting again changes nothing, and the cache holds only derived data.
    cache.put(&key_v4, b"1 2 3").unwrap();
    assert_eq!(cache.get(&key_v4).unwrap().as_deref(), Some(&b"1 2 3"[..]));
}
