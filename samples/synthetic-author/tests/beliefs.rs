// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in proving that a trained model has
// learned what a corpus implies, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: a synthetic author is a pure function of a seed. Its beliefs are
//! few, formal and verifiable: given a situation and its options, code - not a
//! model - says what the author decides and which belief decided it. Every
//! belief an author holds changes some decision, so it can be learned from the
//! decisions it makes; an author holding the opposite beliefs decides
//! differently on the same situations; and the situations are the same for
//! every author, so a situation alone says nothing about who is deciding.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeSet;

use synthetic_author::persona::Persona;
use synthetic_author::scenario::scenario_pool;

const POOL: usize = 2000;

#[test]
fn an_author_is_a_pure_function_of_its_seed() {
    let a = Persona::generate(7);
    let b = Persona::generate(7);
    assert_eq!(a, b);
    let names: BTreeSet<_> = (0..40).map(|s| Persona::generate(s).name).collect();
    assert!(
        names.len() >= 30,
        "{} distinct names in 40 seeds",
        names.len()
    );
}

#[test]
fn a_name_is_two_capitalised_words() {
    for seed in 0..40 {
        let name = Persona::generate(seed).name;
        let words: Vec<_> = name.split(' ').collect();
        assert_eq!(words.len(), 2, "{name}");
        for w in words {
            assert!(w.chars().next().unwrap().is_uppercase(), "{name}");
            assert!(w.chars().all(char::is_alphabetic), "{name}");
        }
    }
}

#[test]
fn an_author_holds_four_distinct_beliefs_each_stated_in_words() {
    for seed in 0..40 {
        let p = Persona::generate(seed);
        assert_eq!(p.beliefs.len(), 4);
        let tenets: BTreeSet<_> = p.beliefs.iter().map(|b| b.tenet).collect();
        assert_eq!(tenets.len(), 4, "a tenet is held once");
        for b in &p.beliefs {
            assert!(b.statement().split_whitespace().count() >= 8);
        }
    }
}

#[test]
fn most_situations_have_one_decision_the_beliefs_determine() {
    let pool = scenario_pool(1, POOL);
    for seed in 0..10 {
        let p = Persona::generate(seed);
        let determined = pool.iter().filter(|s| p.decide(s).determined).count();
        assert!(
            determined * 10 >= POOL * 6,
            "seed {seed}: {determined}/{POOL}"
        );
    }
}

#[test]
fn every_belief_changes_some_decision() {
    let pool = scenario_pool(2, POOL);
    for seed in 0..30 {
        let p = Persona::generate(seed);
        for (i, b) in p.beliefs.iter().enumerate() {
            let flipped = p.with_belief_flipped(i);
            let changes = pool
                .iter()
                .filter(|s| {
                    let (x, y) = (p.decide(s), flipped.decide(s));
                    x.determined && y.determined && x.choice != y.choice
                })
                .count();
            assert!(
                changes > 0,
                "seed {seed}: {:?} never decides anything",
                b.tenet
            );
        }
    }
}

#[test]
fn the_opposite_author_decides_differently_on_the_same_situations() {
    let pool = scenario_pool(3, POOL);
    for seed in 0..10 {
        let p = Persona::generate(seed);
        let q = p.opposite();
        assert_ne!(p.name, q.name);
        let both: Vec<_> = pool
            .iter()
            .filter(|s| p.decide(s).determined && q.decide(s).determined)
            .collect();
        let differ = both
            .iter()
            .filter(|s| p.decide(s).choice != q.decide(s).choice)
            .count();
        assert!(
            differ * 10 >= both.len() * 3,
            "seed {seed}: {differ}/{}",
            both.len()
        );
    }
}

#[test]
fn a_decision_names_the_belief_that_decided_it() {
    let pool = scenario_pool(4, 300);
    let p = Persona::generate(5);
    let held: BTreeSet<_> = p.beliefs.iter().map(|b| b.tenet).collect();
    for s in pool.iter().filter(|s| p.decide(s).determined) {
        let d = p.decide(s);
        assert!(held.contains(&d.deciding.unwrap()), "{d:?}");
    }
}

#[test]
fn the_situations_are_the_same_for_every_author_and_fixed_by_their_seed() {
    assert_eq!(scenario_pool(9, 50), scenario_pool(9, 50));
    assert_ne!(scenario_pool(9, 50), scenario_pool(10, 50));
}
