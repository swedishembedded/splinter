// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The generator's own tests.

use super::*;
use crate::goal::corpus::candidates;
use crate::goal::facts::{FactExtractor, PatternFacts};

#[test]
fn every_label_is_present_and_every_section_has_one() {
    for form in [Form::Property, Form::Behaviour] {
        let c = generate(&Spec::new(7, form)); // seed: any seed must give a full label set
        let cands = candidates(&c.documents);
        assert!(cands.iter().all(|s| c.labels.contains_key(&s.id)));
        let kinds: BTreeSet<Label> = c.labels.values().copied().collect();
        assert_eq!(kinds.len(), 5, "{form:?}");
        assert!(!c.goal.probes.is_empty() && !c.held_out.is_empty());
    }
}

#[test]
fn the_same_seed_gives_the_same_corpus_and_the_facts_are_readable() {
    // seed: determinism is what is tested
    let a = generate(&Spec::new(11, Form::Property));
    assert_eq!(a, generate(&Spec::new(11, Form::Property)));
    let facts: usize = a
        .documents
        .iter()
        .map(|d| PatternFacts.facts(&d.text).len())
        .sum();
    assert!(facts > 100, "{facts}");
}
