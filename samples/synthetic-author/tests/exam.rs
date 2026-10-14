// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in proving that a trained model has
// learned what a corpus implies, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: an author is examined on situations its corpus does not hold. Some
//! are set in topics it never wrote about, some combine facts in a way no
//! article of the corpus does - so the right answer cannot be recalled, it has
//! to be worked out from what the beliefs imply - and every one is put through
//! a tool the model may never have been taught with. The right answer is what
//! the beliefs determine, and the cases are named differently from every case
//! in the corpus.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeSet;

use serde_json::json;
use synthetic_author::exam::{grade, items};
use synthetic_author::persona::Persona;
use synthetic_author::scenario::{held_out_combo, scenario_pool};
use synthetic_author::writer::{plan, HELD_OUT_TOPICS};

fn corpus_cases(p: &Persona, seed: u64) -> BTreeSet<String> {
    plan(p, seed, 100, 3)
        .into_iter()
        .flat_map(|(_, eps)| eps.into_iter().map(|e| e.case.name))
        .collect()
}

#[test]
fn about_a_fifth_of_the_ways_facts_combine_are_kept_out_of_the_corpus() {
    let pool = scenario_pool(1, 4000);
    let held = pool.iter().filter(|s| held_out_combo(9, s)).count();
    assert!((600..=1000).contains(&held), "{held} of 4000");
    // A combination is held out or not whoever asks, and it depends on the seed.
    assert_eq!(held_out_combo(9, &pool[0]), held_out_combo(9, &pool[0]));
    assert!(pool
        .iter()
        .any(|s| held_out_combo(9, s) != held_out_combo(10, s)));
}

#[test]
fn no_article_of_a_corpus_is_set_in_a_held_out_combination() {
    for seed in [3, 4, 5] {
        let p = Persona::generate(seed);
        for (_, eps) in plan(&p, seed, 100, 3) {
            for e in eps {
                assert!(!held_out_combo(seed, &e.scenario));
            }
        }
    }
}

#[test]
fn every_belief_can_still_be_learned_from_the_combinations_the_corpus_holds() {
    for seed in 0..25 {
        let p = Persona::generate(seed);
        let held_in: Vec<_> = scenario_pool(77, 3000)
            .into_iter()
            .filter(|s| !held_out_combo(seed, s))
            .collect();
        for i in 0..p.beliefs.len() {
            let flipped = p.with_belief_flipped(i);
            assert!(
                held_in.iter().any(|s| {
                    let (a, b) = (p.decide(s), flipped.decide(s));
                    a.determined && b.determined && a.choice != b.choice
                }),
                "seed {seed}: belief {i} decides nothing the corpus can show"
            );
        }
    }
}

#[test]
fn the_exam_has_the_same_number_of_items_in_each_kind_of_difficulty() {
    let p = Persona::generate(6);
    let exam = items(&p, 6, 20, &corpus_cases(&p, 6));
    assert_eq!(exam.len(), 80);
    for topic_seen in [true, false] {
        for combo_seen in [true, false] {
            let n = exam
                .iter()
                .filter(|i| i.topic_seen == topic_seen && i.combo_seen == combo_seen)
                .count();
            assert_eq!(n, 20, "topic_seen {topic_seen} combo_seen {combo_seen}");
        }
    }
    // Both kinds of tool in every kind of difficulty.
    assert!(exam.iter().any(|i| i.tool_seen) && exam.iter().any(|i| !i.tool_seen));
    for i in &exam {
        assert_eq!(!i.combo_seen, held_out_combo(6, &i.scenario), "{}", i.id);
        assert_eq!(
            !i.topic_seen,
            HELD_OUT_TOPICS.contains(&i.scenario.topic),
            "{}",
            i.id
        );
        assert!(p.decide(&i.scenario).determined);
        assert_eq!(i.expected, p.decide(&i.scenario).choice);
    }
}

#[test]
fn the_exam_names_no_case_the_corpus_tells_and_is_the_same_every_time() {
    let p = Persona::generate(6);
    let told = corpus_cases(&p, 6);
    let a = items(&p, 6, 20, &told);
    let b = items(&p, 6, 20, &told);
    assert_eq!(
        a.iter().map(|i| &i.id).collect::<Vec<_>>(),
        b.iter().map(|i| &i.id).collect::<Vec<_>>()
    );
    let cases: BTreeSet<_> = a.iter().map(|i| i.case.clone()).collect();
    assert_eq!(cases.len(), a.len(), "every item has a case of its own");
    assert!(cases.is_disjoint(&told));
}

#[test]
fn a_call_is_right_only_if_it_is_to_the_tool_offered_and_names_the_expected_option() {
    let p = Persona::generate(6);
    let exam = items(&p, 6, 5, &corpus_cases(&p, 6));
    let item = &exam[0];
    let schema = &item.schema;
    let position = item
        .options
        .iter()
        .position(|&o| o == item.expected)
        .unwrap();
    let right = json!({ schema.choice: schema.label(item.expected, position), schema.reason: "x" });
    assert!(grade(item, Some((schema.name, &right))));
    assert!(!grade(item, Some(("other_tool", &right))));
    assert!(!grade(item, None));
    let wrong_option = item
        .options
        .iter()
        .find(|&&o| o != item.expected)
        .copied()
        .unwrap();
    let wrong_position = item
        .options
        .iter()
        .position(|&o| o == wrong_option)
        .unwrap();
    let wrong = json!({ schema.choice: schema.label(wrong_option, wrong_position) });
    assert!(!grade(item, Some((schema.name, &wrong))));
}
