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
//!
//! The second version of the examination lists the options of each item in an
//! order drawn from the seed, with the right answer at each position equally
//! often within every kind of difficulty (to within one item), says for each
//! item whether the corpus holds its combination of facts and options, and is
//! a different examination, with a different digest, from the first.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::collections::BTreeSet;

use serde_json::json;
use synthetic_author::binding::items_digest;
use synthetic_author::exam::{draw, grade, items, Spec, Trained, Version};
use synthetic_author::gates::MinedFacts;
use synthetic_author::mine::Verified;
use synthetic_author::persona::Persona;
use synthetic_author::report::baselines;
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

fn v2(p: &Persona, per_cell: usize, trained: &Trained) -> Vec<synthetic_author::exam::Item> {
    let told = corpus_cases(p, 6);
    draw(
        p,
        6,
        &Spec {
            version: Version::V2,
            per_cell,
            corpus_cases: &told,
            trained,
        },
    )
}

#[test]
fn in_the_second_exam_the_right_answer_sits_at_each_position_equally_often_in_every_kind() {
    let p = Persona::generate(6);
    let exam = v2(&p, 20, &Trained::default());
    for topic_seen in [true, false] {
        for combo_seen in [true, false] {
            let mut at = [0_usize; 3];
            for i in exam
                .iter()
                .filter(|i| i.topic_seen == topic_seen && i.combo_seen == combo_seen)
            {
                at[i.options.iter().position(|&o| o == i.expected).unwrap()] += 1;
            }
            // A third of 20 is 6.67: no position strays by more than one item.
            assert!(
                at.iter().all(|&n| (6..=7).contains(&n)),
                "topic_seen {topic_seen} combo_seen {combo_seen}: {at:?}"
            );
        }
    }
    // The same options as the situation offers, in an order the seed draws.
    let mut reordered = 0;
    for i in &exam {
        let mut listed = i.options.clone();
        listed.sort();
        assert_eq!(listed, i.scenario.options, "{}", i.id);
        reordered += usize::from(i.options != i.scenario.options);
    }
    assert!(reordered > exam.len() / 2, "{reordered} of {}", exam.len());
    assert_eq!(
        items_digest(&exam),
        items_digest(&v2(&p, 20, &Trained::default())),
        "drawn again from the same seed it is the same exam"
    );
    let first = items(&p, 6, 20, &corpus_cases(&p, 6));
    assert_ne!(items_digest(&exam), items_digest(&first), "a new exam");
}

#[test]
fn the_exam_reports_what_a_constant_policy_scores_and_the_second_exam_holds_it_near_a_third() {
    let p = Persona::generate(6);
    let first = items(&p, 6, 20, &corpus_cases(&p, 6));
    let second = v2(&p, 20, &Trained::default());
    let score = |exam: &[synthetic_author::exam::Item], name: &str| {
        let arms = baselines(exam);
        let (_, outcomes) = arms.iter().find(|(n, _)| n.contains(name)).unwrap();
        assert_eq!(outcomes.len(), exam.len());
        outcomes.iter().filter(|o| o.correct).count()
    };
    let at_first = |exam: &[synthetic_author::exam::Item]| {
        exam.iter().filter(|i| i.options[0] == i.expected).count()
    };
    assert_eq!(score(&first, "first"), at_first(&first));
    assert_eq!(score(&second, "first"), at_first(&second));
    assert!(
        at_first(&first) > 40,
        "the canonical order favours the first"
    );
    assert!(
        (80 / 3 - 4..=80 / 3 + 4).contains(&at_first(&second)),
        "{}",
        at_first(&second)
    );
    // The majority option is the way the beliefs most often determine.
    let mut counts = std::collections::BTreeMap::new();
    for i in &second {
        *counts.entry(i.expected).or_insert(0_usize) += 1;
    }
    assert_eq!(score(&second, "majority"), *counts.values().max().unwrap());
}

#[test]
fn an_item_is_trained_on_when_a_kept_decision_has_its_facts_and_its_options() {
    let p = Persona::generate(6);
    let none = v2(&p, 20, &Trained::default());
    assert!(none.iter().all(|i| i.combo_trained == Some(false)));
    let target = &none[3];
    let f = target.scenario.features;
    let decision = Verified {
        case: "Any".into(),
        briefing: "x".into(),
        offer_made: f.outsider_offers_help,
        facts: MinedFacts {
            exceeds_authority: Some(f.exceeds_authority),
            urgent: Some(f.urgent),
            binds_future: Some(f.binds_future),
            irreversible: Some(f.irreversible),
            members_harmed: Some(f.members_harmed),
        },
        chose: target.expected,
        open: target.scenario.options.iter().rev().copied().collect(),
        reasons: vec![],
        dropped: 0,
    };
    let trained = Trained::from_decisions(&[decision]);
    let exam = v2(&p, 20, &trained);
    let same = |a: &synthetic_author::exam::Item| {
        a.scenario.options == target.scenario.options
            && serde_json::to_value(a.scenario.features).unwrap()
                == serde_json::to_value(f).unwrap()
    };
    for (item, before) in exam.iter().zip(&none) {
        assert_eq!(item.id, before.id, "the same situations, flagged");
        assert_eq!(item.combo_trained, Some(same(item)), "{}", item.id);
    }
    assert!(exam.iter().any(|i| i.combo_trained == Some(true)));
    assert!(items(&p, 6, 5, &corpus_cases(&p, 6))
        .iter()
        .all(|i| i.combo_trained.is_none()));
}

#[test]
fn the_second_exam_cannot_be_drawn_before_the_corpus_s_decisions_are_read() {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("synthetic-author-exam-v2");
    let _ = std::fs::remove_dir_all(&dir);
    let cache = synthetic_author::cache::Cache::open(dir).unwrap();
    let p = Persona::generate(6);
    let refused =
        synthetic_author::lab::exam_items_of(&cache, &p, 6, 100, 5, Version::V2).unwrap_err();
    assert!(refused.to_string().contains("mine"), "{refused}");
    let first = synthetic_author::lab::exam_items_of(&cache, &p, 6, 100, 5, Version::V1).unwrap();
    assert_eq!(first.len(), 20);
}
