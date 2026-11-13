// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission and red-team testing of training
// environments, for its clients. If your team needs expertise in keeping
// weak verifiers out of reinforcement-learning campaigns, you can procure
// our services by sending an email to info@swedishembedded.com.

//! The document question plugin: questions compiled from documents, answered
//! without them.

#![allow(clippy::unwrap_used, clippy::expect_used)] // a panic is the failure report

use std::collections::{BTreeMap, BTreeSet};

use splinter_core::verifiability::Split;
use splinter_knowledge::factory::admission::{admit, Status};
use splinter_knowledge::factory::band::BandMeasurement;
use splinter_knowledge::factory::host::{Host, Verdict};
use splinter_knowledge::factory::plugin::{InstanceRequest, Plugin};
use splinter_knowledge::factory::plugins::document_questions::DocumentQuestions;

use super::factory_admission::small_policy;

/// Six datasheets of five pumps each: the same attributes, other figures.
fn documents() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for d in 0..6 {
        let mut text = format!("# Datasheet {d}\n\n");
        for p in 0..5 {
            let name = format!("pump {}{}", ["A", "B", "C", "D", "E", "F"][d], p);
            let (rate, pressure) = (11 + 7 * p + 13 * d, 30 + 9 * p + 5 * d);
            text += &format!("The flow rate of {name} is {rate} L/min.\n");
            text += &format!("The rated pressure of {name} is {pressure} bar.\n");
            text += &format!("The supplier of {name} is Norrlands {d}{p}.\n");
        }
        std::fs::write(dir.path().join(format!("sheet-{d}.md")), text).unwrap();
    }
    dir
}

fn request(split: Split, seed: u64, recall: u32, steps: u32) -> InstanceRequest {
    InstanceRequest {
        split,
        seed,
        difficulty: BTreeMap::from([
            ("questions".to_string(), 2),
            ("steps".to_string(), steps),
            ("recall".to_string(), recall),
        ]),
    }
}

#[test]
fn a_derived_question_carries_the_figures_not_the_answer_and_the_reference_gets_full_marks() {
    let dir = documents();
    let plugin = DocumentQuestions::new(dir.path());
    let instance = plugin.generate(&request(Split::Train, 3, 0, 2)).unwrap(); // seed: any
    assert!(
        instance.instruction().contains("Questions:\n1. The "),
        "{}",
        instance.instruction()
    );
    let session = Host::default().open(&instance).unwrap();
    assert!(!session.verify().unwrap().solved());
    assert!(session.solve().unwrap().succeeded());
    assert!(session.verify().unwrap().solved());
    // One line right of two is partial credit and no reward.
    let answer = instance.folder().text("solution/answer.txt").unwrap();
    let first = answer.lines().next().unwrap();
    session
        .write("answer.txt", format!("{first}\n0\n").as_bytes())
        .unwrap();
    let Verdict::Graded(vector) = session.verify().unwrap() else {
        panic!("graded")
    };
    assert_eq!(vector["reward"], 0.0);
    assert_eq!(vector["lines"], 0.5);
}

#[test]
fn a_dev_question_is_about_a_document_no_train_question_used() {
    let dir = documents();
    let plugin = DocumentQuestions::new(dir.path());
    let used = |split| -> BTreeSet<String> {
        // seed: coverage of the document draw over many seeds
        (0..10)
            .map(|seed| plugin.generate(&request(split, seed, 1, 1)).unwrap())
            .flat_map(|i| {
                let grounding = i.folder().text("tests/grounding.json").unwrap().to_string();
                let value: serde_json::Value = serde_json::from_str(&grounding).unwrap();
                value
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|g| g["document"].as_str().unwrap().to_string())
                    .collect::<Vec<_>>()
            })
            .collect()
    };
    let (train, dev) = (used(Split::Train), used(Split::Dev));
    assert!(!train.is_empty() && !dev.is_empty());
    assert!(train.is_disjoint(&dev), "{train:?} {dev:?}");
}

#[test]
fn derived_and_recall_environments_pass_admission_and_no_attack_wins() {
    let dir = documents();
    for recall in [0, 1] {
        let plugin = DocumentQuestions::new(dir.path());
        let difficulty = BTreeMap::from([
            ("questions".to_string(), 2),
            ("steps".to_string(), 2),
            ("recall".to_string(), recall),
        ]);
        let (report, _) = admit(
            &plugin,
            &difficulty,
            &Host::default(),
            &small_policy(),
            90 + u64::from(recall), // seed: any admission seed
            &BTreeSet::new(),
            Some(&BandMeasurement {
                model: "base".into(),
                attempts: 32,
                passes: 16,
            }),
        )
        .unwrap();
        for g in &report.gates {
            assert!(g.passed, "recall {recall} {}: {}", g.name, g.detail);
        }
        assert_eq!(report.status, Status::Admitted, "{:?}", report.reasons);
        assert!(report.beaten.is_empty(), "{:?}", report.beaten);
    }
}

#[test]
fn a_recall_answer_may_not_bury_the_value_in_a_list_of_guesses() {
    let dir = documents();
    let instance = DocumentQuestions::new(dir.path())
        .generate(&request(Split::Train, 8, 1, 1))
        .unwrap(); // seed: any
    let session = Host::default().open(&instance).unwrap();
    session.solve().unwrap();
    let answer = instance.folder().text("solution/answer.txt").unwrap();
    let first = answer.lines().next().unwrap();
    let padded = format!(
        "{first} or 1 or 2 or 3 or 4 or 5 or 6\n{}\n",
        answer.lines().nth(1).unwrap()
    );
    session.write("answer.txt", padded.as_bytes()).unwrap();
    assert!(!session.verify().unwrap().solved());
}
