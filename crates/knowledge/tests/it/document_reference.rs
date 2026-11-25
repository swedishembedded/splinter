// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements document-grounded training questions whose
// every answer is traced to a sentence a document really states, for its
// clients. If your team needs expertise in turning manuals and datasheets
// into checked training material, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec (4B, `document-questions` against independently labelled fixtures):
//! documents are built sentence by sentence, so every fact they state is
//! labelled by construction (whole numbers, decimals, a dotted version, a
//! link and a behaviour) beside sentences that state none. Over many
//! instances of both kinds and both splits, every question is grounded: a
//! recall question asks a labelled fact of the document its grounding names,
//! and expects exactly the labelled value; a derived question's premises are
//! labelled figures of that document, and its expected number is computed
//! here from the labels. The reference answer earns the full reward, a value
//! off by one earns none.

#![allow(clippy::unwrap_used, clippy::expect_used)] // a panic is the failure report

use std::collections::{BTreeMap, BTreeSet};

use splinter_core::verifiability::Split;
use splinter_knowledge::factory::artifact::Instance;
use splinter_knowledge::factory::host::Host;
use splinter_knowledge::factory::plugin::{InstanceRequest, Plugin};
use splinter_knowledge::factory::plugins::document_questions::DocumentQuestions;

/// One labelled fact: its question, as a reader would ask it, and its value.
#[derive(Clone, Debug)]
struct Label {
    question: String,
    value: String,
    /// The figure, for a property whose value is a whole number and a unit.
    figure: Option<(String, String, i64, String)>,
}

/// Nine documents (three of them dev, by name), each its labelled facts.
fn documents() -> (tempfile::TempDir, BTreeMap<String, Vec<Label>>) {
    let dir = tempfile::tempdir().unwrap();
    let mut labels = BTreeMap::new();
    for d in 0..9 {
        let mut text = format!(
            "# Unit sheet {d}\n\nThis sheet is for the service team. It is not a contract.\n\n"
        );
        let mut facts = Vec::new();
        for p in 0..4 {
            let unit = format!(
                "valve {}{p}",
                ["K", "L", "M", "N", "O", "P", "Q", "R", "S"][d]
            );
            let flow = (20 + 6 * p + 11 * d) as i64;
            text += &format!("The flow rate of {unit} is {flow} L/min.\n");
            facts.push(Label {
                question: format!("What is the flow rate of {unit}?"),
                value: format!("{flow} L/min"),
                figure: Some(("flow rate".into(), unit.clone(), flow, "L/min".into())),
            });
            let setting = format!("{}.{} bar", 2 + p, 5 + d);
            text += &format!("The relief setting of {unit} is {setting}.\n");
            facts.push(Label {
                question: format!("What is the relief setting of {unit}?"),
                value: setting,
                figure: None,
            });
            let firmware = format!("v{d}.{p}.7");
            text += &format!("The firmware of {unit} is {firmware}.\n");
            facts.push(Label {
                question: format!("What is the firmware of {unit}?"),
                value: firmware,
                figure: None,
            });
            text += &format!("Each {unit} is checked every spring, and the log of it is kept.\n");
        }
        let site = format!("dock {d}");
        text += &format!(
            "The unit installed at {site} is valve {}0.\n",
            ["K", "L", "M", "N", "O", "P", "Q", "R", "S"][d]
        );
        facts.push(Label {
            question: format!("Which unit is installed at {site}?"),
            value: format!(
                "valve {}0",
                ["K", "L", "M", "N", "O", "P", "Q", "R", "S"][d]
            ),
            figure: None,
        });
        text +=
            &format!("When the invoice is late {d} weeks, a careful client asks for a reminder.\n");
        facts.push(Label {
            question: format!("What does a careful client do when the invoice is late {d} weeks?"),
            value: "asks for a reminder".into(),
            figure: None,
        });
        text += "Values of this kind are measured at the bench. The plant has a small team of its own.\n";
        let name = format!("sheet-{d}.md");
        std::fs::write(dir.path().join(&name), text).unwrap();
        labels.insert(name, facts);
    }
    (dir, labels)
}

fn questions(instance: &Instance) -> Vec<String> {
    let (_, listed) = instance.instruction().split_once("Questions:\n").unwrap();
    listed
        .lines()
        .map(|l| l.split_once(". ").unwrap().1.to_string())
        .collect()
}

fn json(instance: &Instance, path: &str) -> Vec<serde_json::Value> {
    serde_json::from_str(instance.folder().text(path).unwrap()).unwrap()
}

/// The figures a derived question gives: `<entity> <number> <unit>` each.
fn premises(question: &str) -> Vec<(String, i64)> {
    let (_, rest) = question.split_once(" are: ").unwrap();
    let (listed, _) = rest.split_once(". What is ").unwrap();
    listed
        .split("; ")
        .map(|premise| {
            let words: Vec<&str> = premise.split(' ').collect();
            let at = words.iter().position(|w| w.parse::<i64>().is_ok()).unwrap();
            (words[..at].join(" "), words[at].parse().unwrap())
        })
        .collect()
}

fn reward(instance: &Instance, answer: &str) -> Option<f64> {
    let session = Host::default().open(instance).unwrap();
    session.write("answer.txt", answer.as_bytes()).unwrap();
    session.verify().unwrap().reward()
}

#[test]
fn every_question_is_grounded_in_a_labelled_fact_and_expects_its_value() {
    let (dir, labels) = documents();
    let plugin = DocumentQuestions::new(dir.path());
    let mut recalled = BTreeSet::new();
    let mut derived = 0;
    for split in [Split::Train, Split::Dev] {
        for (recall, steps) in [(1, 1), (0, 1), (0, 2), (0, 3)] {
            for seed in 0..12 {
                // seed: any, a dozen of each kind and split
                let instance = plugin
                    .generate(&InstanceRequest {
                        split,
                        seed,
                        difficulty: BTreeMap::from([
                            ("questions".to_string(), 2),
                            ("steps".to_string(), steps),
                            ("recall".to_string(), recall),
                        ]),
                    })
                    .unwrap();
                let expected = json(&instance, "tests/expected.json");
                let grounding = json(&instance, "tests/grounding.json");
                let mut answers = String::new();
                for ((question, expected), grounding) in
                    questions(&instance).iter().zip(&expected).zip(&grounding)
                {
                    let document = grounding["document"].as_str().unwrap();
                    let facts = &labels[document];
                    let context = format!("{document}: {question}");
                    if recall == 1 {
                        let label = facts
                            .iter()
                            .find(|l| l.question.eq_ignore_ascii_case(question))
                            .unwrap_or_else(|| panic!("{context}: asks no labelled fact"));
                        assert_eq!(expected["value"], label.value, "{context}");
                        recalled.insert(label.value.clone());
                        answers += &format!("{}\n", label.value);
                    } else {
                        let given = premises(question);
                        assert_eq!(given.len(), steps as usize + 1, "{context}");
                        for (entity, value) in &given {
                            assert!(
                                facts.iter().any(|l| l.figure.as_ref().is_some_and(
                                    |(_, e, v, _)| e.eq_ignore_ascii_case(entity) && v == value
                                )),
                                "{context}: {entity} {value} is no labelled figure"
                            );
                        }
                        let total: i64 = given.iter().map(|(_, v)| v).sum();
                        let answer = if question.contains("What is their total") {
                            total
                        } else {
                            2 * given[0].1 - total
                        };
                        assert_eq!(expected["value"], answer, "{context}");
                        derived += 1;
                        answers += &format!("{answer}\n");
                    }
                }
                assert_eq!(reward(&instance, &answers), Some(1.0), "{answers}");
                let off: String = expected
                    .iter()
                    .map(|e| match e["value"].as_i64() {
                        Some(n) => format!("{}\n", n + 1),
                        None => "none of these\n".to_string(),
                    })
                    .collect();
                assert_eq!(reward(&instance, &off), Some(0.0), "{off}");
            }
        }
    }
    assert!(derived > 0);
    assert!(
        recalled.iter().any(|v| v.ends_with(" bar")) && recalled.iter().any(|v| v.starts_with('v')),
        "the decimals and versions were asked: {recalled:?}"
    );
}
