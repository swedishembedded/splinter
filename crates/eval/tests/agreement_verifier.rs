// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: agreement between independent answers to the same task. An answer
//! that agrees with a strict majority passes, one that disagrees with a
//! strict majority fails, and without a strict majority there is no
//! verdict. The evidence records the tally and who gave each answer.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::{experience, task, verdict};
use splinter_core::annotation::{Outcome, Strength};
use splinter_core::experience::{Environment, Experience, Task};
use splinter_eval::verifiers::consistency::AgreementVerifier;
use splinter_eval::verifiers::normalise::Normalisation;

fn capital() -> Task {
    task("recall", Environment::closed_book(), vec![])
}

fn answers(task: &Task, pairs: &[(&str, &str)]) -> Vec<Experience> {
    pairs
        .iter()
        .map(|(solver, answer)| experience(task, Some(answer), solver))
        .collect()
}

#[test]
fn three_agreeing_answers_pass_and_the_dissenter_fails() {
    let task = capital();
    let all = answers(
        &task,
        &[
            ("a", "Paris"),
            ("b", " paris."),
            ("c", "PARIS"),
            ("d", "London"),
        ],
    );
    let verifier = AgreementVerifier::new(all.clone(), Normalisation::LENIENT);
    for exp in &all[..3] {
        let (outcome, strength, evidence) = verdict(&verifier, &task, exp);
        assert_eq!(
            (outcome, strength),
            (Outcome::Pass, Strength::Consistency),
            "{evidence}"
        );
    }
    let (outcome, _, evidence) = verdict(&verifier, &task, &all[3]);
    assert_eq!(outcome, Outcome::Fail);
    assert_eq!(evidence["answers"], 4);
    let tally = evidence["tally"].as_array().unwrap();
    assert_eq!(tally.len(), 2);
    assert_eq!(tally[0]["count"], 3);
    assert_eq!(tally[0]["solvers"], serde_json::json!(["a", "b", "c"]));
    assert_eq!(tally[1]["solvers"], serde_json::json!(["d"]));
}

#[test]
fn a_tie_is_abstained_on_and_other_tasks_do_not_vote() {
    let task = capital();
    let mut all = answers(
        &task,
        &[
            ("a", "Paris"),
            ("b", "Paris"),
            ("c", "London"),
            ("d", "London"),
        ],
    );
    let other = Task::new(
        "recall",
        vec![],
        Environment::closed_book(),
        "Another question.",
        vec![],
    )
    .unwrap();
    all.push(experience(&other, Some("Paris"), "e"));
    let verifier = AgreementVerifier::new(all.clone(), Normalisation::LENIENT);
    for exp in &all[..4] {
        let (outcome, _, evidence) = verdict(&verifier, &task, exp);
        assert_eq!(outcome, Outcome::Abstain, "{evidence}");
    }
}
