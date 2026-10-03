// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: the views that teach judgement of answers.
//!
//! * critic: a verified critique of a candidate answer, supervised as the
//!   reply to the task and that answer.
//! * preference: (chosen, rejected) answers to one task, from recorded
//!   preferences and from a pass and a fail decided at the same strength.
//! * verifier: a task and a candidate answer, classified pass or fail by
//!   the decision, with the execution evidence when a check ran.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use serde_json::json;
use splinter_core::annotation::{
    Annotation, AnnotationBody, Outcome, Producer, RelationKind, Strength,
};
use splinter_core::experience::PrivilegedKind;
use splinter_data::{
    Corpus, Critic, Exclusion, Objective, Preference, RecordBody, VerifierView, View,
};
use splinter_lab::verifiers::executable;

const INSTRUCTION: &str = "What is two plus two? Reply with the number only.";

fn chat_turns(body: &RecordBody) -> Vec<(String, String, bool)> {
    match body {
        RecordBody::Chat { messages } => after_system(messages)
            .iter()
            .map(|m| (m.role.clone(), m.content.clone(), m.train))
            .collect(),
        other => panic!("a chat record, got {other:?}"),
    }
}

#[test]
fn critic_supervises_only_verified_critiques_of_a_candidate() {
    let sum = task(
        "arithmetic",
        INSTRUCTION,
        vec![privileged(PrivilegedKind::Reference, "4")],
    );
    let candidate = answered(&sum, "5", "2026-09-30T01:00:00.000Z");
    let critique_task = task(
        "critique",
        "Critique the answer to an arithmetic question.",
        Vec::new(),
    );
    const CRITIQUE: &str = "Wrong: two plus two is four, not five.";
    let verified = answered(&critique_task, CRITIQUE, "2026-09-30T02:00:00.000Z");
    let refuted = answered(&critique_task, "Correct.", "2026-09-30T03:00:00.000Z");

    let mut corpus = Corpus::new();
    corpus
        .insert(
            candidate.clone(),
            vec![verdict(&candidate, Outcome::Fail, Strength::Formal)],
        )
        .unwrap();
    corpus
        .insert(
            verified.clone(),
            vec![
                relation(&verified, RelationKind::CritiqueOf, &candidate),
                verdict(&verified, Outcome::Pass, Strength::Consistency),
            ],
        )
        .unwrap();
    corpus
        .insert(
            refuted.clone(),
            vec![
                relation(&refuted, RelationKind::CritiqueOf, &candidate),
                verdict(&refuted, Outcome::Fail, Strength::Consistency),
            ],
        )
        .unwrap();

    let view = Critic::new(Strength::Consistency);
    assert_eq!(view.objective(), Objective::Sft);
    let projection = view.project(&corpus).unwrap();
    let [record] = &projection.records[..] else {
        panic!("one record, got {:?}", projection.records)
    };
    let turns = chat_turns(&record.body);
    let [(user_role, input, false), (assistant_role, target, true)] = &turns[..] else {
        panic!("a user turn and a supervised reply, got {turns:?}")
    };
    assert_eq!(
        (user_role.as_str(), assistant_role.as_str()),
        ("user", "assistant")
    );
    assert!(input.starts_with(INSTRUCTION), "{input}");
    assert!(
        input.ends_with("5"),
        "the candidate answer closes the input: {input}"
    );
    assert_eq!(target, CRITIQUE);
    assert_eq!(record.metadata.experiences, [id(&verified), id(&candidate)]);
    assert_eq!(projection.count(Exclusion::UnverifiedCritique), 1);
}

#[test]
fn preference_pairs_recorded_preferences_and_comparable_pass_fail_decisions() {
    let sum = task("arithmetic", INSTRUCTION, Vec::new());
    let other = task("arithmetic", "What is three plus three?", Vec::new());
    let pass = answered(&sum, "4", "2026-09-30T01:00:00.000Z");
    let fail = answered(&sum, "5", "2026-09-30T02:00:00.000Z");
    let weak_fail = answered(&sum, "22", "2026-09-30T03:00:00.000Z");
    let preferred = answered(&sum, "four", "2026-09-30T04:00:00.000Z");
    let elsewhere = answered(&other, "6", "2026-09-30T05:00:00.000Z");

    let mut corpus = Corpus::new();
    corpus
        .insert(
            pass.clone(),
            vec![
                verdict(&pass, Outcome::Pass, Strength::Formal),
                // Recorded and also derivable: one pair, not two.
                relation(&pass, RelationKind::PreferredOver, &fail),
            ],
        )
        .unwrap();
    corpus
        .insert(
            fail.clone(),
            vec![verdict(&fail, Outcome::Fail, Strength::Formal)],
        )
        .unwrap();
    // Decided more weakly than the pass: not comparable, so not derived.
    corpus
        .insert(
            weak_fail.clone(),
            vec![verdict(&weak_fail, Outcome::Fail, Strength::Judged)],
        )
        .unwrap();
    corpus.insert(elsewhere.clone(), Vec::new()).unwrap();
    corpus
        .insert(
            preferred.clone(),
            vec![
                relation(&preferred, RelationKind::PreferredOver, &weak_fail),
                relation(&preferred, RelationKind::PreferredOver, &elsewhere),
            ],
        )
        .unwrap();

    let view = Preference::new(Strength::Judged);
    assert_eq!(view.objective(), Objective::Dpo);
    let projection = view.project(&corpus).unwrap();
    let pairs: Vec<(Vec<_>, String, String, String)> = projection
        .records
        .iter()
        .map(|r| match &r.body {
            RecordBody::Preference {
                prompt,
                chosen,
                rejected,
            } => (
                r.metadata.experiences.clone(),
                after_system(prompt)[0].content.clone(),
                chosen.content.clone(),
                rejected.content.clone(),
            ),
            other => panic!("a preference record, got {other:?}"),
        })
        .collect();
    let prompt = INSTRUCTION.to_string();
    assert_eq!(
        pairs,
        [
            (
                vec![id(&pass), id(&fail)],
                prompt.clone(),
                "4".into(),
                "5".into()
            ),
            (
                vec![id(&preferred), id(&weak_fail)],
                prompt,
                "four".into(),
                "22".into()
            ),
        ]
    );
    assert_eq!(projection.count(Exclusion::DifferentTask), 1);
}

#[test]
fn verifier_classifies_a_candidate_by_its_decision_with_execution_evidence() {
    let sum = task(
        "arithmetic",
        INSTRUCTION,
        vec![privileged(PrivilegedKind::Reference, "4")],
    );
    let pass = answered(&sum, "4", "2026-09-30T01:00:00.000Z");
    let fail = answered(&sum, "5", "2026-09-30T02:00:00.000Z");
    let ungraded = answered(&sum, "6", "2026-09-30T03:00:00.000Z");
    let executed = Annotation {
        experience: id(&fail),
        producer: Producer {
            name: splinter_core::evidence::EXECUTABLE_PRODUCER.into(),
            version: executable::VERSION.into(),
        },
        body: AnnotationBody::Verdict {
            outcome: Outcome::Fail,
            strength: Strength::Executable,
            evidence: json!({
                "output": "sha256:00",
                "checks": [{
                    "runtime": {"name": "python", "version": "3.12.1"},
                    "exit_code": 1,
                    "signal": null,
                    "timed_out": false,
                    "passed": false,
                    "unmet": ["exit_code"],
                }],
            }),
        },
    };
    let mut corpus = Corpus::new();
    corpus
        .insert(
            pass.clone(),
            vec![verdict(&pass, Outcome::Pass, Strength::Formal)],
        )
        .unwrap();
    corpus.insert(fail.clone(), vec![executed]).unwrap();
    corpus.insert(ungraded, Vec::new()).unwrap();

    let view = VerifierView::new(Strength::Formal);
    assert_eq!(view.objective(), Objective::Classification);
    let projection = view.project(&corpus).unwrap();
    let [passed, failed] = &projection.records[..] else {
        panic!("two records, got {:?}", projection.records)
    };
    let passed = chat_turns(&passed.body);
    assert!(passed[0].1.starts_with(INSTRUCTION));
    assert!(passed[0].1.contains('4'));
    assert!(
        !passed[0].1.contains("Execution"),
        "no check ran: {}",
        passed[0].1
    );
    assert_eq!((passed[1].1.as_str(), passed[1].2), ("pass", true));

    let failed = chat_turns(&failed.body);
    assert!(failed[0].1.contains("exit code 1"), "{}", failed[0].1);
    assert!(failed[0].1.contains("python 3.12.1"), "{}", failed[0].1);
    assert_eq!((failed[1].1.as_str(), failed[1].2), ("fail", true));
    assert_eq!(projection.count(Exclusion::Undecided), 1);
}
