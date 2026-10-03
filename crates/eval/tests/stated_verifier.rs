// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: the stated-reference verifier passes an answer that states the
//! task's reference - every word of it, in order, bounded by non-word
//! characters - without much else around it, at formal strength. A sentence
//! that contains the fact passes; a different fact, a half of it, the fact
//! buried in padding, or no answer fails; and it abstains on a task without
//! exactly one reference.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use serde_json::json;
use splinter_core::annotation::{AnnotationBody, Outcome, Producer, Strength};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{
    Digest, Environment, Experience, Privileged, PrivilegedKind, Provenance, Span, Task,
};
use splinter_eval::verifiers::formal::StatedReferenceVerifier;
use splinter_eval::verifiers::normalise::Normalisation;
use splinter_eval::verifiers::{annotation, Verifier};
use sven_sdk::atif::{AgentProfile, Trajectory};

fn experience(reference: Option<&str>, output: Option<&str>) -> Experience {
    let source = "the source text";
    let span = Span::new(Digest::of(source.as_bytes()), 0, source.len() as u64).unwrap();
    let privileged = reference
        .map(|r| Privileged {
            kind: PrivilegedKind::Reference,
            content: r.into(),
            span: Some(span.clone()),
        })
        .into_iter()
        .collect();
    let task = Task::new(
        "recall",
        vec![span],
        Environment {
            kind: "closed-book".into(),
            spec: json!({}),
            snapshot: None,
        },
        "What is the default port?",
        privileged,
    )
    .unwrap();
    let profile = AgentProfile {
        name: "t".into(),
        version: "1".into(),
        model_name: None,
        tool_definitions: None,
        extra: None,
    };
    Experience::new(
        task,
        Trajectory::new("ATIF-v1.7", profile),
        output.map(str::to_string),
        Provenance::new("scripted", &FixedClock::new("2026-10-01T00:00:00.000Z")),
    )
    .unwrap()
}

fn judged(reference: Option<&str>, output: Option<&str>) -> (Outcome, Strength) {
    let verifier = StatedReferenceVerifier::new(
        Producer {
            name: "test/stated".into(),
            version: "1".into(),
        },
        Normalisation::LENIENT,
    );
    let exp = experience(reference, output);
    let note = annotation(&verifier as &dyn Verifier, &exp.to_task(), &exp).unwrap();
    match note.body {
        AnnotationBody::Verdict {
            outcome, strength, ..
        } => (outcome, strength),
        other => panic!("not a verdict: {other:?}"),
    }
}

#[test]
fn a_sentence_that_states_the_reference_passes_at_formal_strength() {
    for answer in [
        "8789",
        "The default port is 8789.",
        "  THE DEFAULT PORT IS 8789!\n",
    ] {
        assert_eq!(
            judged(Some("8789"), Some(answer)),
            (Outcome::Pass, Strength::Formal),
            "{answer:?}"
        );
    }
    assert_eq!(
        judged(
            Some("device registry, adapter enumeration, backend open/submit/wait"),
            Some("It traces the device registry, adapter enumeration, backend open/submit/wait."),
        )
        .0,
        Outcome::Pass
    );
}

#[test]
fn a_different_fact_a_fragment_or_no_answer_fails() {
    assert_eq!(
        judged(Some("8789"), Some("The default port is 8790.")).0,
        Outcome::Fail
    );
    assert_eq!(
        judged(Some("8789"), Some("The default port is 18789.")).0,
        Outcome::Fail
    );
    assert_eq!(
        judged(
            Some("hard cap on total memory"),
            Some("a hard cap on memory")
        )
        .0,
        Outcome::Fail
    );
    assert_eq!(judged(Some("8789"), None).0, Outcome::Fail);
}

#[test]
fn the_fact_buried_in_padding_fails() {
    let padding = "It could be many things: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, \
                   16, 17, 18, 19, 20, and among them 8789, or perhaps another value entirely.";
    assert_eq!(judged(Some("8789"), Some(padding)).0, Outcome::Fail);
}

#[test]
fn it_abstains_without_exactly_one_reference() {
    assert_eq!(judged(None, Some("8789")).0, Outcome::Abstain);
}
