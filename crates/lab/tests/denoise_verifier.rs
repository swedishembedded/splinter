// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: the denoise verifier passes an answer iff it equals the reference
//! passage after whitespace normalisation, says so at formal strength, and
//! records what it compared without copying the reference into the verdict.
//! It abstains on what it cannot judge. The exact-match verifier it is built
//! on normalises only what it is configured to.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use serde_json::json;
use splinter_lab::denoise::{FormalVerifier, KIND};
use splinter_lab::verifiers::formal::ExactMatchVerifier;
use splinter_lab::verifiers::normalise::Normalisation;
use splinter_lab::verifiers::{annotation, Verifier};
use splinter_record::annotation::{AnnotationBody, Outcome, Producer, Strength};
use splinter_record::clock::FixedClock;
use splinter_record::experience::{
    Digest, Environment, Experience, Privileged, PrivilegedKind, Provenance, Span, Task,
};
use sven_sdk::atif::{AgentProfile, Trajectory};

const REFERENCE: &str = "the quick brown fox";

fn experience(kind: &str, reference: Option<&str>, output: Option<&str>) -> Experience {
    let span = Span::new(Digest::of(REFERENCE.as_bytes()), 0, REFERENCE.len() as u64).unwrap();
    let privileged = reference
        .map(|r| Privileged {
            kind: PrivilegedKind::Reference,
            content: r.into(),
            span: Some(span.clone()),
        })
        .into_iter()
        .collect();
    let task = Task::new(
        kind,
        vec![span],
        Environment {
            kind: "closed-book".into(),
            spec: json!({}),
            snapshot: None,
        },
        "Restore: quick the fox brown",
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
        Provenance::new("scripted", &FixedClock::new("2026-09-30T00:00:00.000Z")),
    )
    .unwrap()
}

fn outcome_of(exp: &Experience) -> (Outcome, Strength, serde_json::Value) {
    judged_by(&FormalVerifier::new(), exp)
}

fn judged_by(verifier: &dyn Verifier, exp: &Experience) -> (Outcome, Strength, serde_json::Value) {
    let note = annotation(verifier, &exp.to_task(), exp).unwrap();
    assert_eq!(note.experience, exp.id().unwrap());
    match note.body {
        AnnotationBody::Verdict {
            outcome,
            strength,
            evidence,
        } => (outcome, strength, evidence),
        other => panic!("not a verdict: {other:?}"),
    }
}

#[test]
fn a_whitespace_variant_of_the_reference_passes_at_formal_strength() {
    let (outcome, strength, evidence) = outcome_of(&experience(
        KIND,
        Some(REFERENCE),
        Some("  the quick\n brown\tfox "),
    ));
    assert_eq!(outcome, Outcome::Pass);
    assert_eq!(strength, Strength::Formal);
    assert!(
        !evidence.to_string().contains("quick brown"),
        "evidence records digests, not the reference: {evidence}"
    );
}

#[test]
fn a_different_answer_or_no_answer_fails() {
    let (outcome, _, _) = outcome_of(&experience(KIND, Some(REFERENCE), Some("the brown fox")));
    assert_eq!(outcome, Outcome::Fail);
    let (outcome, _, _) = outcome_of(&experience(KIND, Some(REFERENCE), None));
    assert_eq!(outcome, Outcome::Fail);
}

#[test]
fn it_abstains_on_what_it_cannot_judge() {
    let (outcome, strength, _) =
        outcome_of(&experience("recall", Some(REFERENCE), Some(REFERENCE)));
    assert_eq!((outcome, strength), (Outcome::Abstain, Strength::Formal));
    let (outcome, _, _) = outcome_of(&experience(KIND, None, Some(REFERENCE)));
    assert_eq!(outcome, Outcome::Abstain);
}

#[test]
fn exact_match_normalises_only_what_it_is_configured_to() {
    let producer = Producer {
        name: "test/exact".into(),
        version: "1".into(),
    };
    let exp = experience("recall", Some("Paris"), Some("  PARIS! "));
    let exact = ExactMatchVerifier::new(producer.clone(), Normalisation::WHITESPACE);
    assert_eq!(judged_by(&exact, &exp).0, Outcome::Fail);
    let lenient = ExactMatchVerifier::new(producer, Normalisation::LENIENT);
    assert_eq!(judged_by(&lenient, &exp).0, Outcome::Pass);
    let other_kind = lenient.for_kind("denoise");
    assert_eq!(judged_by(&other_kind, &exp).0, Outcome::Abstain);
}
