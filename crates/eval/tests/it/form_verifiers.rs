// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: the form verifiers grade by code what a persona fine-tune can erode
//! and trivia recall cannot show. The final-number verifier passes a worked
//! sum whose last stated number is the reference, at formal strength; the
//! line-count verifier passes an answer with exactly the non-empty lines the
//! reference counts. No answer fails; a reference that is not a number, or a
//! task without exactly one reference, is an abstention.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use atif::{AgentProfile, Trajectory};
use serde_json::json;
use splinter_core::annotation::{AnnotationBody, Outcome, Producer, Strength};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{
    Digest, Environment, Experience, Privileged, PrivilegedKind, Provenance, Span, Task,
};
use splinter_eval::verifiers::form::{FinalNumberVerifier, LineCountVerifier};
use splinter_eval::verifiers::{annotation, Verifier};

fn experience(kind: &str, reference: Option<&str>, output: Option<&str>) -> Experience {
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
        kind,
        vec![span],
        Environment {
            kind: "closed-book".into(),
            spec: json!({}),
            snapshot: None,
        },
        "Work it out.",
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

fn producer() -> Producer {
    Producer {
        name: "test/form".into(),
        version: "1".into(),
    }
}

fn judged(verifier: &dyn Verifier, exp: &Experience) -> (Outcome, Strength) {
    let note = annotation(verifier, &exp.to_task(), exp).unwrap();
    match note.body {
        AnnotationBody::Verdict {
            outcome, strength, ..
        } => (outcome, strength),
        other => panic!("not a verdict: {other:?}"),
    }
}

#[test]
fn a_worked_sum_passes_by_its_last_number_at_formal_strength() {
    let verifier = FinalNumberVerifier::new(producer());
    let sum = "15 pencils at 12 cents is 180 cents; 500 minus 180 is 320.\nThe answer is 320";
    assert_eq!(
        judged(&verifier, &experience("arithmetic", Some("320"), Some(sum))),
        (Outcome::Pass, Strength::Formal)
    );
    assert_eq!(
        judged(
            &verifier,
            &experience(
                "arithmetic",
                Some("320"),
                Some("The answer is 320, not 180.")
            )
        )
        .0,
        Outcome::Fail,
        "the last number stated is the answer"
    );
    assert_eq!(
        judged(&verifier, &experience("arithmetic", Some("320"), None)).0,
        Outcome::Fail
    );
    assert_eq!(
        judged(
            &verifier,
            &experience("arithmetic", Some("three hundred"), Some("320"))
        )
        .0,
        Outcome::Abstain,
        "a reference that is not a number cannot be compared"
    );
    assert_eq!(
        judged(&verifier, &experience("arithmetic", None, Some("320"))).0,
        Outcome::Abstain
    );
}

#[test]
fn a_list_passes_with_exactly_the_lines_asked_for() {
    let verifier = LineCountVerifier::new(producer());
    assert_eq!(
        judged(
            &verifier,
            &experience("format", Some("3"), Some("apple\nbanana\n\ncherry\n"))
        ),
        (Outcome::Pass, Strength::Formal)
    );
    assert_eq!(
        judged(
            &verifier,
            &experience("format", Some("3"), Some("Sure:\napple\nbanana\ncherry"))
        )
        .0,
        Outcome::Fail
    );
    assert_eq!(
        judged(&verifier, &experience("format", Some("3"), None)).0,
        Outcome::Fail
    );
    assert_eq!(
        judged(
            &verifier,
            &experience("format", Some("three"), Some("a\nb\nc"))
        )
        .0,
        Outcome::Abstain
    );
}
