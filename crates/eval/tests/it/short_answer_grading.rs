// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: short-answer grading ignores what is format and keeps what is
//! content. Markdown emphasis and code marks, thousands separators, quotes
//! around the answer and a final full stop are format. A final short answer
//! is taken out of a sentence only when exactly one candidate of the
//! reference's type is there (a number for a numeric reference, the one
//! emphasised phrase otherwise); a hedge that lists several never passes.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use atif::{AgentProfile, Trajectory};
use serde_json::json;
use splinter_core::annotation::{AnnotationBody, Outcome, Producer};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{
    Digest, Environment, Experience, Privileged, PrivilegedKind, Provenance, Span, Task,
};
use splinter_eval::verifiers::formal::{ExactMatchVerifier, StatedReferenceVerifier};
use splinter_eval::verifiers::normalise::Normalisation;
use splinter_eval::verifiers::{annotation, Verifier};

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

fn verdict(verifier: &dyn Verifier, reference: &str, output: &str) -> Outcome {
    let exp = experience(Some(reference), Some(output));
    let note = annotation(verifier, &exp.to_task(), &exp).unwrap();
    match note.body {
        AnnotationBody::Verdict { outcome, .. } => outcome,
        other => panic!("not a verdict: {other:?}"),
    }
}

fn producer() -> Producer {
    Producer {
        name: "test/grading".into(),
        version: "1".into(),
    }
}

fn exact(reference: &str, output: &str) -> Outcome {
    verdict(
        &ExactMatchVerifier::new(producer(), Normalisation::LENIENT),
        reference,
        output,
    )
}

fn stated(reference: &str, output: &str) -> Outcome {
    verdict(
        &StatedReferenceVerifier::new(producer(), Normalisation::LENIENT),
        reference,
        output,
    )
}

#[test]
fn format_is_not_content_for_the_exact_reference() {
    for (reference, answer) in [
        ("9123", "9,123"),
        ("9,123", "9123"),
        ("1234567", "1,234,567"),
        ("K", "**K**"),
        ("K", "`K`"),
        ("K", "*K*"),
        ("Paris", "\"Paris\""),
        ("Paris", "\u{201c}Paris.\u{201d}"),
        ("Paris", "'Paris'"),
        ("Paris", "**Paris**."),
        ("3.5", "3.5."),
    ] {
        assert_eq!(
            exact(reference, answer),
            Outcome::Pass,
            "{reference}: {answer}"
        );
    }
}

#[test]
fn format_normalisation_does_not_merge_different_content() {
    for (reference, answer) in [
        ("123", "1,23"),
        ("9123", "9,12"),
        ("9123", "9,1234"),
        ("Paris", "Pari's"),
        ("6", "2 * 3"),
        ("ab", "a*b"),
    ] {
        assert_eq!(
            exact(reference, answer),
            Outcome::Fail,
            "{reference}: {answer}"
        );
    }
}

#[test]
fn a_sole_candidate_is_taken_out_of_a_sentence() {
    for (reference, answer) in [
        ("42", "The answer is 42."),
        ("3000", "It is 3,000 millilitres."),
        ("Paris", "The capital is **Paris**."),
        ("O(log n)", "The cost is **O(log n)** per lookup."),
    ] {
        assert_eq!(
            exact(reference, answer),
            Outcome::Pass,
            "{reference}: {answer}"
        );
    }
}

#[test]
fn a_hedge_or_a_different_candidate_is_never_taken_as_the_answer() {
    for (reference, answer) in [
        ("8789", "8789 or 8790 or 3000"),
        ("8789", "It is 8789 or maybe 8790."),
        ("42", "15 plus 27 is 42."),
        ("Paris", "It is **Paris** or **Lyon**."),
        ("Paris", "The capital is **Lyon**."),
        ("Paris", "Paris, Lyon, Nice"),
        ("Paris", "The capital is Paris"),
        ("42", "The answer is 43."),
    ] {
        assert_eq!(
            exact(reference, answer),
            Outcome::Fail,
            "{reference}: {answer}"
        );
    }
}

#[test]
fn a_long_verbose_answer_with_one_emphasised_answer_is_decided_by_stated() {
    let lead = "To convert, multiply the volume in litres by one thousand, since a litre is a thousand millilitres. ";
    assert_eq!(
        stated(
            "3000",
            &format!("{lead}So there are **3,000** millilitres.")
        ),
        Outcome::Pass
    );
    assert_eq!(
        stated(
            "K",
            &format!("{lead}The chemical symbol for potassium is **K**.")
        ),
        Outcome::Pass
    );
    assert_eq!(
        stated(
            "3000",
            &format!("{lead}So it is **2,000** or **3,000** millilitres.")
        ),
        Outcome::Abstain,
        "a hedge is left to a judge"
    );
    assert_eq!(
        stated(
            "3000",
            &format!("{lead}So the total is 3000 millilitres, or 2000, or 4000.")
        ),
        Outcome::Abstain
    );
    assert_eq!(
        stated(
            "3000",
            &format!("{lead}So there are **2,000** millilitres, not 3000.")
        ),
        Outcome::Abstain,
        "the emphasised answer is another"
    );
}
