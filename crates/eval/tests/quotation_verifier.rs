// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: the quotation verifier passes an answer only when every passage it
//! presents in quotation marks is in the task's own source text and it gives
//! the advice the reference records; it fails an invented quotation, an
//! answer that quotes nothing, and an answer that quotes real text but not
//! the reference; it abstains when the source text is not available. Its
//! verdicts are at formal strength and name counts and digests, never the
//! quoted text.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use serde_json::json;
use splinter_core::annotation::{AnnotationBody, Outcome, Producer, Strength};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{
    Digest, Environment, Experience, Privileged, PrivilegedKind, Provenance, Span, Task,
};
use splinter_eval::verifiers::quotation::{
    quotations, words, EvidenceText, QuotationPolicy, QuotationVerifier, TextIndex,
};
use splinter_eval::verifiers::{annotation, Verifier, VerifyError};
use sven_sdk::atif::{AgentProfile, Trajectory};

const LETTER: &str = "Dear Peter,--I advise you to fix a habit of study every morning before \
    you do anything else.\nNever let a day pass without reading something of history or ethics, \
    and always write down what you have read, for the memory fails what the pen has not fixed.\n\
    Do not spend your evenings in idle company.";

const ADVICE: &str =
    "I advise you to fix a habit of study every morning before you do anything else. \
    Never let a day pass without reading something of history or ethics, and always write down \
    what you have read, for the memory fails what the pen has not fixed.";

/// The source text of every task, as a fixed list.
struct Fixed(Vec<String>);

impl EvidenceText for Fixed {
    fn of(&self, _task: &Task) -> Result<Vec<String>, VerifyError> {
        Ok(self.0.clone())
    }
}

fn experience(output: Option<&str>) -> Experience {
    let span = Span::new(Digest::of(LETTER.as_bytes()), 0, LETTER.len() as u64).unwrap();
    let task = Task::new(
        "advise",
        vec![span.clone()],
        Environment {
            kind: "closed-book".into(),
            spec: json!({}),
            snapshot: None,
        },
        "I am nineteen and my mornings slip away in idleness; what would you advise?",
        vec![Privileged {
            kind: PrivilegedKind::Reference,
            content: ADVICE.into(),
            span: Some(span),
        }],
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

fn verifier(evidence: Vec<String>) -> QuotationVerifier {
    QuotationVerifier::new(
        Producer {
            name: "test/quotation".into(),
            version: "1".into(),
        },
        Box::new(Fixed(evidence)),
        QuotationPolicy {
            min_words: 8,
            min_reference_recall: 0.5,
        },
    )
}

fn verdict(evidence: Vec<String>, output: Option<&str>) -> (Outcome, Strength, serde_json::Value) {
    let exp = experience(output);
    let note = annotation(&verifier(evidence) as &dyn Verifier, &exp.to_task(), &exp).unwrap();
    match note.body {
        AnnotationBody::Verdict {
            outcome,
            strength,
            evidence,
        } => (outcome, strength, evidence),
        other => panic!("not a verdict: {other:?}"),
    }
}

fn source() -> Vec<String> {
    vec![LETTER.to_string()]
}

#[test]
fn an_answer_that_quotes_real_text_and_gives_the_advice_passes() {
    let answer = format!("In my letter to Peter Carr, I advised: \"{ADVICE}\"");
    let (outcome, strength, evidence) = verdict(source(), Some(&answer));
    assert_eq!(outcome, Outcome::Pass, "{evidence}");
    assert_eq!(strength, Strength::Formal);
}

#[test]
fn a_quotation_is_found_whatever_its_line_breaks_and_punctuation() {
    let answer = "As I told him: \"Never let a day pass without reading something of history or ethics, \
        and always write down what you have read\" - and I advise you to fix a habit of study every morning before you do anything else.";
    let (outcome, _, evidence) = verdict(source(), Some(answer));
    assert_eq!(outcome, Outcome::Pass, "{evidence}");
}

#[test]
fn an_invented_quotation_fails_even_beside_a_real_one() {
    let answer = format!(
        "I wrote: \"{ADVICE}\" and also \"the surest road to wealth is to borrow boldly and repay never\"."
    );
    let (outcome, _, evidence) = verdict(source(), Some(&answer));
    assert_eq!(outcome, Outcome::Fail, "{evidence}");
    assert_eq!(evidence["quotations"], 2);
    assert_eq!(evidence["not_in_source"], 1);
}

#[test]
fn an_answer_that_quotes_nothing_fails() {
    let (outcome, _, _) = verdict(
        source(),
        Some("Study every morning and avoid idle company."),
    );
    assert_eq!(outcome, Outcome::Fail);
    let (outcome, _, _) = verdict(source(), None);
    assert_eq!(outcome, Outcome::Fail, "no answer is not a pass");
}

#[test]
fn real_text_that_is_not_the_advice_fails() {
    let answer =
        "I once wrote: \"Do not spend your evenings in idle company\" and that is all I will say.";
    let (outcome, _, evidence) = verdict(source(), Some(answer));
    assert_eq!(evidence["not_in_source"], 0, "the quotation is real");
    assert_eq!(
        outcome,
        Outcome::Fail,
        "but it does not give the reference advice: {evidence}"
    );
}

#[test]
fn without_source_text_it_abstains() {
    let answer = format!("I advised: \"{ADVICE}\"");
    let (outcome, _, _) = verdict(Vec::new(), Some(&answer));
    assert_eq!(outcome, Outcome::Abstain);
}

#[test]
fn the_verdict_names_counts_and_never_the_quoted_text() {
    let answer = format!("I advised: \"{ADVICE}\"");
    let (_, _, evidence) = verdict(source(), Some(&answer));
    let text = evidence.to_string();
    assert!(!text.contains("habit of study"), "{text}");
    assert!(evidence["quotations"].is_number() && evidence["reference_recall"].is_number());
}

#[test]
fn a_text_index_finds_a_passage_by_whole_words_whichever_edition_prints_it() {
    let index = TextIndex::new([
        "We hold these truths to be self-evident,\nthat all men are created equal, that they\nare endowed by their Creator with certain unalienable rights.",
        "The pursuit of happiness is a right of every citizen of the union",
    ]);
    assert!(
        index.contains("we hold these truths to be self-evident that all men are created equal")
    );
    assert!(index.contains("pursuit of happiness is a right of every citizen"));
    assert!(
        !index.contains("pursuit of happiness is a righ"),
        "words, not letters"
    );
    assert!(
        !index.contains("created equal pursuit of happiness"),
        "a passage does not span two texts"
    );
    assert!(!index.contains("   "), "nothing is not found");
}

#[test]
fn quotations_are_the_marked_passages_of_enough_words() {
    let answer = "He said \"we hold these truths to be self-evident\" and \u{201c}no\u{201d} and \"two words\".";
    assert_eq!(
        quotations(answer, 5),
        ["we hold these truths to be self-evident"]
    );
    assert_eq!(quotations(answer, 1).len(), 3);
    assert_eq!(
        words("Hello, World-wide  Web!"),
        ["hello", "world", "wide", "web"]
    );
}
