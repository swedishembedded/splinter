// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiers that hold a model's claims to the
// source for its clients. If your team needs expertise in catching a model
// that invents what a document says, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec: the grounding verifier fails an answer that states a specific the
//! source does not hold. A specific is a number or a proper name; one is
//! grounded when the source text, the task's instruction or something the
//! other speaker said (the later user turns of a dialogue) contains it. Every
//! number must be grounded; proper names may be ungrounded up to the policy's
//! tolerance. A passage it presents in quotation marks must be in the source
//! word for word (quoting nothing is fine). Words that open a sentence, the pronoun "I" and forms of
//! address are not names. It abstains when the source text is not available
//! and fails when there is no answer. Its verdicts are formal and name
//! counts and digests, never the specifics.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use serde_json::json;
use splinter_core::annotation::{AnnotationBody, Outcome, Producer, Strength};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Environment, Experience, Provenance, Task};
use splinter_eval::verifiers::grounding::{GroundingPolicy, GroundingVerifier};
use splinter_eval::verifiers::quotation::EvidenceText;
use splinter_eval::verifiers::{annotation, Verifier, VerifyError};
use sven_sdk::atif::{AgentProfile, StepOrigin, TraceStep, Trajectory};

const LETTER: &str = "Dear Peter,--I received your letter of the tenth and the three books you \
    sent from Paris. I read Homer in 1760 at Williamsburg under Doctor Small, and I commend the \
    habit of reading every morning.";

struct Fixed(Vec<String>);

impl EvidenceText for Fixed {
    fn of(&self, _task: &Task) -> Result<Vec<String>, VerifyError> {
        Ok(self.0.clone())
    }
}

fn experience(replies: Option<&str>, later_user_turns: &[&str]) -> Experience {
    let task = Task::new(
        "converse",
        vec![],
        Environment::closed_book(),
        "How should I begin reading?",
        vec![],
    )
    .unwrap();
    let mut trajectory = Trajectory::new("ATIF-v1.7", AgentProfile::new("t", "1"));
    let mut steps = vec![TraceStep::new(
        1,
        StepOrigin::User,
        format!("{LETTER}\n\nHow should I begin reading?"),
    )];
    for (n, turn) in later_user_turns.iter().enumerate() {
        steps.push(TraceStep::new(2 + n as u64, StepOrigin::User, *turn));
    }
    trajectory.steps = steps;
    Experience::new(
        task,
        trajectory,
        replies.map(str::to_string),
        Provenance::new("scripted", &FixedClock::new("2026-10-01T00:00:00.000Z")),
    )
    .unwrap()
}

fn verdict(
    evidence: Vec<String>,
    replies: Option<&str>,
    user_turns: &[&str],
    tolerance: usize,
) -> (Outcome, Strength, serde_json::Value) {
    let verifier = GroundingVerifier::new(
        Producer {
            name: "test/grounding".into(),
            version: "1".into(),
        },
        Box::new(Fixed(evidence)),
        GroundingPolicy {
            max_ungrounded_names: tolerance,
            min_quote_words: 6,
        },
    );
    let exp = experience(replies, user_turns);
    let note = annotation(&verifier as &dyn Verifier, &exp.to_task(), &exp).unwrap();
    match note.body {
        AnnotationBody::Verdict {
            outcome,
            strength,
            evidence,
        } => (outcome, strength, evidence),
        other => panic!("not a verdict: {other:?}"),
    }
}

fn letter() -> Vec<String> {
    vec![LETTER.to_string()]
}

#[test]
fn an_answer_that_states_only_what_the_source_holds_passes() {
    let (outcome, strength, evidence) = verdict(
        letter(),
        Some("I read Homer in 1760 under Doctor Small. Begin every morning, as I did at Williamsburg."),
        &[],
        0,
    );
    assert_eq!(outcome, Outcome::Pass, "{evidence}");
    assert_eq!(strength, Strength::Formal);
}

#[test]
fn an_invented_number_fails_whatever_the_tolerance_for_names() {
    let (outcome, _, evidence) = verdict(
        letter(),
        Some("I read Homer in 1762 at Williamsburg."),
        &[],
        9,
    );
    assert_eq!(outcome, Outcome::Fail);
    assert_eq!(evidence["numbers_not_in_source"], 1);
}

#[test]
fn an_invented_name_fails_beyond_the_tolerance_and_passes_within_it() {
    let reply = "I read Homer at Williamsburg, and Mr. Wythe taught me law.";
    assert_eq!(verdict(letter(), Some(reply), &[], 0).0, Outcome::Fail);
    assert_eq!(verdict(letter(), Some(reply), &[], 1).0, Outcome::Pass);
}

#[test]
fn what_the_other_speaker_said_is_grounded() {
    let reply = "Yes, Franklin was kind to me in 1784.";
    assert_eq!(verdict(letter(), Some(reply), &[], 0).0, Outcome::Fail);
    let said = ["Did Franklin help you in 1784?"];
    assert_eq!(verdict(letter(), Some(reply), &said, 0).0, Outcome::Pass);
}

#[test]
fn sentence_openers_the_pronoun_and_forms_of_address_are_not_names() {
    let reply = "Indeed. Sir, I advise reading daily. Reading is a pleasure. Dear friend, begin.";
    assert_eq!(verdict(letter(), Some(reply), &[], 0).0, Outcome::Pass);
}

#[test]
fn no_answer_fails_and_no_source_text_abstains() {
    assert_eq!(verdict(letter(), None, &[], 0).0, Outcome::Fail);
    assert_eq!(
        verdict(vec![String::new()], Some("Read."), &[], 0).0,
        Outcome::Abstain
    );
}

#[test]
fn the_evidence_names_counts_and_never_the_specifics() {
    let (_, _, evidence) = verdict(letter(), Some("I met Wythe in 1762."), &[], 0);
    let text = evidence.to_string();
    assert!(!text.contains("Wythe") && !text.contains("1762"), "{text}");
    assert_eq!(evidence["names_not_in_source"], 1);
    let _ = json!({});
}

#[test]
fn a_quotation_must_be_in_the_source_and_quoting_nothing_is_fine() {
    let said = "I wrote: \"I commend the habit of reading every morning\" and I mean it.";
    assert_eq!(verdict(letter(), Some(said), &[], 0).0, Outcome::Pass);
    let invented = "I wrote: \"the habit of reading every evening is a vice\" and I mean it.";
    let (outcome, _, evidence) = verdict(letter(), Some(invented), &[], 0);
    assert_eq!(outcome, Outcome::Fail);
    assert_eq!(evidence["quotations_not_in_source"], 1);
    assert_eq!(
        verdict(letter(), Some("Read daily."), &[], 0).0,
        Outcome::Pass
    );
}
