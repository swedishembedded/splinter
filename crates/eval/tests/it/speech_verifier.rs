// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiers that keep a model speaking as a
// person and not about a document, for its clients. If your team needs
// expertise in training personas on a person's writing, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: the speech verifier fails a reply that talks about the material
//! the speaker was shown - "according to the material", "the passage says",
//! "the speaker" - because the student it teaches is never shown any, and a
//! reply that refers to a document it cannot see teaches it to pretend to
//! one. A reply in the person's own voice, even one that mentions their
//! letters or the text of a constitution, passes; and passing establishes
//! nothing (the verdict is a constraint's).

// Helpers outside a test function unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use splinter_core::annotation::{AnnotationBody, Outcome, Producer, Strength};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Environment, Experience, Provenance, Task};
use splinter_eval::verifiers::speech::SpeechVerifier;
use splinter_eval::verifiers::{annotation, Verifier};

fn verdict(reply: Option<&str>) -> (Outcome, Strength, serde_json::Value) {
    let task = Task::new(
        "converse",
        vec![],
        Environment::closed_book(),
        "How should I begin reading?",
        vec![],
    )
    .unwrap();
    let mut trajectory = atif::Trajectory::new("ATIF-v1.7", atif::AgentProfile::new("t", "1"));
    trajectory.steps = vec![];
    let exp = Experience::new(
        task,
        trajectory,
        reply.map(str::to_string),
        Provenance::new("scripted", &FixedClock::new("2026-10-01T00:00:00.000Z")),
    )
    .unwrap();
    let verifier = SpeechVerifier::new(Producer {
        name: "test/speech".into(),
        version: "1".into(),
    });
    match annotation(&verifier as &dyn Verifier, &exp.to_task(), &exp)
        .unwrap()
        .body
    {
        AnnotationBody::Verdict {
            outcome,
            strength,
            evidence,
        } => (outcome, strength, evidence),
        other => panic!("not a verdict: {other:?}"),
    }
}

#[test]
fn a_reply_that_talks_about_the_material_it_was_shown_fails() {
    for leak in [
        "According to the material, one should read each morning.",
        "The passage says that study is a habit.",
        "The speaker expresses a hope that the young man will read.",
        "From the provided text, I gather that history comes first.",
        "The records provided do not specify the year.",
    ] {
        let (outcome, strength, evidence) = verdict(Some(leak));
        assert_eq!(outcome, Outcome::Fail, "{leak}: {evidence}");
        assert_eq!(strength, Strength::Constraint);
        assert_eq!(evidence["document_references"], 1, "{leak}");
    }
}

#[test]
fn a_reply_in_the_persons_own_voice_passes_even_when_it_names_letters_and_texts() {
    for voice in [
        "Begin each morning with history, as I did at Williamsburg.",
        "I wrote of this to my nephew in a letter, and I hold to it.",
        "The text of a constitution binds no generation beyond its own.",
        "I cannot speak to the year from my own recollection.",
    ] {
        let (outcome, _, evidence) = verdict(Some(voice));
        assert_eq!(outcome, Outcome::Pass, "{voice}: {evidence}");
    }
}

#[test]
fn no_reply_is_not_this_verifiers_to_judge() {
    let (outcome, _, _) = verdict(None);
    assert_eq!(outcome, Outcome::Abstain);
}
