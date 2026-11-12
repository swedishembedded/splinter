// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements simulated clients, tenants and students
// that behave like the real ones, for its clients who train and test people
// and agents against them. If your team needs expertise in building
// realistic counterpart roles from behaviour data you can procure our
// services by sending an email to info@swedishembedded.com.

//! A counterpart role from behaviour data: select what describes how a
//! type of person behaves, write it as a seat's persona, and measure how
//! well that seat predicts behaviour it was never shown.
//!
//! The selection is the same loop as for any other goal; the goal's role is
//! the counterpart ("play a hesitant tenant"), its probes ask what that
//! person does in a situation, and the material is vignettes, transcripts
//! and descriptions of behaviour. What differs is the artifact: not
//! learned memory but a [`SeatSpec`] a simulated seat is prepped with (and,
//! later, the data an adapter is trained on).
//!
//! Fidelity is measured against held-out behaviour of the same type: the
//! situations the persona was not selected for. It is the share of them a
//! reader acting from the persona alone gets right.

use splinter_agent::sven::seat::SeatSpec;
use splinter_knowledge::goal::corpus::{Candidate, SectionId};
use splinter_knowledge::goal::facts::{Fact, FactExtractor, Form};
use splinter_knowledge::goal::gaps::{pass_rate, Probe, ProbeResult};

use super::participant::{Query, Reader};

/// The behaviour `subject` shows, in the order the sections state it:
/// every [`Form::Behaviour`] fact about that type, as the sentence it was
/// written in.
#[must_use]
pub fn behaviour_of(
    subject: &str,
    sections: &[&Candidate],
    extractor: &dyn FactExtractor,
) -> Vec<Fact> {
    let subject = subject.to_lowercase();
    sections
        .iter()
        .flat_map(|c| extractor.facts(&c.text))
        .filter(|f| f.key.form == Form::Behaviour && f.key.entity == subject)
        .collect()
}

/// The persona text for `subject`: who they are, then how they behave.
#[must_use]
pub fn persona(subject: &str, behaviour: &[Fact]) -> String {
    let mut text = format!("You are a {subject} client. This is how you behave:\n");
    for fact in behaviour {
        text.push_str(&format!(
            "When {}, a {} client {}.\n",
            fact.key.attribute, fact.key.entity, fact.value
        ));
    }
    text
}

/// The seat specification a simulated user is prepped with for `role`, from
/// a persona.
///
/// # Errors
///
/// When the specification does not parse as sven's [`SeatSpec`].
pub fn seat_spec(
    id: &str,
    role: &str,
    objective: &str,
    persona: &str,
) -> Result<SeatSpec, serde_json::Error> {
    serde_json::from_value(serde_json::json!({
        "id": id,
        "persona": persona,
        "goals": [objective],
        "opening": format!("Hello, I am the {role}. I would like to talk about my situation."),
    }))
}

/// How well a reader acting from `persona` alone predicts `held_out`: the
/// pass rate of the questions, each asked with the persona as its only
/// material; `None` when there are none.
#[must_use]
pub fn fidelity(reader: &dyn Reader, persona: &str, held_out: &[Probe]) -> Option<f64> {
    let queries: Vec<Query> = held_out
        .iter()
        .map(|p| Query {
            question: p.question.clone(),
            material: vec![persona.to_string()],
        })
        .collect();
    let results: Vec<ProbeResult> = held_out
        .iter()
        .zip(reader.read_all(&queries))
        .map(|(probe, answer)| ProbeResult {
            probe: probe.clone(),
            answer,
        })
        .collect();
    pass_rate(&results)
}

/// The fidelity of the persona that the sections `learned` give `subject`.
#[must_use]
pub fn learned_fidelity(
    subject: &str,
    candidates: &[Candidate],
    learned: &[SectionId],
    extractor: &dyn FactExtractor,
    reader: &dyn Reader,
    held_out: &[Probe],
) -> Option<f64> {
    let sections: Vec<&Candidate> = candidates
        .iter()
        .filter(|c| learned.contains(&c.id))
        .collect();
    fidelity(
        reader,
        &persona(subject, &behaviour_of(subject, &sections, extractor)),
        held_out,
    )
}
