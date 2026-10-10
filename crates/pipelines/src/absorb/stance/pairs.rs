// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data that teaches a model to hold
// what it knows and to say what it does not, for its clients. If your team
// needs expertise in sycophancy and hallucination control in fine-tuned
// models, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The pairs: a conversation, the reply to prefer and the reply to reject.
//!
//! The reply to reject is what the model really does: a sample of the
//! supervised candidate, accepted only when code finds it giving in,
//! inventing support or inventing a specific. The reply to prefer is written
//! by the teacher, shown what is true, and accepted only when code finds it
//! holding the true value, adding nothing to a statement, or declining
//! without a guess. Whatever code cannot verify is not a pair.

use splinter_agent::chat::Turn;
use splinter_knowledge::stance::{adopts, declines, holds, invented_term};
use splinter_orchestrator::error::OrchestratorError;

use super::items::{AcceptItem, DeclineItem, HoldItem, Kind};
use super::voices::Voices;
use super::write::{ACCEPT_INSTRUCTION, HOLD_INSTRUCTION, YIELD_INSTRUCTION};
use super::{REFUSED_CHOSEN, REFUSED_NO_REJECTED};
use crate::abstain::ModelAbstainer;
use crate::raft::{Abstainer, Abstention};

/// The most words of a reply that only acknowledges a statement.
pub const ACCEPT_MAX_WORDS: usize = 50;

/// Where a pair's rejected reply came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rejection {
    /// A sample of the supervised candidate.
    Sampled,
    /// Written by the teacher, no sample having given in.
    Written,
}

/// One pair.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pair {
    /// What it teaches.
    pub kind: Kind,
    /// What the item is called: its question or its statement.
    pub identity: String,
    /// The conversation before the reply.
    pub prompt: Vec<Turn>,
    /// The reply to prefer.
    pub chosen: String,
    /// The reply to reject.
    pub rejected: String,
    /// Where the rejected reply came from.
    pub rejection: Rejection,
    /// Texts a sealed probe may share words with without leaking: the claim
    /// a pair is about.
    pub allowed: Vec<String>,
}

/// A pair made, or why it was not.
pub enum Made {
    /// The pair.
    Pair(Box<Pair>),
    /// The reason, as the report counts it.
    Refused(&'static str),
}

fn words(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Whether `reply` acknowledges `statement` and states nothing of its own:
/// brief, and no number, name or quoted term the statement does not hold.
#[must_use]
pub fn accepts(reply: &str, statement: &str) -> bool {
    words(reply) <= ACCEPT_MAX_WORDS && invented_term(reply, statement).is_none()
}

/// `turns` with `note` added to the last of them, as the teacher is shown
/// it.
fn with_note(turns: &[Turn], note: &str) -> Vec<Turn> {
    let mut shown = turns.to_vec();
    if let Some(last) = shown.last_mut() {
        last.text = format!("{}\n\n{note}", last.text);
    }
    shown
}

/// The pair of a fact the person pushes a falsehood on.
pub(super) fn hold(voices: &Voices<'_>, item: &HoldItem) -> Result<Made, OrchestratorError> {
    let prompt = item.prompt();
    let note = format!("{HOLD_INSTRUCTION}\n{}", item.statement);
    let chosen = voices.first(&voices.teacher, &with_note(&prompt, &note), |reply| {
        holds(reply, &item.key, &item.replacement)
    })?;
    let Some(chosen) = chosen else {
        return Ok(Made::Refused(REFUSED_CHOSEN));
    };
    let gives_in = |reply: &str| adopts(reply, &item.key, &item.replacement);
    let (rejected, rejection) = match voices.first(&voices.student, &prompt, gives_in)? {
        Some(sampled) => (sampled, Rejection::Sampled),
        None => match voices.first(
            &voices.teacher,
            &with_note(&prompt, YIELD_INSTRUCTION),
            gives_in,
        )? {
            Some(written) => (written, Rejection::Written),
            None => return Ok(Made::Refused(REFUSED_NO_REJECTED)),
        },
    };
    Ok(Made::Pair(Box::new(Pair {
        kind: Kind::Hold,
        identity: item.question.clone(),
        prompt,
        chosen,
        rejected,
        rejection,
        allowed: Vec::new(),
    })))
}

/// The pair of a statement the person makes.
pub(super) fn accept(voices: &Voices<'_>, item: &AcceptItem) -> Result<Made, OrchestratorError> {
    let prompt = item.prompt();
    let chosen = voices.first(
        &voices.teacher,
        &with_note(&prompt, ACCEPT_INSTRUCTION),
        |reply| accepts(reply, &item.statement),
    )?;
    let Some(chosen) = chosen else {
        return Ok(Made::Refused(REFUSED_CHOSEN));
    };
    let invents = |reply: &str| invented_term(reply, &item.statement).is_some();
    let Some(rejected) = voices.first(&voices.student, &prompt, invents)? else {
        return Ok(Made::Refused(REFUSED_NO_REJECTED));
    };
    Ok(Made::Pair(Box::new(Pair {
        kind: Kind::Accept,
        identity: item.message.clone(),
        prompt,
        chosen,
        rejected,
        rejection: Rejection::Sampled,
        allowed: vec![item.statement.clone()],
    })))
}

/// The pair of a question nobody could answer. `known` is what the night
/// holds as true - the live claims' statements - which a reply may state
/// without inventing it.
pub(super) fn decline(
    voices: &Voices<'_>,
    item: &DeclineItem,
    known: &str,
) -> Result<Made, OrchestratorError> {
    let prompt = item.prompt();
    let writer = ModelAbstainer::new(
        voices.ctx(),
        voices.teacher.clone(),
        voices.persona.clone(),
        None,
        voices.cancel.clone(),
    );
    let chosen = writer
        .abstention(&item.question, &[], Abstention::Unknowable)
        .filter(|reply| declines(reply, &item.question));
    let Some(chosen) = chosen else {
        return Ok(Made::Refused(REFUSED_CHOSEN));
    };
    let given = format!("{}\n{known}", item.question);
    let invents = |reply: &str| invented_term(reply, &given).is_some();
    let Some(rejected) = voices.first(&voices.student, &prompt, invents)? else {
        return Ok(Made::Refused(REFUSED_NO_REJECTED));
    };
    Ok(Made::Pair(Box::new(Pair {
        kind: Kind::Decline,
        identity: item.question.clone(),
        prompt,
        chosen,
        rejected,
        rejection: Rejection::Sampled,
        allowed: Vec::new(),
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An acknowledgement adds nothing and is short; an elaboration is
    /// neither, and a long one that adds no term is still no acknowledgement.
    #[test]
    fn an_acknowledgement_is_brief_and_adds_no_term() {
        let statement = "Svc4 uses port 9004.";
        assert!(accepts("Noted, thank you.", statement));
        assert!(accepts("Noted: Svc4 uses port 9004.", statement));
        assert!(!accepts("Noted, it moved there in 2021.", statement));
        assert!(!accepts(&"Noted. ".repeat(ACCEPT_MAX_WORDS + 1), statement));
    }
}
