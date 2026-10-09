// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission gates that keep ungrounded
// claims out of training data, for its clients. If your team needs
// expertise in synthetic data quality or provenance, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The gates that decide one proposal against the session it came from; see
//! the module documentation of [`super`].

use splinter_core::claim::{CitedQuote, Claim, ClaimKind, ClaimProposal, Quote, Refusal};
use splinter_core::experience::{PartRef, Span};

use super::personal::third_party_personal;
use super::terms::unsupported_term;
use crate::session::{SessionView, Speaker, Step, TextPart};

/// The longest statement, in characters: a claim is one thing a person
/// taught, standing on its own.
pub const MAX_STATEMENT_CHARS: usize = 600;

/// The longest question, in characters.
pub const MAX_QUESTION_CHARS: usize = 300;

fn malformed(detail: impl Into<String>) -> Refusal {
    Refusal::Malformed {
        detail: detail.into(),
    }
}

fn procedure(detail: impl Into<String>) -> Refusal {
    Refusal::Procedure {
        detail: detail.into(),
    }
}

/// `proposal` ruled on against `view`: the claim it becomes, or why it is
/// refused.
pub fn rule(proposal: &ClaimProposal, view: &SessionView) -> Result<Claim, Refusal> {
    check_shape(proposal)?;
    let quotes = proposal
        .quotes
        .iter()
        .map(|q| user_quote(q, view))
        .collect::<Result<Vec<_>, _>>()?;
    if quotes.is_empty() {
        return Err(Refusal::NoUserEvidence);
    }
    let observations = proposal
        .observations
        .iter()
        .map(|q| observation_quote(q, view))
        .collect::<Result<Vec<_>, _>>()?;
    let calls = called_steps(proposal, view)?;
    if proposal.kind == ClaimKind::Procedure && observations.is_empty() {
        return Err(procedure(
            "a procedure cites the tool observation that proved it",
        ));
    }
    check_wrong_answer(proposal, view)?;

    let mut evidence: Vec<&str> = quotes
        .iter()
        .chain(&observations)
        .map(|q| q.text.as_str())
        .collect();
    evidence.extend(calls.iter().map(|c| c.text.as_str()));
    if let Some(unsupported) = unsupported_term(&proposal.statement, &evidence.join("\n")) {
        return Err(Refusal::UnsupportedTerm {
            term_kind: unsupported.kind.into(),
            term: unsupported.term,
        });
    }
    if let Some(category) = third_party_personal(&proposal.statement, proposal.subject) {
        return Err(Refusal::ThirdPartyPersonal {
            category: category.into(),
        });
    }
    Ok(Claim {
        kind: proposal.kind,
        statement: proposal.statement.trim().to_string(),
        question: proposal.question.trim().to_string(),
        session: view.source().clone(),
        quotes,
        observations,
        calls: proposal.calls.clone(),
        said_wrong: proposal
            .said_wrong
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string),
    })
}

fn check_shape(proposal: &ClaimProposal) -> Result<(), Refusal> {
    for (field, text, limit) in [
        ("statement", &proposal.statement, MAX_STATEMENT_CHARS),
        ("question", &proposal.question, MAX_QUESTION_CHARS),
    ] {
        let chars = text.trim().chars().count();
        if chars == 0 {
            return Err(malformed(format!("the {field} is empty")));
        }
        if chars > limit {
            return Err(malformed(format!(
                "the {field} is {chars} characters; a claim's is at most {limit}"
            )));
        }
    }
    if proposal.kind != ClaimKind::Procedure && !proposal.calls.is_empty() {
        return Err(procedure("tool calls belong to a procedure"));
    }
    if proposal.kind != ClaimKind::Correction && proposal.said_wrong.is_some() {
        return Err(malformed(
            "what the agent said wrong belongs to a correction",
        ));
    }
    Ok(())
}

/// A span of the first occurrence of `quote` in `part`, or `None`.
fn locate(part: &TextPart, view: &SessionView, quote: &str, step: u64) -> Option<Quote> {
    let start = part.text.find(quote)?;
    let span = Span::in_part(
        PartRef {
            source: view.source().clone(),
            name: part.name.clone(),
        },
        part.content.clone(),
        start as u64,
        (start + quote.len()) as u64,
    )
    .ok()?;
    Some(Quote {
        step,
        text: quote.to_string(),
        span,
    })
}

fn step_of(view: &SessionView, id: u64) -> Result<&Step, Refusal> {
    view.step(id).ok_or(Refusal::UnknownStep { step: id })
}

fn trimmed(cited: &CitedQuote) -> Result<&str, Refusal> {
    let text = cited.text.trim();
    if text.is_empty() {
        return Err(malformed(format!("an empty quote of step {}", cited.step)));
    }
    Ok(text)
}

/// A quote of the person's words, verbatim in the user step it cites.
fn user_quote(cited: &CitedQuote, view: &SessionView) -> Result<Quote, Refusal> {
    let step = step_of(view, cited.step)?;
    if step.speaker != Speaker::User {
        return Err(Refusal::AssistantEvidence { step: cited.step });
    }
    let text = trimmed(cited)?;
    step.message
        .as_ref()
        .and_then(|part| locate(part, view, text, cited.step))
        .ok_or_else(|| Refusal::QuoteNotVerbatim {
            step: cited.step,
            quote: text.to_string(),
        })
}

/// A quote of what a tool returned, verbatim in the observation of the agent
/// step it cites.
fn observation_quote(cited: &CitedQuote, view: &SessionView) -> Result<Quote, Refusal> {
    let step = step_of(view, cited.step)?;
    let text = trimmed(cited)?;
    step.observation
        .as_ref()
        .and_then(|part| locate(part, view, text, cited.step))
        .ok_or_else(|| Refusal::QuoteNotVerbatim {
            step: cited.step,
            quote: text.to_string(),
        })
}

/// The tool calls of the cited steps; a procedure cites at least one.
fn called_steps<'a>(
    proposal: &ClaimProposal,
    view: &'a SessionView,
) -> Result<Vec<&'a TextPart>, Refusal> {
    if proposal.kind == ClaimKind::Procedure && proposal.calls.is_empty() {
        return Err(procedure(
            "a procedure cites the agent steps that made its calls",
        ));
    }
    proposal
        .calls
        .iter()
        .map(|&id| {
            step_of(view, id)?
                .calls
                .as_ref()
                .ok_or_else(|| procedure(format!("step {id} made no tool call")))
        })
        .collect()
}

/// What the agent is said to have got wrong was something it said.
fn check_wrong_answer(proposal: &ClaimProposal, view: &SessionView) -> Result<(), Refusal> {
    let Some(wrong) = proposal
        .said_wrong
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    else {
        return Ok(());
    };
    let said = view
        .steps()
        .filter(|s| s.speaker == Speaker::Agent)
        .filter_map(|s| s.message.as_ref())
        .any(|m| m.text.contains(wrong));
    if said {
        Ok(())
    } else {
        Err(Refusal::WrongAnswerNotSaid)
    }
}
