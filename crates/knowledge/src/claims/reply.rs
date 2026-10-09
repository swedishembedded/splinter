// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission gates that keep ungrounded
// claims out of training data, for its clients. If your team needs
// expertise in synthetic data quality or provenance, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The one shape an extractor model's reply may take, parsed strictly.
//!
//! ```json
//! {"claims": [{
//!   "kind": "correction",
//!   "statement": "...",
//!   "question": "...",
//!   "quotes": [{"step": 4, "quote": "words the person wrote, verbatim"}],
//!   "observations": [{"step": 7, "quote": "words a tool returned"}],
//!   "calls": [7],
//!   "said_wrong": "what the agent said that was wrong",
//!   "subject": "self" | "world" | "third_party"
//! }]}
//! ```
//!
//! `observations`, `calls` and `said_wrong` may be left out; `subject` may not (a reply without it is sent back for correction). Anything else
//! (a missing or mistyped field, a field the shape does not name) makes the
//! whole reply malformed. The extractor asks for it as a typed call, so sven
//! parses the reply and sends a malformed one back for correction; so it does
//! one that breaks [`ClaimReply::check`], whose message says what to change.
//! Nothing in a well-formed reply is trusted: see [`super::rule`].

use schemars::JsonSchema;
use serde::Deserialize;
use splinter_core::claim::{CitedQuote, ClaimKind, ClaimProposal, ClaimSubject};

use super::{MAX_QUESTION_CHARS, MAX_STATEMENT_CHARS};

/// The most claims one session's reply may hold: a session teaches a handful
/// of things, and a reply of hundreds is the model repeating itself.
pub const MAX_CLAIMS_PER_SESSION: usize = 40;

/// An extractor's reply: the claims it proposes for one session.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ClaimReply {
    claims: Vec<WireClaim>,
}

/// What a claim teaches.
#[derive(Clone, Copy, Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum WireKind {
    Correction,
    Fact,
    Procedure,
}

/// One claim as the model wrote it, before anything about it is trusted.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct WireClaim {
    kind: WireKind,
    /// The claim, standing on its own.
    statement: String,
    /// The question the statement answers.
    question: String,
    /// The person's words that support it: the step they are in and the
    /// words, copied exactly.
    quotes: Vec<WireQuote>,
    #[serde(default)]
    observations: Vec<WireQuote>,
    #[serde(default)]
    calls: Vec<u64>,
    #[serde(default)]
    said_wrong: Option<String>,
    /// Whom the claim is about; a reply that leaves it out is sent back.
    #[serde(default)]
    subject: Option<WireSubject>,
}

/// Whom a claim is about.
#[derive(Clone, Copy, Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum WireSubject {
    #[serde(rename = "self")]
    Own,
    World,
    ThirdParty,
}

/// Words cited from a step.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct WireQuote {
    step: u64,
    quote: String,
}

impl ClaimReply {
    /// What the reply must satisfy beyond its shape; the message says what
    /// to change, and is what the model is sent back.
    pub fn check(&self) -> Result<(), String> {
        if self.claims.len() > MAX_CLAIMS_PER_SESSION {
            return Err(format!(
                "{} claims is too many: reply with the at most {MAX_CLAIMS_PER_SESSION} things \
                 the person taught that matter most",
                self.claims.len()
            ));
        }
        for (i, claim) in self.claims.iter().enumerate() {
            for (field, text, limit) in [
                ("statement", &claim.statement, MAX_STATEMENT_CHARS),
                ("question", &claim.question, MAX_QUESTION_CHARS),
            ] {
                let chars = text.trim().chars().count();
                if chars == 0 || chars > limit {
                    return Err(format!(
                        "claim {i}: the {field} must be between 1 and {limit} characters, not {chars}"
                    ));
                }
            }
            if claim.subject.is_none() {
                return Err(format!(
                    "claim {i}: say whom it is about with `subject`: \"self\" (the person), \
                     \"world\" (the world, their project or tools) or \"third_party\" (someone else)"
                ));
            }
        }
        Ok(())
    }

    /// The proposals the reply holds, in its order.
    #[must_use]
    pub fn into_proposals(self) -> Vec<ClaimProposal> {
        let quotes = |list: Vec<WireQuote>| {
            list.into_iter()
                .map(|q| CitedQuote {
                    step: q.step,
                    text: q.quote,
                })
                .collect()
        };
        self.claims
            .into_iter()
            .map(|c| ClaimProposal {
                kind: match c.kind {
                    WireKind::Correction => ClaimKind::Correction,
                    WireKind::Fact => ClaimKind::Fact,
                    WireKind::Procedure => ClaimKind::Procedure,
                },
                statement: c.statement,
                question: c.question,
                quotes: quotes(c.quotes),
                observations: quotes(c.observations),
                calls: c.calls,
                said_wrong: c.said_wrong,
                subject: c.subject.map(|s| match s {
                    WireSubject::Own => ClaimSubject::Own,
                    WireSubject::World => ClaimSubject::World,
                    WireSubject::ThirdParty => ClaimSubject::ThirdParty,
                }),
            })
            .collect()
    }
}
