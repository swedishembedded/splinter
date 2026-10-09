// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems whose every record is
// content-addressed and traceable to its source, for its clients. If your
// team needs expertise in training-data provenance or reproducible
// pipelines, you can procure our services by sending an email to
// info@swedishembedded.com.

//! A claim: something a person taught an agent in conversation that the
//! model should be able to use later, tied to the exact words that taught it.
//!
//! A [`ClaimProposal`] is what an extractor says: the statement, the
//! question it answers and the words it cites by session step. Nothing in it
//! is trusted. Code rules on it ([`Ruling`]): it becomes a [`Claim`], whose
//! quotes are spans of the session source's step parts, or it is refused with
//! a [`Refusal`]. Every ruling is a [`LedgerEntry`], kept whatever the ruling,
//! so a claim that was refused, replaced or admitted can always be explained.
//!
//! Only the person's own words are evidence. What the agent said is context
//! at most ([`Claim::said_wrong`]) and is never a quote.

use serde::{Deserialize, Serialize};

use crate::digest::{canonical_json, Digest};
use crate::experience::Span;
use crate::release::ReleaseId;
use crate::source::SourceId;

/// What a claim teaches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimKind {
    /// The person said an answer was wrong and gave the right one.
    Correction,
    /// The person stated something about the world, themselves or their
    /// project.
    Fact,
    /// The person showed or confirmed a sequence of tool calls that worked.
    Procedure,
}

impl ClaimKind {
    /// The kind as it is serialized.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Correction => "correction",
            Self::Fact => "fact",
            Self::Procedure => "procedure",
        }
    }
}

/// Words an extractor cites from one step of a session, before they are
/// checked against it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CitedQuote {
    /// The ATIF `step_id` the words are said to be in.
    pub step: u64,
    /// The words, claimed to be verbatim.
    pub text: String,
}

/// A claim as an extractor proposed it; see the module documentation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimProposal {
    /// What it teaches.
    pub kind: ClaimKind,
    /// The claim, standing on its own.
    pub statement: String,
    /// The question the statement answers.
    pub question: String,
    /// The person's words that support it, each in a user step.
    pub quotes: Vec<CitedQuote>,
    /// Words of tool observations that support it (a procedure's proof),
    /// each in the observation of the cited agent step.
    pub observations: Vec<CitedQuote>,
    /// The agent steps whose tool calls make up a procedure, in order.
    pub calls: Vec<u64>,
    /// What the agent said that was wrong (corrections): context only.
    pub said_wrong: Option<String>,
}

/// Words of a session step, resolved to bytes of its stored part.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Quote {
    /// The ATIF `step_id`.
    pub step: u64,
    /// The words.
    pub text: String,
    /// Where they are: a span of the step's part in the session source.
    pub span: Span,
}

/// The content address of a [`Claim`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ClaimId(pub Digest);

impl std::fmt::Display for ClaimId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// A proposal that passed the gates: every quote found in the session.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Claim {
    /// What it teaches.
    pub kind: ClaimKind,
    /// The claim, standing on its own.
    pub statement: String,
    /// The question the statement answers.
    pub question: String,
    /// The session it comes from.
    pub session: SourceId,
    /// The person's words that support it.
    pub quotes: Vec<Quote>,
    /// The tool observations that support it.
    pub observations: Vec<Quote>,
    /// The agent steps whose tool calls make up a procedure.
    pub calls: Vec<u64>,
    /// What the agent said that was wrong: context, never trained on.
    pub said_wrong: Option<String>,
}

impl Claim {
    /// The claim's content address.
    pub fn id(&self) -> Result<ClaimId, serde_json::Error> {
        Ok(ClaimId(Digest::of(&canonical_json(self)?)))
    }
}

/// Why a proposal was not admitted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum Refusal {
    /// A field is empty or too long to be a claim.
    Malformed {
        /// What is wrong.
        detail: String,
    },
    /// A cited step is not in the session.
    UnknownStep {
        /// The step.
        step: u64,
    },
    /// A quote cites what the agent said: the agent's words are never
    /// evidence.
    AssistantEvidence {
        /// The agent step.
        step: u64,
    },
    /// No quote of the person's supports it.
    NoUserEvidence,
    /// A quote is not verbatim in the step it cites.
    QuoteNotVerbatim {
        /// The step.
        step: u64,
        /// The words that are not in it.
        quote: String,
    },
    /// The statement carries a number, name, date or quoted term that the
    /// cited words do not.
    UnsupportedTerm {
        /// `number`, `name` or `quoted_term`.
        term_kind: String,
        /// The term.
        term: String,
    },
    /// A procedure without the tool calls or the observation that proved it,
    /// or tool calls cited by a claim that is not a procedure.
    Procedure {
        /// What is wrong.
        detail: String,
    },
    /// What the agent is said to have got wrong was never said.
    WrongAnswerNotSaid,
    /// An admitted claim already establishes it.
    Duplicate {
        /// That claim.
        of: ClaimId,
    },
}

impl Refusal {
    /// The reason as a stable name, for counting.
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::Malformed { .. } => "malformed",
            Self::UnknownStep { .. } => "unknown_step",
            Self::AssistantEvidence { .. } => "assistant_evidence",
            Self::NoUserEvidence => "no_user_evidence",
            Self::QuoteNotVerbatim { .. } => "quote_not_verbatim",
            Self::UnsupportedTerm { .. } => "unsupported_term",
            Self::Procedure { .. } => "procedure",
            Self::WrongAnswerNotSaid => "wrong_answer_not_said",
            Self::Duplicate { .. } => "duplicate",
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Malformed { detail } | Self::Procedure { detail } => f.write_str(detail),
            Self::UnknownStep { step } => write!(f, "step {step} is not in the session"),
            Self::AssistantEvidence { step } => write!(
                f,
                "step {step} is the agent's: its words are context, never evidence"
            ),
            Self::NoUserEvidence => f.write_str("no quote of the person's own words supports it"),
            Self::QuoteNotVerbatim { step, quote } => {
                write!(f, "{quote:?} is not verbatim in step {step}")
            }
            Self::UnsupportedTerm { term_kind, term } => write!(
                f,
                "the statement states the {term_kind} {term:?}, which the cited words do not"
            ),
            Self::WrongAnswerNotSaid => f.write_str(
                "the wrong answer attributed to the agent was never said in the session",
            ),
            Self::Duplicate { of } => write!(f, "claim {of} already establishes it"),
        }
    }
}

/// What the gates decided about a proposal.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "ruling", rename_all = "snake_case")]
pub enum Ruling {
    /// Admitted. It supersedes the earlier admitted claims on the same
    /// question that it disagrees with; they stay in the ledger.
    Admitted {
        /// The claim.
        claim: Claim,
        /// The claims it replaces as the answer to its question.
        supersedes: Vec<ClaimId>,
    },
    /// Refused.
    Refused {
        /// Why.
        reason: Refusal,
    },
}

/// One ruling, kept: which proposal of which extraction, from which session,
/// and what was decided.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerEntry {
    /// The claim set the proposal is in.
    pub claim_set: Digest,
    /// Its position among that set's proposals, in set order.
    pub index: usize,
    /// The session it was proposed from.
    pub session: SourceId,
    /// The proposal, as made.
    pub proposal: ClaimProposal,
    /// What was decided.
    pub ruling: Ruling,
}

/// The proposals one extraction made for one session, or why it made none.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionClaims {
    /// The session source.
    pub session: SourceId,
    /// The proposals, in the order the extractor made them.
    pub proposals: Vec<ClaimProposal>,
    /// Why the extractor produced nothing usable for the session, when it
    /// did not: the reply stayed malformed after correction, or the call
    /// ended without one.
    pub failure: Option<String>,
}

/// The result of one extraction run: the proposals for each session, kept
/// whole so the gates can be run again, on this set or another ledger.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimSet {
    /// The model that proposed them, as an experience records a model.
    pub extractor: String,
    /// One entry per session, in the order the sessions were given.
    pub sessions: Vec<SessionClaims>,
}

impl std::error::Error for Refusal {}

/// What a task made from a claim is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskRole {
    /// The claim's own question: trained on.
    Question,
    /// A differently worded question about the same fact: trained on.
    Train,
    /// A differently worded question about the same fact that no training
    /// record contains: the claim's own stopping and gate set, only measured.
    Stopping,
    /// What the person first asked, before the agent was corrected: answered
    /// with the verified corrected answer, trained on.
    Hindsight,
}

/// A task and the claim it was made from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimTaskLink {
    /// The claim.
    pub claim: ClaimId,
    /// The task, by its content address.
    pub task: Digest,
    /// What it is for.
    pub role: TaskRole,
}

/// The release that first took a claim in: from then on the claim is
/// replayed, not taught again.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Absorption {
    /// The claim.
    pub claim: ClaimId,
    /// The release trained on it.
    pub release: ReleaseId,
}
