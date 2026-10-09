// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements admission gates that keep ungrounded
// claims out of training data, for its clients. If your team needs
// expertise in synthetic data quality or provenance, you can procure our
// services by sending an email to info@swedishembedded.com.

//! What a person taught an agent in conversation, extracted by a model and
//! admitted by code.
//!
//! * [`extract`] - what the extractor model is shown of a session and told
//!   to do, and the one shape ([`reply::ClaimReply`]) its reply may take.
//! * [`rule`] - the gates that decide one proposal on its own, in order:
//!   it is well formed; every quote is verbatim in the user step it cites,
//!   and a quote of the agent's words is refused whatever it says; a
//!   procedure has its tool calls and the observation that proved it; what
//!   the agent is said to have got wrong was said; every number, name and
//!   quoted term of the statement occurs in the cited words ([`terms`]).
//! * [`Ledger`] - the rulings kept, and the gates that need them: a repeat of
//!   a live claim collapses into it; a claim on the same question that
//!   disagrees with live ones supersedes them. Claims are never deleted: a
//!   superseded or refused one stays with its reason.
//!
//! * [`task`] - a live claim as a task: the question it answers, graded
//!   against the statement by [`answer`], grounded in the quotes' spans of
//!   the session.
//!
//! No model is involved in a ruling: the same proposals and ledger give the
//! same rulings.

pub mod answer;
pub mod extract;
mod gates;
mod ledger;
pub mod reply;
pub mod task;
pub mod terms;

pub use gates::{rule, MAX_QUESTION_CHARS, MAX_STATEMENT_CHARS};
pub use ledger::{GateError, Ledger};
