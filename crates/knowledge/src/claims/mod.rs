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
//! * [`agreement`] - of several extraction passes over one session, only the
//!   claims every pass produced are kept.
//! * [`rule`] - the gates that decide one proposal on its own, in order:
//!   it is well formed; every quote is verbatim in the user step it cites,
//!   and a quote of the agent's words is refused whatever it says; a
//!   procedure has its tool calls and the observation that proved it; what
//!   the agent is said to have got wrong was said; every number, name and
//!   quoted term of the statement occurs in the cited words ([`terms`]); it
//!   does not give personal data of someone other than the person
//!   ([`personal`]).
//! * [`Ledger`] - the rulings kept, and the gates that need them: claims are
//!   paired by the names they share ([`pairing`]); a judge, or without one
//!   the statements, decides whether a later claim supersedes the earlier
//!   (it contradicts it), reinforces it (the same fact said again, recorded,
//!   never refused) or is separate. Claims are never deleted: a superseded
//!   or refused one stays with its reason.
//!
//! * [`task`] - a live claim as a task: the question it answers, graded
//!   against the statement by [`answer`], grounded in the quotes' spans of
//!   the session.
//!
//! Without a judge no model is involved in a ruling: the same proposals and
//! ledger give the same rulings.

pub mod agreement;
pub mod answer;
pub mod extract;
pub mod forms;
mod gates;
pub mod judge;
mod ledger;
pub mod pairing;
pub mod personal;
pub mod reply;
pub mod task;
pub mod terms;

pub use gates::{rule, MAX_QUESTION_CHARS, MAX_STATEMENT_CHARS};
pub use judge::{ClaimJudge, JudgeError, PairVerdict};
pub use ledger::{GateError, Ledger, RuleRequest};
