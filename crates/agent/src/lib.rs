// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Solving through sven's engine: a task solved in the environment it
//! records, and the models that grade, critique and retry it.
//!
//! * [`solve`] - a task solved in exactly the environment it records:
//!   closed-book with no tools, or a runtime with the one tool
//!   [`run_code`], bounded by sven's run options, under Splinter's own
//!   system prompt ([`solve::SYSTEM_PROMPT`]) - the one its training
//!   records show; the result is what an experience records. A teacher's
//!   solve is the same solve with the task's grounding material shown
//!   beside the instruction ([`solve::open_book_prompt`]).
//! * [`judge`] - the judged verifier: a different model grading an answer
//!   closed-book through that same solve, from the output and the task's
//!   reference material alone.
//! * [`critic`] - a model saying what is wrong with a failed attempt and
//!   why, closed-book through that same solve, from the instruction, the
//!   answer and the verifiers' evidence summaries, never the task's
//!   teacher-only material; the critique stored as an experience of its
//!   own.
//! * [`repair`] - a retry with the critique, graded by the task's
//!   verifiers, the critique verified by its outcome, the chain recorded as
//!   relations; and the bounded loop of critique and retry.
//! * [`claims`] - a model asked what a person taught in a session, its
//!   proposals left to the knowledge crate to rule on.
//! * [`claim_judge`] - a model asked what a gate cannot decide by code about
//!   claims: whether a later one supersedes, restates or is apart from an
//!   earlier one.
//! * [`mapper`] - a model asked how a documented variable maps onto a
//!   shared concept, its proposal left to the knowledge crate to admit.
//! * [`prompts`] - a model asked to write user requests of a domain, each
//!   a string, which requests are kept left to the caller.
//! * [`replay`] - an experience's code calls run again in the environment
//!   it records, each result compared with the one it observed.
//! * [`work`] - a coding agent working in one directory it is confined to,
//!   with sven's file, search and shell tools, a cap on tool calls, an
//!   observer of every event, and suspend and resume.

#![warn(missing_docs)]

mod budget;
pub mod claim_judge;
pub mod claims;
pub mod converse;
pub mod critic;
pub mod forms;
pub mod judge;
pub mod mapper;
pub mod prompts;
pub mod proposer;
pub mod repair;
pub mod replay;
pub mod run_code;
pub mod solve;
mod system_prompt;
pub mod typed;
pub mod voice;
pub mod work;

pub use sven_sdk as sven;
pub use sven_sdk::schemars;
pub use sven_sdk::{CallError, CancelToken, RunConclusion};
