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
//! * [`chat`] - a model's next reply to a conversation it was not in, the
//!   turns given as they are, an assistant turn included.
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
//! * [`skill_planner`] - a planner model asked to propose the skills a
//!   documented domain asks of a learner, its proposal admitted by code.
//! * [`replay`] - an experience's code calls run again in the environment
//!   it records, each result compared with the one it observed.
//! * [`correction`] - a teacher continuing an attempt the student could not
//!   finish, from the state it actually reached: the student's transcript is
//!   replayed through sven's fork into a fresh copy of the task's snapshot
//!   and refused unless it reproduces, the teacher takes the next turn, and
//!   the composed trajectory keeps the student's steps as copied context
//!   with the teacher's actions supervised under per-action provenance.
//! * [`work`] - a coding agent working in one directory it is confined to,
//!   with sven's file, search and shell tools, a cap on tool calls, an
//!   observer of every event, and suspend and resume; and
//!   [`work::solve_in_workspace`], a whole episode on a task that changes a
//!   repository, ending in the patch read back from git.
//! * [`flywheel`] - the flywheel harnesses (invent, validate) run as `sven
//!   harness run` subprocesses, and what their runs left on disk read back.
//! * [`timed`] - a model provider that counts its requests and how long each
//!   waited for its first stream item, so a run that took long can be told
//!   from one that did a lot.

#![warn(missing_docs)]

pub mod blocking;
mod budget;
pub mod chat;
pub mod claim_judge;
pub mod claims;
pub mod converse;
pub mod correction;
pub mod critic;
pub mod demonstration;
pub mod flywheel;
pub mod forms;
pub mod harness_edit;
pub mod harness_work;
pub mod judge;
pub mod knowledge;
pub mod mapper;
pub mod prompts;
pub mod proposer;
pub mod repair;
pub mod replay;
pub mod run_code;
pub mod skill_planner;
pub mod solve;
mod system_prompt;
pub mod teach;
pub mod timed;
pub mod tool_turn;
pub mod typed;
pub mod voice;
pub mod work;
mod workspace_shell;

pub use sven_sdk as sven;
pub use sven_sdk::schemars;
pub use sven_sdk::{CallError, CancelToken, RunConclusion};
