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
//! * [`replay`] - an experience's code calls run again in the environment
//!   it records, each result compared with the one it observed.

#![warn(missing_docs)]

mod budget;
pub mod converse;
pub mod critic;
pub mod judge;
pub mod proposer;
pub mod repair;
pub mod replay;
pub mod run_code;
pub mod solve;
mod system_prompt;
