// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Solving through sven's engine: one delegated task attempt, and a task
//! solved in the environment it records.
//!
//! * [`runner`] - run or resume an attempt: the engine turn raced against
//!   its wall-clock, tool-round and usage limits and a cross-process cancel,
//!   the completion checks, a resumable checkpoint.
//! * [`events`] - the engine's event stream mapped into the run's trace,
//!   tallying the usage the limits and the outcome read.
//! * [`outcome`] - the structured, reviewable result: status, changed
//!   files, check evidence, usage, artifacts.
//! * [`budget`] - the rule that decides a usage limit has been spent.
//! * [`solve`] - a task solved in exactly the environment it records:
//!   closed-book with no tools, or a runtime with the one tool
//!   [`run_code`], bounded by sven's run options; the result is what an
//!   experience records.
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
//!
//! Everything an attempt produces lands in its run directory under the
//! state root, so a run is reviewable - and resumable - after the process
//! that wrote it is gone.

#![warn(missing_docs)]

pub mod budget;
pub mod critic;
pub mod events;
pub mod judge;
pub mod outcome;
pub mod repair;
pub mod run_code;
pub mod runner;
pub mod solve;

pub use outcome::{Outcome, Status};
pub use runner::{resume, run, AttemptOptions};
