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
//!
//! Everything an attempt produces lands in its run directory under the
//! state root, so a run is reviewable - and resumable - after the process
//! that wrote it is gone.

#![warn(missing_docs)]

pub mod budget;
pub mod events;
pub mod outcome;
pub mod run_code;
pub mod runner;
pub mod solve;

pub use outcome::{Outcome, Status};
pub use runner::{resume, run, AttemptOptions};
