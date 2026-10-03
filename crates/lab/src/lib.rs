// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What a learning run needs to measure honestly: verifiers by strength, the
//! holdout rule, paired comparison, pass@k, and the shapes training data is
//! written in.
//!
//! Swedish Embedded AB implements closed-loop learning systems - agents that
//! improve from their own verified experience rather than from hand-written
//! training data - for its clients. If your team needs expertise in agent
//! training loops, sample design, or promotion gating for small models,
//! you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # What the parts are for
//!
//! * [`holdout`] - the rule that decides which records of a training set
//!   are held out for scoring.
//! * [`paired`] - two models graded on the same items, compared only where
//!   both have a verdict.
//! * [`frontier`] - pass@k: a task's pass rate over k attempts and whether
//!   a teacher's answer to it was verified, and whether that makes it worth
//!   training on.
//! * [`WireMessage`] - one message of the `generic-messages-v2` format
//!   training records are written in.
//! * [`SYSTEM_PROMPT`] - the one system turn every model run on a task is
//!   sent and every chat record starts with, so what the policy is trained
//!   on is what it sees when it answers.
//! * [`verifiers`] - verifiers by strength (executable, formal, consistency,
//!   judged), each grading an experience from its output and the task's
//!   privileged material alone, and the composite that annotates every
//!   verdict so the store's decision rule lets the strongest decide.
//! * [`denoise`] - the denoise task family's formal verifier, grading an
//!   experience against the reference passage it never showed the solver.
//! * [`overlap`] - texts that print one passage form a group, so a held-out
//!   split can keep a group whole.

#![warn(missing_docs)]

pub mod denoise;
pub mod frontier;
pub mod holdout;
pub mod overlap;
pub mod paired;
mod system_prompt;
pub mod verifiers;
mod wire;

pub use system_prompt::SYSTEM_PROMPT;
pub use wire::{WireFunction, WireMessage, WireToolCall};
