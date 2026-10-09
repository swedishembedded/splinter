// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Measurement: what a learning run needs to measure honestly: verifiers by strength,
//! paired comparison and pass@k.
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
//! * [`gate`] - the release gate's four checks, the numbers each records and
//!   how each is decided.
//! * [`metric_gate`] - the release decision over continuous metrics and
//!   their intervals, for a model judged by a metric rather than item by
//!   item: requirements as data, unmeasured as failure.
//! * [`predictive_gate`] - the release gate for a predictive model over
//!   paired held-out units: performance, calibration, retention on
//!   subgroups, serving correctness and data-policy compliance, built on
//!   [`metric_gate`] requirements.
//! * [`speech_reward`] - the reward a spoken take earns (naturalness against
//!   clarity) and how takes of one sentence are compared.
//! * [`significance`] - the paired sign test the gate rests on, handed in by
//!   the model backend.
//! * [`paired`] - two models graded on the same items, compared only where
//!   both have a verdict.
//! * [`frontier`] - pass@k: a task's pass rate over k attempts and whether
//!   a teacher's answer to it was verified, and whether that makes it worth
//!   training on.
//! * [`verifiers`] - verifiers by strength (executable, formal, consistency,
//!   judged), each grading an experience from its output and the task's
//!   privileged material alone, and the composite that annotates every
//!   verdict so the store's decision rule lets the strongest decide.
//! * [`denoise`] - the denoise task family's formal verifier, grading an
//!   experience against the reference passage it never showed the solver.
//! * [`overlap`] - texts that print one passage form a group, so a held-out
//!   split can keep a group whole.
//!
//! Every verdict names its producer, and those names keep the `splinter-lab/`
//! prefix this crate was first published under: they are recorded in each
//! annotation, and the decision rule and the views read them back.

#![warn(missing_docs)]

pub mod denoise;
pub mod frontier;
pub mod gate;
pub mod metric_gate;
pub mod overlap;
pub mod paired;
pub mod predictive_gate;
pub mod significance;
pub mod speech_reward;
pub mod timeline_metrics;
pub mod verifiers;
