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
//! * [`canary`] - tasks that cannot be solved honestly: a pass on one refuses
//!   the candidate.
//! * [`claim_gate`] - the release gate for a model updated from facts a
//!   person taught it: a declared rule on counts, no significance test.
//! * [`metric_gate`] - the release decision over continuous metrics and
//!   their intervals, for a model judged by a metric rather than item by
//!   item: requirements as data, unmeasured as failure.
//! * [`predictive_gate`] - the release gate for a predictive model over
//!   paired held-out units: performance, calibration, retention on
//!   subgroups, serving correctness and data-policy compliance, built on
//!   [`metric_gate`] requirements.
//! * [`speech_judge`] - a judge of how natural a take sounds, learned from a
//!   listener's pairwise choices and tested on choices it never saw.
//! * [`speech_reward`] - the reward a spoken take earns (naturalness against
//!   clarity) and how takes of one sentence are compared.
//! * [`continual`] - continual-learning metrics over a results matrix with
//!   unmeasured cells absent: average final performance, backward transfer,
//!   forgetting, forward transfer against the untouched base, learning gain.
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
//! * [`exposure`] - the audit that fails an experience whose trajectory shows
//!   the solver material only the grader may see (hidden paths, a shingle of
//!   hidden content, the state root).
//! * [`overlap`] - texts that print one passage form a group, so a held-out
//!   split can keep a group whole.
//!
//! Every verdict names its producer, and those names keep the `splinter-lab/`
//! prefix this crate was first published under: they are recorded in each
//! annotation, and the decision rule and the views read them back.

#![warn(missing_docs)]

pub mod canary;
pub mod claim_gate;
pub mod competence;
pub mod continual;
pub mod denoise;
pub mod exposure;
mod exposure_paths;
pub mod frontier;
pub mod gate;
pub mod metric_gate;
pub mod overlap;
pub mod paired;
pub mod pilot;
pub mod power;
pub mod predictive_gate;
pub mod significance;
pub mod speech_features;
pub mod speech_judge;
pub mod speech_reward;
pub mod teaching_check;
pub mod timeline_metrics;
pub mod verifiers;
