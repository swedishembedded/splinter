// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Splinter's laboratory: what every learning experiment needs to measure
//! honestly.
//!
//! Each experiment is controlled: a frozen task catalog, an information
//! boundary the agent cannot read around, a verifier it cannot reach, and a
//! before/after table. This crate holds the parts every one of them needs, so
//! an experiment's own source is its design and nothing else.
//!
//! It links both halves of the loop - sven's SDK facade to run the agent,
//! brain to train and gate the weights.
//!
//! Swedish Embedded AB implements closed-loop learning systems - agents that
//! improve from their own verified experience rather than from hand-written
//! training data - for its clients. If your team needs expertise in agent
//! training loops, experiment design, or promotion gating for small models,
//! you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # What the parts are for
//!
//! * [`ServedModel`] - which weights an arm actually measured. Two serving
//!   paths reach the same checkpoint and only one receives adapters; getting
//!   this wrong produces a convincing null result.
//! * [`ArmScore`] / [`Outcome`] - what an arm measured and what it refuses to
//!   claim, including the rule that an infrastructure fault is excluded from
//!   both the numerator and the denominator.
//! * [`AdapterPath`] - proof that promoted adapters reach the model an arm is
//!   about to measure. Scoring requires one, so measuring the wrong weights is
//!   not a mistake to catch in review.
//! * [`Verdict`] / [`PredicateSet`] - the only source of a "solved". There is
//!   no constructor an experiment could use to mark its own work correct.
//! * [`Family`] - a task contract that does not load unless every predicate it
//!   scores says where it came from.
//! * [`Demonstrator`] - a scripted model at the wire, so a demonstration's
//!   observations are produced by the real tool executor rather than invented.
//! * [`Recorder`] - the exact request the agent sent, captured at the wire.
//!   An agent's stored history holds neither the system prompt nor the tool
//!   schemas, so it is not enough to train on.
//! * [`holdout`] - the rule that decides which records of a training set
//!   are held out for scoring.
//! * [`paired`] - two models graded on the same items, compared only where
//!   both have a verdict.
//! * [`record_from_episode`] - training data from a VERIFIED episode only,
//!   supervising assistant turns and nothing else, and refusing a transcript
//!   with a hole in it. [`WireMessage`] is one message of the
//!   `generic-messages-v2` format those records are written in.
//! * [`verifiers`] - verifiers by strength (executable, formal, consistency,
//!   judged), each grading an experience from its output and the task's
//!   privileged material alone, and the composite that annotates every
//!   verdict so the store's decision rule lets the strongest decide.
//! * [`denoise`] - the denoise task family's formal verifier, grading an
//!   experience against the reference passage it never showed the solver.
//!
//! # Making the expensive mistakes impossible rather than noticed
//!
//! Three of these types exist because of failures that were observed while
//! building this harness, and each shares a shape: the run completes, reports
//! a plausible number, and the number is wrong. A warning is a poor defence
//! against that - one of these failures HAD a warning available and it
//! scrolled past in a log - so where it was possible to make the mistake
//! unrepresentable instead, that is what these types do.

#![warn(missing_docs)]

mod dataset;
mod demonstrate;
pub mod denoise;
mod endpoint;
mod episode;
mod family;
pub mod holdout;
mod model_id;
pub mod paired;
mod perform;
mod recorder;
mod score;
mod verdict;
pub mod verifiers;

pub use dataset::{
    record_from_episode, record_from_requests, records_from_performance, to_jsonl, Excluded,
    Provenance, Record, RecordMetadata, WireFunction, WireMessage, WireToolCall,
};
pub use demonstrate::{Demonstrator, Step};
pub use endpoint::{AdapterPath, NoAdapterPath};
pub use episode::{baseline_effective, run_verifier, run_witness, Episode, EpisodeError};
pub use family::{Family, FamilyError};
pub use model_id::ServedModel;
pub use perform::{perform, perform_all, Action, PerformError, Performed};
pub use recorder::{capture_path, upstream_of, Recorder};
pub use score::{ArmScore, Outcome};
pub use verdict::{PredicateSet, Unevaluated, Verdict};
