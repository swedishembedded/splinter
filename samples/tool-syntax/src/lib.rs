// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The tool-syntax sample's harness: what it needs to measure honestly.
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
//! * [`ServedModel`] - which weights an arm actually measured. Two serving
//!   paths reach the same checkpoint and only one receives adapters; getting
//!   this wrong produces a convincing null result.
//! * [`ArmScore`] / [`Outcome`] - what an arm measured and what it refuses to
//!   claim, including the rule that an infrastructure fault is excluded from
//!   both the numerator and the denominator.
//! * [`Verdict`] / [`PredicateSet`] - the only source of a "solved". There is
//!   no constructor a sample could use to mark its own work correct.
//! * [`Family`] - a task contract that does not load unless every predicate it
//!   scores says where it came from.
//! * [`Recorder`] - the exact request the agent sent, captured at the wire.
//!   An agent's stored history holds neither the system prompt nor the tool
//!   schemas, so it is not enough to train on.
//! * [`records_from_performance`] - training data from a VERIFIED performance
//!   only, supervising the assistant's decision and nothing else.

#![warn(missing_docs)]

mod dataset;
mod episode;
mod family;
mod model_id;
mod perform;
mod recorder;
mod score;
mod verdict;

pub use dataset::{
    records_from_performance, to_jsonl, Excluded, Provenance, Record, RecordMetadata,
};
pub use episode::{baseline_effective, run_verifier, run_witness, Episode, EpisodeError};
pub use family::{Family, FamilyError};
pub use model_id::ServedModel;
pub use perform::{perform, perform_all, Action, PerformError, Performed};
pub use recorder::{capture_path, upstream_of, Recorder};
pub use score::{ArmScore, Outcome};
pub use verdict::{PredicateSet, Unevaluated, Verdict};
