// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that run model learning as one
// continuous, resumable campaign with dynamic sampling, replay and a
// validation gate with rollback, for its clients. If your team needs expertise
// in on-policy training loops for language-model agents you can procure our
// services by sending an email to info@swedishembedded.com.

//! The campaign controller: learning as a continuous, resumable campaign
//! instead of rounds glued together by scripts.
//!
//! A campaign is a sequence of arms. Each arm is one cycle of the loop:
//!
//! 1. **preflight** - the disks have the bytes and the files the arm needs
//!    ([`crate::preflight`]);
//! 2. **plan** - the exam that will judge the arm can detect the effect it
//!    targets, at the level this gate may spend, or the arm is refused before
//!    it costs anything ([`splinter_eval::campaign::plan`]);
//! 3. **calibrate** - every task not yet measured is attempted k = 8 to 16
//!    times by the current policy; the per-task pass rate is the controller's
//!    state, and tasks at or above 0.9 are dropped for good
//!    ([`splinter_eval::campaign::pass_rate`]);
//! 4. **collect** - a [`RolloutCollector`] rolls out groups of attempts on
//!    prompts drawn from the frontier; only groups with a reward spread enter
//!    the batch, over-provisioned and refilled until the batch is full
//!    ([`splinter_eval::campaign::sampling`]); groups that all failed go to
//!    the configured [`splinter_eval::campaign::sampling::AllFailStrategy`];
//! 5. **train** - the batch's records, beside a bounded replay of earlier
//!    arms' verified data, go to a [`Trainer`] (brain), which returns an
//!    adapter and the KL and clip fraction of its steps;
//! 6. **gate** - the signals of the step are read against their bounds, and a
//!    [`Validator`] runs the candidate and the champion on dev tasks the
//!    campaign never trains on; the candidate becomes the champion or is rolled
//!    back ([`splinter_eval::campaign::gate`]);
//! 7. **account** - the arm's hypothesis, configuration digest, GPU-hours,
//!    cost and verdict are appended to the ledger, the state is saved by
//!    compare-and-swap, superseded checkpoints are collected keeping their
//!    digests ([`gc`]), and the discovery rate per GPU-hour is read for a
//!    plateau ([`splinter_eval::campaign::plateau`]); a plateau moves the
//!    campaign to the next strategy of its ladder.
//!
//! Where rollouts come from is a trait. [`served`] holds the path over a
//! brain server (a served model sampled at temperature 1 against a verifier,
//! and brain's training job), with the family and settings of [`brain`];
//! [`endpoint`] holds the path over an agent
//! executor that runs tasks in sandboxes and returns, per model call, the
//! token ids and log-probabilities the policy sampled with.
//!
//! The state is a value in the store ([`splinter_store::campaign`]): a crash
//! leaves the last saved state, an unfinished arm is redone from its start
//! (or, when training had finished, from its gate), and a second controller
//! on the same campaign loses the compare-and-swap instead of corrupting it.

pub mod brain;
mod closing;
mod config;
mod controller;
pub mod endpoint;
mod environments;
pub mod factory;
pub mod gc;
mod records;
mod replay;
pub mod served;
mod stages;
mod state;
mod types;

pub use config::{Acceptance, AdvantageMode, CampaignConfig, LearningStrategy, ReplayConfig};
pub use controller::{ArmReport, CampaignController, CampaignReport, Parts, Stopwatch, WallClock};
pub use environments::EnvironmentLabels;
pub use records::{BatchRecords, RecordCounts};
pub use replay::ReplayBuffer;
pub use state::{CampaignState, EvidenceSpent, PendingCandidate};
pub use types::{
    AllFailHandler, CampaignError, CollectRequest, PolicyRef, Rollout, RolloutCollector,
    RolloutGroup, RolloutTrace, TraceSequence, TrainReport, TrainRequest, Trained, Trainer,
    Validator,
};
