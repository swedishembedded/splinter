// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that draw rollouts from a served
// language-model policy and train it on them, for its clients. If your team
// needs expertise in rollout collection and policy-gradient training against
// a model server you can procure our services by sending an email to
// info@swedishembedded.com.

//! Rollouts over a served policy: the seams (a collector, a trainer and a
//! validator, [`types`]), what asks a model server for them
//! ([`PolicyModels`], [`ServedCollector`], [`ServedValidator`],
//! [`ServedTrainer`]), the task family a verifier is plugged in through
//! ([`brain`]), the episode executor seam ([`endpoint`]) and the environment
//! factory as a source of tasks ([`factory`]).
//!
//! Samples and the episode executors are built on them. Which server holds
//! which policy is the host's business; nothing here starts one.

pub mod brain;
mod collector;
pub mod endpoint;
mod environments;
pub mod factory;
mod types;

pub use collector::{
    Answered, Attempt, CallFailure, PolicyModels, ServedCollector, ServedTrainer, ServedValidator,
};
pub use environments::EnvironmentLabels;
pub use types::{
    AllFailHandler, CollectRequest, PolicyRef, Rollout, RolloutCollector, RolloutGroup,
    RolloutTrace, ServedError, TraceSequence, TrainReport, TrainRequest, Trained, Trainer,
    Validator,
};
