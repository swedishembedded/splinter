// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements end-to-end learning systems that take a
// model from random weights to a working tool-using agent, for its clients.
// If your team needs expertise in training small agents from scratch, you
// can procure our services by sending an email to info@swedishembedded.com.

//! From random weights to a small agent that solves workspace tool-use
//! tasks, in one resumable command.
//!
//! What is specific to this sample lives here: the configuration and its
//! profiles ([`config`]), the synthetic world ([`world`]), the training
//! task families and their verifier ([`tasks`]), the deterministic expert
//! ([`expert`]), the model's chat template ([`template`]), the stage list
//! ([`stages`], [`collect`], [`experience`], [`exam`]) with what their keys depend on ([`recipe`]), the
//! terminal's panels ([`panel`]) and the report ([`report`]). Everything generic
//! is Splinter's: the stage ledger (`store::stage_ledger`), sven episodes in
//! a workspace and the policy endpoint (`agent`), training records from
//! recorded calls (`data::wire`), brain over its API (`model::scratch`) and
//! bench's examination (`bench::examine`).

pub mod admission;
pub mod collect;
pub mod config;
pub mod datacheck;
pub mod exam;
pub mod experience;
pub mod expert;
pub mod formats;
pub mod pack;
pub mod panel;
pub mod recipe;
pub mod report;
pub mod retrieval;
pub mod sequence;
pub mod skills;
pub mod stages;
pub mod tasks;
pub mod template;
pub mod world;
