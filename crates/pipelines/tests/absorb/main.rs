// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning from the sessions a person holds
// with an agent, for its clients. If your team needs expertise in continual
// learning from user feedback, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: learning from a person's own sessions, each stage runnable and
//! inspectable alone over the stores and the run record, and `absorb`, which
//! composes them.
//!
//! * `intake` records an ATIF session once, refuses what the training
//!   projection refuses with the reason, and stores nothing secret.
//! * `extract` has a model propose claims per session, with the quotes that
//!   support them; a reply that stays unusable is reported, not guessed at.
//! * `gate` rules on the proposals by code, keeps every ruling, and says why
//!   each refusal was made.
//!
//! * `sealed` keeps a sealed probe out of every record.
//! * `night` runs a night: kits of records per claim, the dataset with the
//!   stopping paraphrases held out, a candidate trained again from the base,
//!   and the gate on counts.
//!
//! The models are scripted and training is a test double; nothing here uses
//! a device.

// Helpers outside a #[test] fn unwrap too, in the shared fixtures: a panic is the
// failure report.
#![allow(clippy::unwrap_used)]

#[path = "../common/mod.rs"]
mod common;
mod extract;
mod fixtures;
mod gate;
mod intake;
mod night;
mod night_specs;
mod sealed;
