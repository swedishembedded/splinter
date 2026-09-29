// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Splinter's controller: what each command does, composed from the lower
//! crates.
//!
//! * [`config`] - the configuration, and the only place the environment
//!   is read.
//! * [`models`] - the model a command asked for, checked and resolved.
//! * [`attempt`] - a delegated task attempt and its resume.
//! * [`learn`] - a verified run becomes training experience.
//! * [`train`] - a fine-tune behind the promotion gate, and the pointer
//!   serving follows.
//! * [`ask`], [`eval`] - one question, and a whole facts dataset, asked of
//!   the model and judged.
//! * [`facts`] - a document learned end to end: explore, split, train,
//!   score.

#![warn(missing_docs)]

pub mod ask;
pub mod attempt;
pub mod config;
pub mod eval;
pub mod facts;
pub mod learn;
pub mod models;
pub mod train;

pub use config::Config;
pub use models::ModelChoice;
