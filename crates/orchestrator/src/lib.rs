// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that acquire a capability
// from a document or a tool and prove it with evidence. If your team needs
// expertise in agent infrastructure or small-model training loops, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What a learning run is carried out with: the configuration, the services a
//! process shares, the context one command works in, the models that play
//! each role, the run record every command that writes state keeps, and the
//! stores a pipeline decides through.
//!
//! * [`config`] - the settings, and the only reader of the environment.
//! * [`context`] - a [`Runtime`] shared by a process and a [`Context`] for
//!   one command.
//! * [`roles`] and [`model_ref`] - who plays each role, and what a model
//!   reference resolves to under the configuration.
//! * [`runs`] - the recorded run and its cross-process cancel.
//! * [`pipeline`] - stages run in order over one state, each recorded.
//! * [`releases`] and [`answers`] - the release store with its aliases, and
//!   the answers `ask` gave.
//! * [`ids`] - an id, or a unique prefix of one, resolved against what a
//!   store holds.
//! * [`error`] - the error every command reports.
//!
//! The stages and pipelines that compose these belong to
//! `splinter-pipelines`.

#![warn(missing_docs)]

pub mod answers;
pub mod config;
pub mod context;
pub mod error;
pub mod ids;
pub mod model_ref;
pub mod pipeline;
pub mod releases;
pub mod roles;
pub mod runs;

pub use config::Config;
pub use context::{Context, Runtime};
pub use error::OrchestratorError;
