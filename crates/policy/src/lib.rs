// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The model Splinter trains, and the only crate that touches brain.
//!
//! * [`local`] - brain's chat pipeline in-process behind sven's
//!   `ModelProvider` seam: load, stream a generation, stop it cleanly.
//! * [`selection`] - which model a stage runs on (local weights, or a model
//!   reached over an API), the identity every record gives it, and the
//!   model loaded for a command.
//! * [`train`] - one LoRA fine-tune scored on held-out records.
//! * [`stats`] - brain's paired sign test, which a release decision rests
//!   on.
//! * [`error`] - why a model could not be loaded, reached or trained.
//!
//! Every other crate reaches the model through these types, and these types
//! reach brain only through its public SDK, the `brain` crate.

#![warn(missing_docs)]

pub mod error;
pub mod local;
pub mod selection;
pub mod stats;
pub mod train;

pub use error::PolicyError;
pub use local::LocalWeights;
pub use selection::{LoadedModel, ModelSelection, RemoteModel};
