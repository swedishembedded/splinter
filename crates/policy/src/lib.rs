// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The model Splinter trains, and the only crate that touches brain.
//!
//! * [`local`] - brain's chat pipeline in-process behind sven's
//!   `ModelProvider` seam: load, stream a generation, stop it cleanly.
//! * [`residency`] - the one owner of resident bases: one loaded copy per
//!   base checkpoint, shared by every model on it, each model's adapter
//!   attached before its generation, and every base released when other
//!   work needs the device.
//! * [`selection`] - which model a stage runs on (local weights, or a model
//!   reached over an API), the identity every record gives it, and the
//!   model loaded for a command.
//! * [`train`] - one LoRA fine-tune, supervised (chat) or by DPO
//!   (preference pairs), scored on held-out records.
//! * [`stats`] - brain's paired sign test, which a release decision rests
//!   on.
//! * [`error`] - why a model could not be loaded, reached or trained.
//!
//! Every other crate reaches the model through these types, and these types
//! reach brain only through its public SDK, the `brain` crate.

#![warn(missing_docs)]

pub mod error;
pub mod local;
pub mod residency;
pub mod selection;
pub mod stats;
pub mod train;

pub use error::PolicyError;
pub use local::{LocalWeights, Sampling, AGENT_SAMPLING};
pub use residency::Residency;
pub use selection::{LoadedModel, ModelSelection, RemoteModel};
