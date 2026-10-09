// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The model Splinter trains, and the only crate that touches brain.
//!
//! * [`answer`] - one local model, with or without an adapter, answering one
//!   question at a time, greedily: what an evaluation asks.
//! * [`exam`] - a frozen exam run against one local model, resumable, each
//!   answer graded by code and recorded as one line.
//! * [`report`] - two arms of an exam paired question by question, with the
//!   paired sign test.
//! * [`rl`] - brain's reinforcement-learning surface: an environment, a
//!   verifier, and the chat tokenizer a prompt is read through.
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
//! * [`capabilities`] - which dataset formats brain trains, and the check of a
//!   dataset file against brain's own parser before it is stored.
//! * [`stats`] - brain's paired sign test, which a release decision rests
//!   on.
//! * [`speech`] - speech in and out around a text model: recognition,
//!   synthesis, one turn of the cascade, and what a round trip loses.
//! * [`error`] - why a model could not be loaded, reached or trained.
//!
//! Every other crate reaches the model through these types, and these types
//! reach brain only through its public SDK, the `brain` crate.

#![warn(missing_docs)]

pub mod answer;
pub mod capabilities;
pub mod device;
pub mod embed;
pub mod error;
pub mod exam;
pub mod local;
pub mod report;
pub mod residency;
pub mod rl;
pub mod selection;
pub mod speech;
pub mod stats;
pub mod timeline;
pub mod tokens;
pub mod train;

pub use capabilities::{BrainDatasetCheck, TrainingCapabilities};
pub use error::PolicyError;
pub use local::{LocalWeights, Sampling, AGENT_SAMPLING};
pub use residency::Residency;
pub use selection::{LoadedModel, ModelSelection, RemoteModel};
pub use tokens::TokenCounter;
