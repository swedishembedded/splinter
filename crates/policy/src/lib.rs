// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The model Splinter trains, and the only crate that touches brain.
//!
//! * [`local`] - brain's Qwen3 stack in-process behind sven's
//!   `ModelProvider` seam: load, stream a generation, stop it cleanly.
//! * [`selection`] - which model a stage runs on (local weights, or a model
//!   reached over an API), and the identity every record gives it.
//! * [`train`] - one LoRA fine-tune scored on held-out records.
//! * [`complete`] - one prompt in, one reply's text out.
//!
//! Every other crate reaches the model through these types, so the day
//! brain's public SDK covers chat inference and fine-tuning, this crate is
//! the one place that changes.

pub mod complete;
pub mod local;
pub mod selection;
pub mod train;

pub use complete::complete_text;
pub use local::{panic_message, resolve_base, LocalQwen, LocalWeights};
pub use selection::{local_model_name, ModelSelection, RemoteModel};
