// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems that acquire a capability
// from a document or a tool and prove it with evidence. If your team needs
// expertise in agent infrastructure or small-model training loops, you can
// procure our services by sending an email to info@swedishembedded.com.

//! How wide a model is asked.
//!
//! [`fan_out`] and [`fan_out_each`] (re-exported from the vocabulary) run work items
//! concurrently up to a limit; [`Runtime::concurrency_for`] decides the
//! limit for a model.

use crate::context::Runtime;
use splinter_core::model_ref::ModelRef;

pub use splinter_core::concurrency::{fan_out, fan_out_blocking, fan_out_each};

impl Runtime {
    /// How many requests to `model` may be in flight at once: the
    /// configured width ([`remote_concurrency`](crate::config::Config::remote_concurrency)) for one reached
    /// over the network, one for a model run in this process.
    ///
    /// A `brain serve` named as `remote:brain/<model>` is reached over the
    /// network, so its width is the configured one: set it to the server's
    /// `--max-batch`. An in-process model is one resident engine behind
    /// a lock, and a new request on its handle cancels the one still
    /// decoding: it has no decode slots to fill, and asking it several
    /// things at once would make the requests cancel each other. A caller
    /// that launches a server itself knows its slots and sets them on the
    /// [`Model`](splinter_agent::solve::Model) it asks.
    ///
    /// A local reference served by brain endpoints (`--brain-endpoint`) is a
    /// service with decode slots of its own: it is asked as many attempts at
    /// once as the endpoints are given
    /// ([`splinter_model::BrainEndpoints::attempt_concurrency`]), so brain's
    /// continuous batching decodes them together.
    #[must_use]
    pub fn concurrency_for(&self, model: &ModelRef) -> usize {
        if let (Some(brain), false) = (&self.config().brain, model.is_remote()) {
            return brain.attempt_concurrency();
        }
        if model.is_remote() {
            self.config().remote_concurrency.max(1)
        } else {
            1
        }
    }
}
