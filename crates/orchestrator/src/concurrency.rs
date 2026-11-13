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
    /// How many requests to `model` are asked at once.
    ///
    /// A model reached over the network is asked the configured width
    /// ([`Config::remote_concurrency`](crate::config::Config)): set it to the
    /// server's `--max-batch`. A local reference is requested from a brain
    /// server with decode slots of its own, so brain's continuous batching
    /// decodes the requests together: as many at once as the configured
    /// endpoints are given
    /// ([`splinter_model::BrainEndpoints::attempt_concurrency`]), else the
    /// width of the server this runtime starts
    /// ([`splinter_model::server::DEFAULT_SERVE_SLOTS`]). A caller that
    /// launches a server itself knows its slots and sets them on the
    /// [`Model`](splinter_agent::solve::Model) it asks.
    #[must_use]
    pub fn concurrency_for(&self, model: &ModelRef) -> usize {
        if model.is_remote() {
            return self.config().remote_concurrency.max(1);
        }
        self.config().brain.as_ref().map_or_else(
            || {
                self.config()
                    .brain_max_batch
                    .unwrap_or(splinter_model::server::DEFAULT_SERVE_SLOTS) as usize
            },
            splinter_model::BrainEndpoints::attempt_concurrency,
        )
    }
}
