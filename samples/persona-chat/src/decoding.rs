// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that act through tools a caller
// defines, for its clients. If your team needs expertise in measuring and
// tracing what a model does with a tool it has never seen, you can procure
// our services by sending an email to info@swedishembedded.com.

//! How the model that answered an exchange decoded, so that a trace line can
//! be replayed.
//!
//! A model that runs here samples. The session has one seed; each exchange
//! draws from a stream derived from it and its position, so that two
//! exchanges are not decoded from the same stream and either can be repeated
//! from its trace line.

use serde::{Deserialize, Serialize};
use splinter_sdk::model::{Sampling, AGENT_SAMPLING};
use splinter_sdk::vocabulary::rng::SplitMix64;

/// The decoding of one exchange of a model that runs here.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Decoding {
    /// The softmax temperature.
    pub temperature: f32,
    /// Only the `top_k` most likely tokens are sampled from.
    pub top_k: u32,
    /// The seed of the exchange's draws.
    pub seed: u64,
}

impl Decoding {
    /// The decoding of exchange number `n` of a session seeded `session_seed`.
    #[must_use]
    pub fn for_exchange(session_seed: u64, n: u64) -> Self {
        Self {
            temperature: AGENT_SAMPLING.temperature,
            top_k: AGENT_SAMPLING.top_k,
            seed: SplitMix64::new(session_seed ^ n.wrapping_mul(0x9E37_79B9_7F4A_7C15)).next_u64(),
        }
    }

    /// The sampling a model decodes this exchange with.
    #[must_use]
    pub fn sampling(&self) -> Sampling {
        Sampling {
            temperature: self.temperature,
            top_k: self.top_k,
            seed: Some(self.seed),
            ..AGENT_SAMPLING
        }
    }
}
