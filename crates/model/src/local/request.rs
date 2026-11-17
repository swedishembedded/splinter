// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! How a local model samples each reply, as a request to its server carries it.

/// How a local model samples each reply: the generation cap for a request
/// that names none, the temperature and the top-k cut.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Sampling {
    /// Generation cap when the request names none.
    pub max_new_tokens: u32,
    /// The softmax temperature.
    pub temperature: f32,
    /// Only the `top_k` most likely tokens are sampled from.
    pub top_k: u32,
    /// Nucleus sampling: only the smallest set of tokens whose probability
    /// reaches `top_p` is sampled from. `None` leaves brain's default (all).
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Whether the model may reason before it answers. Off, a model whose
    /// template opens a reasoning block is asked with the block closed, so
    /// the reply is the answer and comes at once; on, it reasons first and
    /// the reply cap leaves room for that.
    #[serde(default)]
    pub thinking: bool,
    /// Seed of the decoding's draws. Set, the same prompt on the same
    /// weights decodes to the same reply, so a rollout can be repeated;
    /// `None` leaves the server to draw a fresh seed for each sampled
    /// request; a greedy request (temperature zero) is never seeded.
    #[serde(default)]
    pub seed: Option<u64>,
}

impl Sampling {
    /// The per-request sampling controls this stands for, as sven sends them
    /// with every model request of a run (`RunOptions::sampling`), whatever
    /// provider serves the model: the temperature, the nucleus mass, the
    /// top-k cut (`0` is no cut) and the seed. The generation cap and
    /// whether the model reasons are the provider's configuration, not a
    /// request's sampling.
    #[must_use]
    pub fn request_params(&self) -> sven_sdk::model::SamplingParams {
        sven_sdk::model::SamplingParams {
            temperature: Some(f64::from(self.temperature)),
            top_p: self.top_p.map(f64::from),
            top_k: Some(self.top_k),
            seed: self.seed,
        }
    }
}
