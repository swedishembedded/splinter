// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A language model that listens: speech features projected into its input.
//!
//! brain trains the projector; this module is the seam Splinter reaches it
//! through. The model, and any persona adapter on it, is held at a zero
//! learning rate, so what it does with text is exactly what it did before.

use std::path::{Path, PathBuf};

use brain::{IngressHyper, SpeechExample, SpeechIngress, SpeechIngressOptions};

use crate::error::PolicyError;

pub use brain::AudioFeatures;

/// Width of one row of Qwen3-ASR's projected audio embeddings.
pub const FEATURE_WIDTH: usize = 2048;

/// What to load.
#[derive(Clone, Debug)]
pub struct IngressOptions {
    /// The language model's checkpoint directory.
    pub base: PathBuf,
    /// A persona adapter that stays on, frozen.
    pub adapter: Option<PathBuf>,
    /// Rows of audio in every example (thirteen per second of the window).
    pub rows: usize,
    /// Tokens in one training row.
    pub block: u32,
    /// Seed of the projector's initial weights.
    pub seed: u64,
}

/// One example, ready for the model.
pub type IngressExample = SpeechExample;

/// The settings of one step.
pub type StepSettings = IngressHyper;

/// A language model with a trainable projector in front of it.
pub struct Ingress {
    inner: SpeechIngress,
}

fn failed(reason: impl std::fmt::Display) -> PolicyError {
    PolicyError::Transcription {
        reason: reason.to_string(),
    }
}

impl Ingress {
    /// Load the model (bf16) and a fresh projector.
    pub fn load(opts: &IngressOptions) -> Result<Self, PolicyError> {
        let inner = SpeechIngress::new(&SpeechIngressOptions {
            base: opts.base.clone(),
            adapter: opts.adapter.clone(),
            input_dim: FEATURE_WIDTH,
            rows: opts.rows,
            block: opts.block,
            seed: opts.seed,
            bf16: true,
        })
        .map_err(failed)?;
        Ok(Self { inner })
    }

    /// An example: `features` heard under the system turn `system` and the
    /// user's `instruction` (empty for none), to be answered with `reply`.
    pub fn example(
        &mut self,
        features: &AudioFeatures,
        system: &str,
        instruction: &str,
        reply: &str,
    ) -> Result<IngressExample, PolicyError> {
        if features.embed_dim != FEATURE_WIDTH {
            return Err(failed(format!(
                "feature rows are {} wide, the projector reads {FEATURE_WIDTH}",
                features.embed_dim
            )));
        }
        self.inner
            .example(
                &features.embeds,
                &SpeechIngress::prefix(system),
                &SpeechIngress::suffix(instruction),
                reply,
            )
            .map_err(failed)
    }

    /// Mean cross-entropy of the reply of `example` given its audio.
    pub fn loss(&mut self, example: &IngressExample) -> f32 {
        self.inner.loss(example)
    }

    /// One optimiser step (1-based `t`) on the mean gradient of `batch`;
    /// returns the mean loss.
    pub fn step(&mut self, batch: &[&IngressExample], t: u32, settings: &StepSettings) -> f32 {
        self.inner.step(batch, t, settings)
    }

    /// Write the projector to `path`.
    pub fn save_projector(&self, path: &Path) -> Result<(), PolicyError> {
        self.inner.save_projector(path).map_err(failed)
    }

    /// Replace the projector with the one in `path`.
    pub fn load_projector(&mut self, path: &Path) -> Result<(), PolicyError> {
        self.inner.load_projector(path).map_err(failed)
    }
}
