// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training of speech synthesizers from feedback
// on how they sound, for its clients. If your team needs expertise in
// reinforcement learning for voice models, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The synthesizer as a policy that learns from scores of its own takes.
//!
//! This is brain's trainable synthesizer behind splinter's own error type.
//! What a take is worth is not decided here: a pipeline scores takes (a
//! listener's preference, a judge, the words a recogniser still hears) and
//! hands back each with its advantage in its group.

use std::path::{Path, PathBuf};

use brain::{PolicyOptions, PolicyUpdate, TtsOptions, TtsPipeline, TtsPolicy};

use super::Clip;
use crate::error::PolicyError;

/// One sampled take.
pub struct SpeechTake {
    inner: brain::Take,
}

impl SpeechTake {
    /// The take as audio.
    #[must_use]
    pub fn audio(&self) -> &Clip {
        self.inner.audio()
    }

    /// Codec frames in the take (12.5 to the second).
    #[must_use]
    pub fn frames(&self) -> usize {
        self.inner.frames()
    }
}

/// What one update saw, before its step.
#[derive(Clone, Copy, Debug)]
pub struct UpdateReport {
    /// The objective.
    pub loss: f32,
    /// Mean log-probability of the sampled tokens.
    pub mean_logprob: f32,
    /// Mean KL estimate to the original synthesizer.
    pub mean_kl: f32,
    /// Tokens the step was over.
    pub tokens: usize,
}

/// How an update is taken.
#[derive(Clone, Copy, Debug)]
pub struct UpdateRule {
    /// Learning rate.
    pub learning_rate: f32,
    /// Weight of the KL term that holds the voice near the original.
    pub kl_weight: f32,
}

impl Default for UpdateRule {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            kl_weight: 0.02,
        }
    }
}

/// A synthesizer that can be trained from scores of its takes.
pub struct SpeechPolicy {
    inner: TtsPolicy,
    voice: Option<PathBuf>,
}

fn failed(reason: impl std::fmt::Display) -> PolicyError {
    PolicyError::Synthesis {
        reason: reason.to_string(),
    }
}

impl SpeechPolicy {
    /// The synthesis model `model` names, made trainable. Takes are spoken in
    /// the speaker of the recording `voice`, when given, so seeds vary
    /// delivery and not who speaks.
    pub fn load(model: &str, voice: Option<&Path>) -> Result<Self, PolicyError> {
        let pipeline = TtsPipeline::builder(model)
            .device(crate::device::selected().spec())
            .load()
            .map_err(|e| PolicyError::Load {
                path: model.into(),
                reason: e.to_string(),
            })?;
        let inner = pipeline.policy(PolicyOptions::default()).map_err(failed)?;
        Ok(Self {
            inner,
            voice: voice.map(Path::to_path_buf),
        })
    }

    /// One take of `text`, drawn with `seed` under the current policy.
    pub fn take(&mut self, text: &str, seed: u64) -> Result<SpeechTake, PolicyError> {
        let mut options = TtsOptions::new().seed(seed).max_frames(300);
        if let Some(voice) = &self.voice {
            options = options.voice(voice, "");
        }
        let inner = self.inner.rollout(text, &options).map_err(failed)?;
        Ok(SpeechTake { inner })
    }

    /// One step over a group of takes of the same sentence, each with the
    /// advantage it earned.
    pub fn update(
        &mut self,
        group: &[(&SpeechTake, f32)],
        rule: &UpdateRule,
    ) -> Result<UpdateReport, PolicyError> {
        let group: Vec<(&brain::Take, f32)> = group.iter().map(|(t, a)| (&t.inner, *a)).collect();
        let stats = self
            .inner
            .update(
                &group,
                &PolicyUpdate {
                    lr: rule.learning_rate,
                    kl_beta: rule.kl_weight,
                    clip: 1.0,
                },
            )
            .map_err(failed)?;
        Ok(UpdateReport {
            loss: stats.loss,
            mean_logprob: stats.mean_logprob,
            mean_kl: stats.mean_kl,
            tokens: stats.tokens,
        })
    }

    /// One direct-preference step on two takes of the same sentence. Returns the
    /// probability the policy gave the preference before the step.
    pub fn prefer(
        &mut self,
        chosen: &SpeechTake,
        rejected: &SpeechTake,
        beta: f32,
        rule: &UpdateRule,
    ) -> Result<f32, PolicyError> {
        self.inner
            .update_preference(
                &chosen.inner,
                &rejected.inner,
                beta,
                &PolicyUpdate {
                    lr: rule.learning_rate,
                    kl_beta: 0.0,
                    clip: 1.0,
                },
            )
            .map_err(failed)
    }

    /// Write the trained adapter to `path`.
    pub fn save(&self, path: &Path) -> Result<(), PolicyError> {
        self.inner.save(path).map_err(failed)
    }
}
