// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Recognition and synthesis on brain's pipelines.

use brain::{ResidentTts, TranscribePipeline, TtsOptions, TtsPipeline};
use splinter_core::speech::SpeakerProfile;

use super::{Clip, Recognizer, Synthesizer, Transcription};
use crate::error::PolicyError;

/// The recognizer used where none is named: a small streaming model, which
/// brain also uses to judge its own synthesis.
pub const DEFAULT_RECOGNIZER: &str = "nvidia/nemotron-3.5-asr-streaming-0.6b";

/// The synthesizer used where none is named: the smallest Qwen3-TTS.
pub const DEFAULT_SYNTHESIZER: &str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base";

/// A recognition model loaded on brain.
pub struct BrainRecognizer {
    pipeline: TranscribePipeline,
}

impl BrainRecognizer {
    /// The recognition model `model` names, in brain's model store.
    pub fn load(model: &str) -> Result<Self, PolicyError> {
        let pipeline =
            TranscribePipeline::from_pretrained(model).map_err(|e| PolicyError::Load {
                path: model.into(),
                reason: e.to_string(),
            })?;
        Ok(Self { pipeline })
    }
}

impl BrainRecognizer {
    /// The audio encoder's features of `clip`, for a language model that
    /// listens rather than reads a transcript. Qwen3-ASR only.
    pub fn features(&self, clip: &Clip) -> Result<brain::AudioFeatures, PolicyError> {
        self.pipeline
            .features(clip)
            .map_err(|e| PolicyError::Transcription {
                reason: e.to_string(),
            })
    }
}

impl BrainRecognizer {
    /// [`Self::features`] for several clips with the audio encoder built once.
    pub fn features_many(&self, clips: &[&Clip]) -> Result<Vec<brain::AudioFeatures>, PolicyError> {
        self.pipeline
            .features_many(clips)
            .map_err(|e| PolicyError::Transcription {
                reason: e.to_string(),
            })
    }
}

impl Recognizer for BrainRecognizer {
    fn transcribe(&self, clip: &Clip) -> Result<Transcription, PolicyError> {
        let heard =
            self.pipeline
                .transcribe_audio(clip)
                .map_err(|e| PolicyError::Transcription {
                    reason: e.to_string(),
                })?;
        Ok(Transcription {
            text: heard.text.trim().to_string(),
            truncated: heard.truncated.is_some(),
        })
    }
}

/// A synthesis model loaded on brain and kept resident: its checkpoints are
/// read once, not on every sentence.
pub struct BrainSynthesizer {
    engine: ResidentTts,
}

impl BrainSynthesizer {
    /// The synthesis model `model` names, in brain's model store.
    pub fn load(model: &str) -> Result<Self, PolicyError> {
        let load_error = |e: brain::Error| PolicyError::Load {
            path: model.into(),
            reason: e.to_string(),
        };
        let engine = TtsPipeline::from_pretrained(model)
            .and_then(|pipeline| pipeline.resident())
            .map_err(load_error)?;
        Ok(Self { engine })
    }
}

impl Synthesizer for BrainSynthesizer {
    fn speak(&self, text: &str, speaker: &SpeakerProfile) -> Result<Clip, PolicyError> {
        self.engine
            .speak_with(text, TtsOptions::new().seed(speaker.seed()))
            .map_err(|e| PolicyError::Synthesis {
                reason: e.to_string(),
            })
    }
}
