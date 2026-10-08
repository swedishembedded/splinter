// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Speech doubles for specs with no weights.
//!
//! [`ScriptedSynthesizer`] writes a sentence into the samples of its clip, one
//! byte of the text per sample, so [`ScriptedRecognizer`] can read it back and
//! a degradation applied to what it heard stands for a recognition error. A
//! spec can then say exactly what was lost.

use splinter_core::speech::SpeakerProfile;

use super::{Clip, Recognizer, Synthesizer, Transcription};
use crate::error::PolicyError;

/// The rate the doubles' clips claim.
const RATE: u32 = 16_000;

/// A synthesizer whose clips hold the text they were asked to speak.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScriptedSynthesizer;

impl Synthesizer for ScriptedSynthesizer {
    fn speak(&self, text: &str, _speaker: &SpeakerProfile) -> Result<Clip, PolicyError> {
        Ok(Clip::new(
            text.bytes().map(|b| f32::from(b) / 255.0).collect(),
            RATE,
        ))
    }
}

/// A recognizer that reads the text back out of a [`ScriptedSynthesizer`]
/// clip and then degrades it with a function: identity for perfect hearing.
pub struct ScriptedRecognizer {
    degrade: Box<dyn Fn(String) -> String + Send + Sync>,
}

impl ScriptedRecognizer {
    /// Hears exactly what was spoken.
    #[must_use]
    pub fn perfect() -> Self {
        Self::degrading(|text| text)
    }

    /// Hears what `degrade` makes of what was spoken.
    pub fn degrading(degrade: impl Fn(String) -> String + Send + Sync + 'static) -> Self {
        Self {
            degrade: Box::new(degrade),
        }
    }
}

impl Recognizer for ScriptedRecognizer {
    fn transcribe(&self, clip: &Clip) -> Result<Transcription, PolicyError> {
        let bytes: Vec<u8> = clip
            .samples()
            .iter()
            .map(|s| (s * 255.0).round().clamp(0.0, 255.0) as u8)
            .collect();
        let spoken = String::from_utf8_lossy(&bytes).into_owned();
        Ok(Transcription {
            text: (self.degrade)(spoken),
            truncated: false,
        })
    }
}
