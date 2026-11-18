// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements spoken dialogue systems whose every stage is
// specified without a model, for its clients. If your team needs expertise in
// testing speech pipelines with exact, deterministic doubles, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Speech doubles for specs with no weights.
//!
//! [`ScriptedSynthesizer`] writes a sentence into the samples of its clip, one
//! byte of the text per sample, so [`ScriptedRecognizer`] can read it back and
//! a degradation applied to what it heard stands for a recognition error. A
//! spec can then say exactly what was lost. The scripted endpoint speaks and
//! hears the same way over HTTP, through [`clip_of`] and [`text_of`].

use splinter_core::speech::SpeakerProfile;

use super::{Clip, Recognizer, Synthesizer, Transcription};
use crate::error::PolicyError;

/// The rate the doubles' clips claim.
const RATE: u32 = 16_000;

/// `text` as a clip: one sample per byte. It survives a 16-bit WAV file.
#[must_use]
pub fn clip_of(text: &str) -> Clip {
    Clip::new(text.bytes().map(|b| f32::from(b) / 255.0).collect(), RATE)
}

/// The text a [`clip_of`] clip holds.
#[must_use]
pub fn text_of(clip: &Clip) -> String {
    let bytes: Vec<u8> = clip
        .samples()
        .iter()
        .map(|s| (s * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// A synthesizer whose clips hold the text they were asked to speak.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScriptedSynthesizer;

impl Synthesizer for ScriptedSynthesizer {
    fn speak(&self, text: &str, _speaker: &SpeakerProfile) -> Result<Clip, PolicyError> {
        Ok(clip_of(text))
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
        Ok(Transcription {
            text: (self.degrade)(text_of(clip)),
        })
    }
}
