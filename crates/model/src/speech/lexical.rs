// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A synthesizer that says words as it was taught to.

use splinter_core::speech::SpeakerProfile;
use splinter_core::speech_lesson::Lexicon;

use super::{Clip, Synthesizer};
use crate::error::PolicyError;

/// `inner`, speaking each text with the words a [`Lexicon`] knows replaced by
/// how they are to be said. What is written, and what the persona answered,
/// stays as it was: only what is handed to the voice changes.
pub struct Lexical<S> {
    inner: S,
    lexicon: Lexicon,
}

impl<S> Lexical<S> {
    /// `inner`, taught `lexicon`.
    pub fn new(inner: S, lexicon: Lexicon) -> Self {
        Self { inner, lexicon }
    }
}

impl<S: Synthesizer> Synthesizer for Lexical<S> {
    fn speak(&self, text: &str, speaker: &SpeakerProfile) -> Result<Clip, PolicyError> {
        self.inner.speak(&self.lexicon.apply(text), speaker)
    }
}
