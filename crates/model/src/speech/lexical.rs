// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A synthesizer that says words as it was taught to.

use splinter_core::speech::SpeakerProfile;
use splinter_core::speech_lesson::Lexicon;
use std::sync::PoisonError;

use super::desk::SharedLexicon;
use super::{Clip, Synthesizer};
use crate::error::PolicyError;

/// `inner`, speaking each text with the words a [`Lexicon`] knows replaced by
/// how they are to be said. What is written, and what the persona answered,
/// stays as it was: only what is handed to the voice changes.
pub struct Lexical<S> {
    inner: S,
    lexicon: SharedLexicon,
}

impl<S> Lexical<S> {
    /// `inner`, taught `lexicon`.
    pub fn new(inner: S, lexicon: Lexicon) -> Self {
        Self::shared(inner, std::sync::Arc::new(std::sync::RwLock::new(lexicon)))
    }

    /// `inner`, taught what a lexicon shared with a
    /// [`LessonDesk`](super::LessonDesk) comes to know, as it learns it.
    pub fn shared(inner: S, lexicon: SharedLexicon) -> Self {
        Self { inner, lexicon }
    }

    fn apply(&self, text: &str) -> String {
        self.lexicon
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .apply(text)
    }
}

impl<S: Synthesizer> Synthesizer for Lexical<S> {
    fn speak(&self, text: &str, speaker: &SpeakerProfile) -> Result<Clip, PolicyError> {
        self.inner.speak(&self.apply(text), speaker)
    }

    fn speak_streaming(
        &self,
        text: &str,
        speaker: &SpeakerProfile,
        on_audio: &mut dyn FnMut(&[f32]),
    ) -> Result<Clip, PolicyError> {
        self.inner
            .speak_streaming(&self.apply(text), speaker, on_audio)
    }
}
