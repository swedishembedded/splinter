// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The cascade as a [`Listener`]: recognise the question, let a text model
//! answer it, and hand the answer on as it is written, so the first sentence
//! can be spoken while the rest is still being composed.

use super::{Clip, Listener, Recognizer};
use crate::error::PolicyError;

/// A recogniser in front of a text model that streams its answer.
pub struct CascadeListener<R, A> {
    recognizer: R,
    answer: A,
}

impl<R, A> CascadeListener<R, A>
where
    R: Recognizer,
    A: Fn(&str, &mut dyn FnMut(&str)) -> Result<String, PolicyError>,
{
    /// `answer` is given the transcript and a callback for each new piece of
    /// the answer; it returns the whole answer.
    pub fn new(recognizer: R, answer: A) -> Self {
        Self { recognizer, answer }
    }
}

impl<R, A> Listener for CascadeListener<R, A>
where
    R: Recognizer,
    A: Fn(&str, &mut dyn FnMut(&str)) -> Result<String, PolicyError>,
{
    fn answer(&self, clip: &Clip, on_text: &mut dyn FnMut(&str)) -> Result<String, PolicyError> {
        let heard = self.recognizer.transcribe(clip)?;
        if heard.text.trim().is_empty() {
            return Err(PolicyError::Transcription {
                reason: format!(
                    "no speech was recognised in {:.1} seconds of audio",
                    clip.seconds()
                ),
            });
        }
        (self.answer)(&heard.text, on_text)
    }
}
