// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Speech that is kept only if it can be understood again.
//!
//! A synthesizer sometimes mumbles: the same sentence from another seed can be
//! twice as long, or too quiet to recognise at all. A clip is therefore
//! admitted only when a recognizer that did not make it hears the sentence
//! back well enough, and a sentence no seed manages to speak is rejected
//! rather than kept badly.

use brain::wer;
use splinter_core::speech::SpeakerProfile;

use super::{Clip, Recognizer, Synthesizer};
use crate::error::PolicyError;

/// A clip that was heard back.
#[derive(Clone, Debug)]
pub struct Verified {
    /// The speech.
    pub clip: Clip,
    /// The speaker that spoke it: the seed that worked, with the portrayal.
    pub speaker: SpeakerProfile,
    /// What the recognizer heard.
    pub heard: String,
    /// Words wrong over words spoken.
    pub word_error_rate: f32,
    /// How many seeds were tried, this one included.
    pub attempts: usize,
}

/// Speak `text` in `speaker`'s voice and keep it if it is heard back with a
/// word error rate of at most `max_word_error_rate`; otherwise try the next
/// seed, up to `attempts` seeds in all. `None` when none is heard well enough.
pub fn speak_verified(
    synthesizer: &dyn Synthesizer,
    recognizer: &dyn Recognizer,
    speaker: &SpeakerProfile,
    text: &str,
    max_word_error_rate: f32,
    attempts: usize,
) -> Result<Option<Verified>, PolicyError> {
    for attempt in 0..attempts {
        let candidate = SpeakerProfile::new(
            speaker.seed().wrapping_add(attempt as u64),
            speaker.portrayal().clone(),
        );
        let clip = synthesizer.speak(text, &candidate)?;
        let heard = recognizer.transcribe(&clip)?.text;
        let word_error_rate = wer::word_error_rate(&wer::normalize(text), &wer::normalize(&heard));
        if word_error_rate <= max_word_error_rate {
            return Ok(Some(Verified {
                clip,
                speaker: candidate,
                heard,
                word_error_rate,
                attempts: attempt + 1,
            }));
        }
    }
    Ok(None)
}
