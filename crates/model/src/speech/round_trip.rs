// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What survives being spoken and heard again.

use brain::wer;
use splinter_core::speech::SpeakerProfile;

use super::{Recognizer, Synthesizer};
use crate::error::PolicyError;

/// A word error rate above this means most of the sentence was lost.
const LOST_THRESHOLD: f32 = 0.5;

/// Total word errors over total words of `(spoken, heard)` pairs, after case
/// and punctuation are ignored; `None` when no words were spoken.
#[must_use]
pub fn corpus_word_error_rate<'a>(
    pairs: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Option<f32> {
    wer::corpus_wer(pairs)
}

/// One sentence, spoken and heard again.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct RoundTripItem {
    /// The sentence that was spoken.
    pub text: String,
    /// What was heard.
    pub heard: String,
    /// Words wrong over words spoken, after case and punctuation are ignored.
    pub word_error_rate: f32,
    /// Length of the spoken clip in seconds.
    pub seconds: f64,
}

/// A set of sentences spoken and heard again.
#[derive(Clone, Debug, PartialEq)]
pub struct RoundTrip {
    /// Each sentence.
    pub items: Vec<RoundTripItem>,
}

impl RoundTrip {
    /// Total word errors over total words spoken: long sentences weigh more
    /// than short ones. `None` when no words were spoken, because a rate that
    /// was not measured is not zero.
    #[must_use]
    pub fn corpus_word_error_rate(&self) -> Option<f32> {
        wer::corpus_wer(
            self.items
                .iter()
                .map(|i| (i.text.as_str(), i.heard.as_str())),
        )
    }

    /// How many sentences lost more than half their words.
    #[must_use]
    pub fn mostly_lost(&self) -> usize {
        self.items
            .iter()
            .filter(|i| i.word_error_rate > LOST_THRESHOLD)
            .count()
    }
}

/// Speak each of `sentences` in `speaker`'s voice, have `recognizer` hear it,
/// and score what was heard against what was spoken.
pub fn round_trip(
    synthesizer: &dyn Synthesizer,
    recognizer: &dyn Recognizer,
    speaker: &SpeakerProfile,
    sentences: &[String],
) -> Result<RoundTrip, PolicyError> {
    let mut items = Vec::with_capacity(sentences.len());
    for text in sentences {
        let clip = synthesizer.speak(text, speaker)?;
        let heard = recognizer.transcribe(&clip)?.text;
        let word_error_rate = wer::word_error_rate(&wer::normalize(text), &wer::normalize(&heard));
        items.push(RoundTripItem {
            text: text.clone(),
            heard,
            word_error_rate,
            seconds: clip.seconds(),
        });
    }
    Ok(RoundTrip { items })
}
