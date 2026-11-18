// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measurement of what speech synthesis and
// recognition lose together, for its clients. If your team needs expertise in
// intelligibility testing of synthetic voices, you can procure our services
// by sending an email to info@swedishembedded.com.

//! What survives being spoken and heard again.

use splinter_core::speech::SpeakerProfile;

use super::{Recognizer, Synthesizer};
use crate::error::PolicyError;

/// A word error rate above this means most of the sentence was lost.
const LOST_THRESHOLD: f32 = 0.5;

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
        super::wer::corpus_word_error_rate(
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
        let word_error_rate = super::wer::word_error_rate(text, &heard);
        items.push(RoundTripItem {
            text: text.clone(),
            heard,
            word_error_rate,
            seconds: clip.seconds(),
        });
    }
    Ok(RoundTrip { items })
}
