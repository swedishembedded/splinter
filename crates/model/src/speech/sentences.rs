// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Long text spoken a sentence at a time.
//!
//! A synthesizer renders a bounded stretch of speech and stops: a persona's
//! answer of a hundred words is cut off mid-sentence. Speaking it piece by
//! piece, at sentence ends, keeps every word and lets the first sentence be
//! heard while the rest is still being made.

use splinter_core::speech::SpeakerProfile;

use super::{Clip, Synthesizer};
use crate::error::PolicyError;

/// The silence between two spoken pieces, in milliseconds.
pub const PAUSE_MILLIS: u32 = 250;

/// Words that end in a full stop without ending a sentence.
const ABBREVIATIONS: [&str; 12] = [
    "mr", "mrs", "ms", "dr", "st", "gen", "col", "capt", "gov", "hon", "rev", "jr",
];

/// A synthesizer that speaks text of any length through one that speaks a
/// bounded stretch.
pub struct Sentences<S> {
    inner: S,
    max_words: usize,
}

impl<S> Sentences<S> {
    /// `inner`, asked for no more than `max_words` words at a time.
    pub fn new(inner: S, max_words: usize) -> Self {
        Self {
            inner,
            max_words: max_words.max(1),
        }
    }

    /// The synthesizer that does the speaking.
    pub fn inner(&self) -> &S {
        &self.inner
    }
}

impl<S: Synthesizer> Synthesizer for Sentences<S> {
    fn speak(&self, text: &str, speaker: &SpeakerProfile) -> Result<Clip, PolicyError> {
        let pieces = split(text, self.max_words);
        if pieces.is_empty() {
            return Err(PolicyError::Synthesis {
                reason: "there are no words to speak".to_string(),
            });
        }
        let mut samples = Vec::new();
        let mut rate = None;
        for piece in &pieces {
            let clip = self.inner.speak(piece, speaker)?;
            let expected = *rate.get_or_insert(clip.sample_rate());
            if clip.sample_rate() != expected {
                return Err(PolicyError::Synthesis {
                    reason: format!(
                        "pieces of one text came back at {expected} Hz and {} Hz",
                        clip.sample_rate()
                    ),
                });
            }
            if !samples.is_empty() {
                let pause = (u64::from(expected) * u64::from(PAUSE_MILLIS) / 1000) as usize;
                samples.resize(samples.len() + pause, 0.0);
            }
            samples.extend_from_slice(clip.samples());
        }
        Ok(Clip::new(samples, rate.unwrap_or_default()))
    }
}

/// `text` as pieces of at most `max_words` words, broken at sentence ends,
/// then at the last comma inside the bound, then at the bound.
fn split(text: &str, max_words: usize) -> Vec<String> {
    let mut pieces = Vec::new();
    for sentence in sentences(text) {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        let mut rest = &words[..];
        while !rest.is_empty() {
            let take = if rest.len() <= max_words {
                rest.len()
            } else {
                rest[..max_words]
                    .iter()
                    .rposition(|w| w.ends_with(','))
                    .map_or(max_words, |i| i + 1)
            };
            pieces.push(rest[..take].join(" "));
            rest = &rest[take..];
        }
    }
    pieces
}

/// The sentences of `text`: broken after `.`, `!` or `?` that ends a word,
/// except after an abbreviation.
fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    for word in text.split_whitespace() {
        current.push(word);
        let closed = word.trim_end_matches(['"', '\'', ')']);
        let ends = closed.ends_with(['.', '!', '?']);
        let abbreviation = closed.ends_with('.')
            && ABBREVIATIONS.contains(&closed.trim_end_matches('.').to_lowercase().as_str());
        if ends && !abbreviation {
            out.push(current.join(" "));
            current.clear();
        }
    }
    if !current.is_empty() {
        out.push(current.join(" "));
    }
    out
}
