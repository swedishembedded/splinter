// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements spoken dialogue systems whose every stage is
// measured on its own, for its clients. If your team needs expertise in speech
// recognition, speech synthesis and their evaluation, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Speech in, speech out, around a model that reads and writes text.
//!
//! This is the cascade: a [`Recognizer`] turns the listener's speech into
//! text, the persona answers in text, and a [`Synthesizer`] speaks the answer
//! in a [`SpeakerProfile`]'s voice. It is the reference every later,
//! speech-native stage is measured against, and the source of the spoken
//! examples those stages learn from. Each stage is timed on its own, because
//! which one costs the time is what decides where to work next.
//!
//! brain does the recognition and the synthesis ([`BrainRecognizer`],
//! [`BrainSynthesizer`]); this module decides nothing about when they are
//! used. [`round_trip`] measures what survives being spoken and heard again,
//! with the word error rate brain's own gates use.

mod brain_backed;
mod cascade;
mod ingress;
mod lexical;
mod listener;
mod round_trip;
mod sentences;
mod spoken_turn;
mod turn;
mod verified;
mod window;

#[cfg(any(test, feature = "scripted"))]
pub mod scripted;
#[cfg(test)]
mod tests;

pub use brain::Audio as Clip;
pub use brain_backed::{
    BrainRecognizer, BrainSynthesizer, DEFAULT_RECOGNIZER, DEFAULT_SYNTHESIZER,
};
pub use cascade::CascadeListener;
pub use ingress::{
    AudioFeatures, Ingress, IngressExample, IngressOptions, StepSettings, FEATURE_WIDTH,
};
pub use lexical::Lexical;
pub use listener::{BrainListener, ListenerOptions};
pub use round_trip::corpus_word_error_rate;
pub use round_trip::{round_trip, RoundTrip, RoundTripItem};
pub use sentences::{SentenceStream, Sentences, PAUSE_MILLIS};
pub use spoken_turn::{take_spoken_turn, Listener, SpokenTimings, SpokenTurn};
pub use turn::{take_turn, StageTimings, Turn};
pub use verified::{speak_verified, Verified};
pub use window::padded_to;

use splinter_core::speech::SpeakerProfile;

use crate::error::PolicyError;

/// What a listener's speech was heard as.
#[derive(Clone, Debug, PartialEq)]
pub struct Transcription {
    /// The words heard.
    pub text: String,
    /// Whether the recognizer dropped part of the clip because it was longer
    /// than the recognizer can take at once.
    pub truncated: bool,
}

/// Speech to text.
pub trait Recognizer {
    /// What `clip` says. A clip with no recognisable speech is an empty text,
    /// not an error: the caller decides what silence means.
    fn transcribe(&self, clip: &Clip) -> Result<Transcription, PolicyError>;
}

/// Text to speech.
pub trait Synthesizer {
    /// `text`, spoken by `speaker`. The same speaker and the same text give the
    /// same voice.
    fn speak(&self, text: &str, speaker: &SpeakerProfile) -> Result<Clip, PolicyError>;

    /// [`Self::speak`], handing `on_audio` the samples as they are made, so
    /// sound can start before the text is finished. The returned clip is the
    /// whole; the pieces joined are the same samples. A synthesizer that cannot
    /// stream hands over everything at the end, which is this default.
    fn speak_streaming(
        &self,
        text: &str,
        speaker: &SpeakerProfile,
        on_audio: &mut dyn FnMut(&[f32]),
    ) -> Result<Clip, PolicyError> {
        let clip = self.speak(text, speaker)?;
        on_audio(clip.samples());
        Ok(clip)
    }
}
