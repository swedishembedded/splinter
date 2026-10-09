// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A turn in which the model listens to the audio itself, and speaks each
//! sentence as soon as it is written.

use std::time::{Duration, Instant};

use splinter_core::speech::SpeakerProfile;

use super::sentences::{join, SentenceStream};
use super::{Clip, Synthesizer};
use crate::error::PolicyError;

/// A model that hears speech and answers in text.
pub trait Listener {
    /// The answer to what `clip` says. `on_text` is handed each new piece of the
    /// answer as it is written; the returned text is the whole answer.
    fn answer(&self, clip: &Clip, on_text: &mut dyn FnMut(&str)) -> Result<String, PolicyError>;
}

/// How long a spoken turn took, on the wall clock.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpokenTimings {
    /// From the end of the question to the first sentence spoken.
    pub first_audio: Duration,
    /// The model listening and writing, speaking left out.
    pub think: Duration,
    /// Speaking, summed over the sentences.
    pub synthesise: Duration,
    /// The whole turn.
    pub total: Duration,
}

/// What a spoken turn produced.
#[derive(Clone, Debug)]
pub struct SpokenTurn {
    /// What the model answered, as text.
    pub answer: String,
    /// The answer, spoken.
    pub reply: Clip,
    /// What each part cost.
    pub timings: SpokenTimings,
}

/// Let `listener` hear `input` and speak its answer one sentence at a time, as
/// each is written. An answer with no words is an error, not silence.
pub fn take_spoken_turn(
    listener: &dyn Listener,
    synthesizer: &dyn Synthesizer,
    speaker: &SpeakerProfile,
    input: &Clip,
    max_words: usize,
) -> Result<SpokenTurn, PolicyError> {
    let started = Instant::now();
    let mut stream = SentenceStream::new(max_words);
    let mut clips: Vec<Clip> = Vec::new();
    let mut first_audio = None;
    let mut synthesise = Duration::ZERO;
    let mut failure: Option<PolicyError> = None;
    let mut speak = |sentence: &str, clips: &mut Vec<Clip>, failure: &mut Option<PolicyError>| {
        if failure.is_some() {
            return;
        }
        let began = Instant::now();
        match synthesizer.speak(sentence, speaker) {
            Ok(clip) => {
                clips.push(clip);
                first_audio.get_or_insert_with(|| started.elapsed());
            }
            Err(e) => *failure = Some(e),
        }
        synthesise += began.elapsed();
    };

    let answer = listener.answer(input, &mut |delta| {
        for sentence in stream.push(delta) {
            speak(&sentence, &mut clips, &mut failure);
        }
    })?;
    if answer.trim().is_empty() {
        return Err(PolicyError::Synthesis {
            reason: "the model gave no answer".to_string(),
        });
    }
    for sentence in stream.finish() {
        speak(&sentence, &mut clips, &mut failure);
    }
    if let Some(e) = failure {
        return Err(e);
    }
    let total = started.elapsed();
    Ok(SpokenTurn {
        answer,
        reply: join(&clips)?,
        timings: SpokenTimings {
            first_audio: first_audio.unwrap_or(total),
            think: total.saturating_sub(synthesise),
            synthesise,
            total,
        },
    })
}
