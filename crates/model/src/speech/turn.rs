// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! One turn of the cascade: hear, answer, speak.

use std::time::{Duration, Instant};

use splinter_core::speech::SpeakerProfile;

use super::{Clip, Recognizer, Synthesizer};
use crate::error::PolicyError;

/// How long each stage of a turn took, measured on the wall clock.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StageTimings {
    /// Recognising the listener.
    pub recognise: Duration,
    /// The persona composing its answer.
    pub respond: Duration,
    /// Speaking the answer.
    pub synthesise: Duration,
}

impl StageTimings {
    /// The three stages together.
    #[must_use]
    pub fn total(&self) -> Duration {
        self.recognise + self.respond + self.synthesise
    }
}

/// What a turn produced.
#[derive(Clone, Debug)]
pub struct Turn {
    /// What the listener was heard to say.
    pub heard: String,
    /// What the persona answered, as text.
    pub answer: String,
    /// The answer, spoken.
    pub reply: Clip,
    /// What each stage cost.
    pub timings: StageTimings,
}

/// Hear `input`, have `respond` answer what was heard, and speak the answer.
///
/// Silence is not answered: a clip with no recognisable speech, or an answer
/// with no words, ends the turn with an error naming the stage, because
/// speaking a made-up reply to nothing is worse than saying nothing.
pub fn take_turn(
    recognizer: &dyn Recognizer,
    respond: &mut dyn FnMut(&str) -> Result<String, PolicyError>,
    synthesizer: &dyn Synthesizer,
    speaker: &SpeakerProfile,
    input: &Clip,
) -> Result<Turn, PolicyError> {
    let started = Instant::now();
    let heard = recognizer.transcribe(input)?;
    let recognise = started.elapsed();
    if heard.text.trim().is_empty() {
        return Err(PolicyError::Transcription {
            reason: format!(
                "no speech was recognised in {:.1} seconds of audio",
                input.seconds()
            ),
        });
    }

    let started = Instant::now();
    let answer = respond(&heard.text)?;
    let respond_time = started.elapsed();
    if answer.trim().is_empty() {
        return Err(PolicyError::Synthesis {
            reason: format!("the persona gave no answer to {:?}", heard.text),
        });
    }

    let started = Instant::now();
    let reply = synthesizer.speak(&answer, speaker)?;
    let synthesise = started.elapsed();

    Ok(Turn {
        heard: heard.text,
        answer,
        reply,
        timings: StageTimings {
            recognise,
            respond: respond_time,
            synthesise,
        },
    })
}
