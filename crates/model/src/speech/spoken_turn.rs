// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A turn in which the model listens to the audio itself, and speaks each
//! sentence as soon as it is written.

use std::sync::mpsc;
use std::thread;
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
    /// From the start to the first word the model wrote; a cascade's
    /// recognition and the model's first token are in it.
    pub first_text: Duration,
    /// From the start to the first piece handed to be spoken: the wait for a
    /// clause end after the first word.
    pub first_piece: Duration,
    /// From the start until the model had written its whole answer; speaking
    /// runs beside it, so this and `synthesise` overlap.
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
/// each is written. Speaking runs beside the listener, so the model goes on
/// writing while the first sentences are heard. An answer with no words is an
/// error, not silence.
pub fn take_spoken_turn(
    listener: &dyn Listener,
    synthesizer: &(dyn Synthesizer + Sync),
    speaker: &SpeakerProfile,
    input: &Clip,
    max_words: usize,
) -> Result<SpokenTurn, PolicyError> {
    let started = Instant::now();
    let mut stream = SentenceStream::new(max_words);
    let mut first_text = None;
    let mut first_piece = None;
    // Bounded, so a synthesizer that falls behind slows the writer rather
    // than letting the unspoken text grow without limit.
    let (queue, sentences) = mpsc::sync_channel::<String>(QUEUED_SENTENCES);

    let (answer, answered, spoken) = thread::scope(|scope| {
        let speaking = scope.spawn(move || {
            let mut clips: Vec<Clip> = Vec::new();
            let mut first_audio = None;
            let mut synthesise = Duration::ZERO;
            for sentence in sentences {
                let began = Instant::now();
                let clip = synthesizer.speak_streaming(&sentence, speaker, &mut |_| {
                    first_audio.get_or_insert_with(|| started.elapsed());
                })?;
                synthesise += began.elapsed();
                clips.push(clip);
            }
            Ok::<_, PolicyError>((clips, first_audio, synthesise))
        });
        // A send fails only once the speaking side has stopped on an error,
        // which the join below reports; the writer has nothing more to do.
        let answer = listener.answer(input, &mut |delta| {
            first_text.get_or_insert_with(|| started.elapsed());
            for sentence in stream.push(delta) {
                first_piece.get_or_insert_with(|| started.elapsed());
                let _ = queue.send(sentence);
            }
        });
        let answered = started.elapsed();
        if answer.is_ok() {
            for sentence in stream.finish() {
                let _ = queue.send(sentence);
            }
        }
        drop(queue);
        let spoken = speaking.join().unwrap_or_else(|_| {
            Err(PolicyError::Synthesis {
                reason: "the speaking thread panicked".to_string(),
            })
        });
        (answer, answered, spoken)
    });

    let answer = answer?;
    if answer.trim().is_empty() {
        return Err(PolicyError::Synthesis {
            reason: "the model gave no answer".to_string(),
        });
    }
    let (clips, first_audio, synthesise) = spoken?;
    let total = started.elapsed();
    Ok(SpokenTurn {
        answer,
        reply: join(&clips)?,
        timings: SpokenTimings {
            first_audio: first_audio.unwrap_or(total),
            first_text: first_text.unwrap_or(total),
            first_piece: first_piece.unwrap_or(answered),
            think: answered,
            synthesise,
            total,
        },
    })
}

/// How many written sentences may wait to be spoken.
const QUEUED_SENTENCES: usize = 16;
