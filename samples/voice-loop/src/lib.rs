// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements spoken dialogue systems whose every stage is
// measured on its own, for its clients. If your team needs expertise in
// speech recognition, speech synthesis and their evaluation, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Can a persona be talked to and talk back, and what does each stage of the
//! loop cost and lose?
//!
//! The reference loop is a cascade: recognise the listener, let the persona
//! answer in text, speak the answer. It is written on Splinter's SDK alone.
//! The reports here are what a speech-native model is later compared with, so
//! they keep what was heard and what was said, and time every stage.

use std::time::Duration;

use serde::Serialize;
use splinter_sdk::model::speech::{RoundTrip, RoundTripItem, StageTimings, Turn};
use splinter_sdk::vocabulary::speech::SpeakerProfile;

/// The sentences in a text file: one per line, blank lines and lines starting
/// with `#` skipped.
#[must_use]
pub fn read_sentences(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_string)
        .collect()
}

/// A round trip, as `roundtrip` prints and keeps it.
#[derive(Debug, Serialize)]
pub struct RoundTripReport<'a> {
    /// The recognition model.
    pub recognizer: &'a str,
    /// The synthesis model.
    pub synthesizer: &'a str,
    /// The speaker, with the portrayal it declares.
    pub speaker: &'a SpeakerProfile,
    /// How many sentences were spoken.
    pub sentences: usize,
    /// Total word errors over total words; absent when no words were spoken.
    pub corpus_word_error_rate: Option<f32>,
    /// Sentences that lost more than half their words.
    pub mostly_lost: usize,
    /// Each sentence, what was heard, and its rate.
    pub items: &'a [RoundTripItem],
}

impl<'a> RoundTripReport<'a> {
    /// The report of `round_trip` made with these models and this speaker.
    #[must_use]
    pub fn new(
        recognizer: &'a str,
        synthesizer: &'a str,
        speaker: &'a SpeakerProfile,
        round_trip: &'a RoundTrip,
    ) -> Self {
        Self {
            recognizer,
            synthesizer,
            speaker,
            sentences: round_trip.items.len(),
            corpus_word_error_rate: round_trip.corpus_word_error_rate(),
            mostly_lost: round_trip.mostly_lost(),
            items: &round_trip.items,
        }
    }
}

/// Seconds, the unit every duration in a report is written in.
fn seconds(d: Duration) -> f64 {
    d.as_secs_f64()
}

/// One turn, as `turn` prints and keeps it.
#[derive(Debug, Serialize)]
pub struct TurnReport<'a> {
    /// The persona that answered.
    pub persona: &'a str,
    /// The speaker, with the portrayal it declares.
    pub speaker: &'a SpeakerProfile,
    /// What the listener was heard to say.
    pub heard: &'a str,
    /// What the persona answered.
    pub answer: &'a str,
    /// Length of the spoken reply in seconds.
    pub reply_seconds: f64,
    /// Seconds recognising.
    pub recognise_seconds: f64,
    /// Seconds composing the answer.
    pub respond_seconds: f64,
    /// Seconds speaking it.
    pub synthesise_seconds: f64,
    /// The three together.
    pub total_seconds: f64,
}

impl<'a> TurnReport<'a> {
    /// The report of `turn` taken by `persona` in `speaker`'s voice.
    #[must_use]
    pub fn new(persona: &'a str, speaker: &'a SpeakerProfile, turn: &'a Turn) -> Self {
        let StageTimings {
            recognise,
            respond,
            synthesise,
        } = turn.timings;
        Self {
            persona,
            speaker,
            heard: &turn.heard,
            answer: &turn.answer,
            reply_seconds: turn.reply.seconds(),
            recognise_seconds: seconds(recognise),
            respond_seconds: seconds(respond),
            synthesise_seconds: seconds(synthesise),
            total_seconds: seconds(turn.timings.total()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use splinter_sdk::model::speech::Clip;
    use splinter_sdk::vocabulary::speech::Portrayal;

    #[test]
    fn a_sentence_file_skips_blank_lines_and_comments() {
        let text = "# spoken-style sentences\n\nThe quick brown fox.\n  Jumps over.  \n#skipped\n";
        assert_eq!(
            read_sentences(text),
            ["The quick brown fox.", "Jumps over."]
        );
    }

    #[test]
    fn a_report_carries_the_portrayal_and_leaves_an_unmeasured_rate_absent() {
        let speaker = SpeakerProfile::new(1, Portrayal::synthetic_theatrical());
        let empty = RoundTrip { items: Vec::new() };
        let json =
            serde_json::to_value(RoundTripReport::new("asr", "tts", &speaker, &empty)).unwrap();

        assert!(json["speaker"]["portrayal"]
            .as_str()
            .unwrap()
            .contains("no recording"));
        assert!(
            json["corpus_word_error_rate"].is_null(),
            "not measured is absent, not zero"
        );
        assert_eq!(json["sentences"], 0);
    }

    #[test]
    fn a_turn_report_totals_the_stages() {
        let speaker = SpeakerProfile::new(1, Portrayal::synthetic_theatrical());
        let turn = Turn {
            heard: "Hello".into(),
            answer: "Good day.".into(),
            reply: Clip::new(vec![0.0; 8_000], 16_000),
            timings: StageTimings {
                recognise: Duration::from_millis(100),
                respond: Duration::from_millis(200),
                synthesise: Duration::from_millis(300),
            },
        };
        let json = serde_json::to_value(TurnReport::new("Samuel Adams", &speaker, &turn)).unwrap();

        assert_eq!(json["reply_seconds"], 0.5);
        assert_eq!(json["total_seconds"], 0.6);
        assert_eq!(json["persona"], "Samuel Adams");
    }
}
