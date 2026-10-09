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

pub mod batch;
pub mod ingress;
pub mod spoken;

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

/// The median and the 95th percentile of a set of measurements, by nearest
/// rank.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Spread {
    /// Half the measurements are at or below this.
    pub p50: f64,
    /// Nineteen in twenty are at or below this.
    pub p95: f64,
}

impl Spread {
    /// The spread of `values`; `None` for no measurements, because a latency
    /// that was not measured is not zero.
    #[must_use]
    pub fn of(values: &[f64]) -> Option<Spread> {
        if values.is_empty() {
            return None;
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let rank = |p: f64| {
            let n = ((p / 100.0) * sorted.len() as f64).ceil().max(1.0) as usize;
            sorted[n.min(sorted.len()) - 1]
        };
        Some(Spread {
            p50: rank(50.0),
            p95: rank(95.0),
        })
    }
}

/// Each recording with the question it was made from, when the questions are
/// known. The questions are matched to the recordings in order, so there must
/// be exactly as many.
pub fn pair_questions(
    recordings: &[std::path::PathBuf],
    questions: Option<&[String]>,
) -> Result<Vec<(std::path::PathBuf, Option<String>)>, String> {
    match questions {
        None => Ok(recordings.iter().map(|r| (r.clone(), None)).collect()),
        Some(q) if q.len() == recordings.len() => Ok(recordings
            .iter()
            .cloned()
            .zip(q.iter().cloned().map(Some))
            .collect()),
        Some(q) => Err(format!(
            "{} questions for {} recordings: they are matched in order, so the counts must agree",
            q.len(),
            recordings.len()
        )),
    }
}

/// One turn of a batch.
#[derive(Debug, Serialize)]
pub struct TurnItem {
    /// The recording.
    pub recording: String,
    /// What was asked, when known.
    pub asked: Option<String>,
    /// What was heard.
    pub heard: String,
    /// What the persona answered.
    pub answer: String,
    /// The spoken answer, heard again by the recognizer.
    pub answer_heard: String,
    /// Length of the spoken answer in seconds.
    pub reply_seconds: f64,
    /// Seconds recognising the question.
    pub recognise_seconds: f64,
    /// Seconds composing the answer.
    pub respond_seconds: f64,
    /// Seconds speaking it.
    pub synthesise_seconds: f64,
}

/// A batch of turns, as `turns` prints and keeps it.
#[derive(Debug, Serialize)]
pub struct TurnsReport<'a> {
    /// The persona that answered.
    pub persona: &'a str,
    /// The speaker, with the portrayal it declares.
    pub speaker: &'a SpeakerProfile,
    /// How many turns were taken.
    pub turns: usize,
    /// Question words lost in recognition; absent when the questions are not known.
    pub question_word_error_rate: Option<f32>,
    /// Answer words lost between being spoken and being heard again.
    pub answer_word_error_rate: Option<f32>,
    /// Seconds recognising.
    pub recognise_seconds: Option<Spread>,
    /// Seconds composing the answer.
    pub respond_seconds: Option<Spread>,
    /// Seconds speaking it.
    pub synthesise_seconds: Option<Spread>,
    /// The three together, per turn.
    pub total_seconds: Option<Spread>,
    /// Each turn.
    pub items: &'a [TurnItem],
}

/// A sentence kept as a recording.
#[derive(Debug, Serialize)]
pub struct Recorded {
    /// The sentence.
    pub text: String,
    /// The recording, relative to the output directory.
    pub file: String,
    /// The seed whose voice was heard back well enough.
    pub seed: u64,
    /// How many seeds were tried.
    pub attempts: usize,
    /// Words wrong over words spoken when it was heard back.
    pub word_error_rate: f32,
    /// Length of the recording in seconds.
    pub seconds: f64,
}

/// A set of sentences recorded, as `speak-set` prints and keeps it.
#[derive(Debug, Serialize)]
pub struct SpeakSetReport<'a> {
    /// The recognition model that admitted the recordings.
    pub recognizer: &'a str,
    /// The synthesis model.
    pub synthesizer: &'a str,
    /// The portrayal every speaker declares.
    pub portrayal: &'a str,
    /// The word error rate above which a recording is not kept.
    pub max_word_error_rate: f32,
    /// Recordings kept.
    pub kept: &'a [Recorded],
    /// Sentences no seed spoke well enough: not recorded.
    pub rejected: &'a [String],
}

/// One turn of a batch answered by a model that listens.
#[derive(Debug, Serialize)]
pub struct ListenItem {
    /// The recording.
    pub recording: String,
    /// What was asked, when known.
    pub asked: Option<String>,
    /// What the model answered.
    pub answer: String,
    /// The spoken answer, heard again by a recogniser that did not make it.
    pub answer_heard: String,
    /// Length of the spoken answer in seconds.
    pub reply_seconds: f64,
    /// Seconds from the end of the question to the first sentence spoken.
    pub first_audio_seconds: f64,
    /// Seconds the model spent listening and writing.
    pub think_seconds: f64,
    /// Seconds speaking, summed over the sentences.
    pub synthesise_seconds: f64,
    /// The whole turn.
    pub total_seconds: f64,
}

/// A batch of turns answered by a model that listens, as `listen-turns` keeps it.
#[derive(Debug, Serialize)]
pub struct ListenReport<'a> {
    /// The persona that answered.
    pub persona: &'a str,
    /// The speaker, with the portrayal it declares.
    pub speaker: &'a SpeakerProfile,
    /// How many turns were taken.
    pub turns: usize,
    /// Answer words lost between being spoken and being heard again.
    pub answer_word_error_rate: Option<f32>,
    /// Seconds from the end of the question to the first sentence spoken.
    pub first_audio_seconds: Option<Spread>,
    /// Seconds listening and writing.
    pub think_seconds: Option<Spread>,
    /// Seconds speaking.
    pub synthesise_seconds: Option<Spread>,
    /// The whole turn.
    pub total_seconds: Option<Spread>,
    /// Each turn.
    pub items: &'a [ListenItem],
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
    fn a_spread_is_the_nearest_rank_percentiles_and_absent_for_nothing() {
        let values: Vec<f64> = (1..=10).map(f64::from).collect();
        assert_eq!(
            Spread::of(&values),
            Some(Spread {
                p50: 5.0,
                p95: 10.0
            })
        );
        assert_eq!(Spread::of(&[2.5]), Some(Spread { p50: 2.5, p95: 2.5 }));
        assert_eq!(Spread::of(&[]), None, "no turns, no latency; not zero");
        // Order does not matter.
        assert_eq!(
            Spread::of(&[3.0, 1.0, 2.0]),
            Some(Spread { p50: 2.0, p95: 3.0 })
        );
    }

    #[test]
    fn questions_pair_with_recordings_in_order_and_a_mismatch_is_refused() {
        let wavs = ["q01.wav", "q02.wav"].map(std::path::PathBuf::from);
        let paired =
            pair_questions(&wavs, Some(&["Why?".to_string(), "How?".to_string()])).unwrap();
        assert_eq!(paired[1].1.as_deref(), Some("How?"));
        assert!(pair_questions(&wavs, Some(&["Why?".to_string()])).is_err());
        assert!(pair_questions(&wavs, None)
            .unwrap()
            .iter()
            .all(|(_, q)| q.is_none()));
    }

    #[test]
    fn a_speak_set_report_names_what_was_rejected() {
        let report = SpeakSetReport {
            recognizer: "asr",
            synthesizer: "tts",
            portrayal: "Synthetic",
            max_word_error_rate: 0.2,
            kept: &[],
            rejected: &["A sentence nobody could speak.".to_string()],
        };
        let json = serde_json::to_value(report).unwrap();
        assert_eq!(json["rejected"][0], "A sentence nobody could speak.");
        assert_eq!(json["kept"].as_array().unwrap().len(), 0);
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
