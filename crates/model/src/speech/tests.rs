// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Specs of the cascade: what a turn hears, answers and speaks, and what a
//! round trip says was lost.

use splinter_core::speech::{Portrayal, SpeakerProfile};

use super::scripted::{ScriptedRecognizer, ScriptedSynthesizer};
use super::*;

fn speaker() -> SpeakerProfile {
    SpeakerProfile::new(1, Portrayal::synthetic_theatrical())
}

fn said(text: &str) -> Clip {
    ScriptedSynthesizer.speak(text, &speaker()).unwrap()
}

#[test]
fn a_turn_answers_what_was_heard_and_speaks_the_answer() {
    let mut asked = Vec::new();
    let mut respond = |heard: &str| {
        asked.push(heard.to_string());
        Ok("Liberty is not given.".to_string())
    };
    let turn = take_turn(
        &ScriptedRecognizer::perfect(),
        &mut respond,
        &ScriptedSynthesizer,
        &speaker(),
        &said("What is liberty?"),
    )
    .unwrap();

    assert_eq!(asked, ["What is liberty?"]);
    assert_eq!(turn.heard, "What is liberty?");
    assert_eq!(turn.answer, "Liberty is not given.");
    assert_eq!(
        turn.reply,
        said("Liberty is not given."),
        "the answer is what was spoken"
    );
    assert_eq!(
        turn.timings.total(),
        turn.timings.recognise + turn.timings.respond + turn.timings.synthesise
    );
}

#[test]
fn silence_is_not_answered() {
    let mut called = false;
    let mut respond = |_: &str| {
        called = true;
        Ok("anything".to_string())
    };
    let err = take_turn(
        &ScriptedRecognizer::perfect(),
        &mut respond,
        &ScriptedSynthesizer,
        &speaker(),
        &said("   "),
    )
    .unwrap_err();

    assert!(matches!(err, PolicyError::Transcription { .. }), "{err}");
    assert!(!called, "the persona must not be asked to answer nothing");
}

#[test]
fn an_empty_answer_is_not_spoken() {
    let mut respond = |_: &str| Ok("  ".to_string());
    let err = take_turn(
        &ScriptedRecognizer::perfect(),
        &mut respond,
        &ScriptedSynthesizer,
        &speaker(),
        &said("Hello"),
    )
    .unwrap_err();

    assert!(matches!(err, PolicyError::Synthesis { .. }), "{err}");
}

#[test]
fn a_failing_stage_stops_the_turn_with_its_own_error() {
    let mut respond = |_: &str| {
        Err(PolicyError::Remote {
            spec: "policy".into(),
            reason: "down".into(),
        })
    };
    let err = take_turn(
        &ScriptedRecognizer::perfect(),
        &mut respond,
        &ScriptedSynthesizer,
        &speaker(),
        &said("Hello"),
    )
    .unwrap_err();

    assert!(err.to_string().contains("down"), "{err}");
}

#[test]
fn a_round_trip_with_perfect_hearing_loses_nothing() {
    let sentences = vec![
        "The quick brown fox.".to_string(),
        "Jumps over.".to_string(),
    ];
    let report = round_trip(
        &ScriptedSynthesizer,
        &ScriptedRecognizer::perfect(),
        &speaker(),
        &sentences,
    )
    .unwrap();

    assert_eq!(report.corpus_word_error_rate(), Some(0.0));
    assert_eq!(report.mostly_lost(), 0);
}

#[test]
fn the_corpus_rate_weighs_a_long_sentence_more_than_a_short_one() {
    // The listener loses the last word of each sentence: one of two words in
    // the short one (0.5), one of eight in the long one (0.125).
    let drop_last = ScriptedRecognizer::degrading(|text| {
        let mut words: Vec<&str> = text.split_whitespace().collect();
        words.pop();
        words.join(" ")
    });
    let sentences = vec![
        "Hello there".to_string(),
        "one two three four five six seven eight".to_string(),
    ];
    let report = round_trip(&ScriptedSynthesizer, &drop_last, &speaker(), &sentences).unwrap();

    assert_eq!(report.items[0].word_error_rate, 0.5);
    assert_eq!(report.items[1].word_error_rate, 0.125);
    assert_eq!(
        report.corpus_word_error_rate(),
        Some(0.2),
        "2 errors in 10 words"
    );
    assert_eq!(report.mostly_lost(), 0);
}

#[test]
fn a_round_trip_of_nothing_has_no_rate() {
    let report = round_trip(
        &ScriptedSynthesizer,
        &ScriptedRecognizer::perfect(),
        &speaker(),
        &[],
    )
    .unwrap();
    assert_eq!(report.corpus_word_error_rate(), None);
}
