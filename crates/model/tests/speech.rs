// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements spoken dialogue systems that reach their
// speech models only through a serving API, for its clients. If your team
// needs expertise in recognition and synthesis behind a shared model service,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: recognition and synthesis are requests to a brain server. The
//! recognizer posts the clip as a base64 WAV to `/audio/transcriptions` under
//! the model it names; the synthesizer posts `{"model", "input",
//! "response_format": "wav"}` to `/audio/speech`, sending nothing the route
//! refuses, and hands back the WAV's samples at the server's rate. A model the
//! server does not serve is an error naming it, never an empty transcript or
//! a silent clip. Over the scripted server a round trip loses nothing.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use splinter_core::speech::{Portrayal, SpeakerProfile};
use splinter_model::endpoint::BrainEndpoints;
use splinter_model::scripted_endpoint::{Card, ScriptedEndpoint, SCRIPTED_KEY};
use splinter_model::speech::scripted::clip_of;
use splinter_model::speech::{
    round_trip, BrainRecognizer, BrainSynthesizer, Recognizer, Synthesizer,
};
use splinter_model::PolicyError;

const ASR: &str = "nvidia/asr";
const TTS: &str = "Qwen/tts";

fn served() -> (ScriptedEndpoint, BrainEndpoints) {
    let server = ScriptedEndpoint::start(vec![Card::base(ASR), Card::base(TTS)]);
    let endpoints = BrainEndpoints::new(vec![server.url()], SCRIPTED_KEY).unwrap();
    (server, endpoints)
}

fn speaker() -> SpeakerProfile {
    // seed: any voice; the served route takes none, so its value is irrelevant.
    SpeakerProfile::new(7, Portrayal::synthetic_theatrical())
}

#[test]
fn the_recognizer_sends_the_clip_as_a_wav_under_its_model_and_returns_the_words() {
    let (server, endpoints) = served();
    let heard = BrainRecognizer::new(&endpoints, ASR)
        .transcribe(&clip_of("  Liberty is not given.  "))
        .unwrap();
    assert_eq!(heard.text, "Liberty is not given.", "trimmed");
    let asked = server.speech_asked();
    assert_eq!(asked.len(), 1);
    let (route, body) = &asked[0];
    assert_eq!(route, "transcriptions");
    assert_eq!(body["model"], ASR);
    assert_eq!(body["input_audio"]["format"], "wav");
}

#[test]
fn the_synthesizer_asks_only_for_what_the_route_takes_and_returns_the_servers_audio() {
    let (server, endpoints) = served();
    let clip = BrainSynthesizer::new(&endpoints, TTS)
        .speak("Resist.", &speaker())
        .unwrap();
    let expected = clip_of("Resist.");
    assert_eq!(clip.sample_rate(), expected.sample_rate());
    assert_eq!(clip.samples().len(), expected.samples().len());
    let asked = server.speech_asked();
    assert_eq!(
        asked,
        vec![(
            "speech".to_string(),
            serde_json::json!({"model": TTS, "input": "Resist.", "response_format": "wav"})
        )]
    );
}

#[test]
fn a_model_the_server_does_not_serve_is_an_error_naming_it() {
    let (_server, endpoints) = served();
    let err = BrainRecognizer::new(&endpoints, "missing/asr")
        .transcribe(&clip_of("x"))
        .unwrap_err();
    assert!(matches!(err, PolicyError::Transcription { .. }), "{err}");
    assert!(err.to_string().contains("missing/asr"), "{err}");
    let err = BrainSynthesizer::new(&endpoints, "missing/tts")
        .speak("x", &speaker())
        .unwrap_err();
    assert!(matches!(err, PolicyError::Synthesis { .. }), "{err}");
    assert!(
        err.to_string().contains("unknown model missing/tts"),
        "{err}"
    );
}

#[test]
fn a_round_trip_through_the_server_loses_nothing_when_hearing_is_perfect() {
    let (_server, endpoints) = served();
    let sentences = vec![
        "We resolved.".to_string(),
        "Then we petitioned, and they did not listen.".to_string(),
    ];
    let report = round_trip(
        &BrainSynthesizer::new(&endpoints, TTS),
        &BrainRecognizer::new(&endpoints, ASR),
        &speaker(),
        &sentences,
    )
    .unwrap();
    assert_eq!(report.corpus_word_error_rate(), Some(0.0));
    assert_eq!(report.items[1].heard, sentences[1]);
}
