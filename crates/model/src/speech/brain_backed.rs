// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements spoken dialogue systems that reach their
// speech models only through a serving API, for its clients. If your team
// needs expertise in recognition and synthesis behind a shared model
// service, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Recognition and synthesis as requests to a brain server.
//!
//! * Recognition is `POST {url}/audio/transcriptions` with OpenRouter's JSON
//!   body, `{"model", "input_audio": {"data": <base64 WAV>, "format": "wav"}}`;
//!   the answer is `{"text", ...}`. The server refuses a recording longer
//!   than it takes (300 seconds) instead of cutting it.
//! * Synthesis is `POST {url}/audio/speech` with `{"model", "input",
//!   "response_format": "wav", "seed"}` and, for a speaker that imitates a
//!   recorded voice, `"input_references"`: the recording as an `input_audio`
//!   part and its transcript as a `text` part. The answer is the speech
//!   itself, a 16-bit mono WAV at the model's own rate, with the seed it was
//!   drawn with in `X-Seed`.
//!
//! The model is named by the id the server lists it under (`GET /v1/models`),
//! such as `Qwen/Qwen3-TTS-12Hz-0.6B-Base`. Every request for one model goes to
//! the same endpoint ([`BrainEndpoints::pin`] of its id), so a server keeps it
//! resident. The calls block, on a thread of their own, so a caller inside an
//! async runtime may make them.
//!
//! A [`SpeakerProfile`]'s seed chooses the take: the same seed, text and
//! recording give the same clip. A server that answers with another seed than
//! the one asked for is refused, so no run believes a seed chose a take it did
//! not. A model that cannot take what the speaker asks (a reference voice on a
//! model that does not clone, say) is refused by the server with the field
//! named, never spoken in another voice.

use std::path::Path;
use std::time::Duration;

use base64::Engine as _;
use serde_json::json;
use splinter_core::speech::{SpeakerProfile, VoiceReference};

use super::{Clip, Recognizer, Synthesizer, Transcription};
use crate::endpoint::{post_for_bytes, post_json, BrainEndpoints};
use crate::error::PolicyError;

/// The recognizer used where none is named: a small streaming model, which
/// brain also uses to judge its own synthesis.
pub const DEFAULT_RECOGNIZER: &str = "nvidia/nemotron-3.5-asr-streaming-0.6b";

/// The synthesizer used where none is named: the smallest Qwen3-TTS.
pub const DEFAULT_SYNTHESIZER: &str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base";

/// How long one recognition or synthesis may take, a server loading the
/// model on first use included.
const SPEECH_TIMEOUT: Duration = Duration::from_secs(600);

/// The largest recording a speaker may imitate, read whole into the request.
/// The server takes at most 30 seconds of it; this bounds the read before
/// that, at a minute of 48 kHz stereo 16-bit audio.
const MAX_REFERENCE_BYTES: u64 = 12 * 1024 * 1024;

/// One endpoint of `endpoints`, the one requests for `model` are pinned to.
fn endpoint_for(endpoints: &BrainEndpoints, model: &str) -> String {
    endpoints.urls()[endpoints.pin(model.as_bytes())].clone()
}

/// A recognition model a brain server serves.
pub struct BrainRecognizer {
    url: String,
    key: String,
    model: String,
}

impl BrainRecognizer {
    /// The recognition model the servers of `endpoints` list as `model`.
    #[must_use]
    pub fn new(endpoints: &BrainEndpoints, model: &str) -> Self {
        Self {
            url: endpoint_for(endpoints, model),
            key: endpoints.key().to_string(),
            model: model.to_string(),
        }
    }
}

impl Recognizer for BrainRecognizer {
    fn transcribe(&self, clip: &Clip) -> Result<Transcription, PolicyError> {
        let failed = |reason: String| PolicyError::Transcription {
            reason: format!("{} at {}: {reason}", self.model, self.url),
        };
        let body = json!({
            "model": self.model,
            "input_audio": {
                "data": base64::engine::general_purpose::STANDARD.encode(clip.to_wav()),
                "format": "wav",
            },
        });
        let reply = post_json(
            &format!("{}/audio/transcriptions", self.url),
            &self.key,
            &body,
            SPEECH_TIMEOUT,
        )
        .map_err(failed)?;
        let text = reply["text"]
            .as_str()
            .ok_or_else(|| failed(format!("the answer has no \"text\": {reply}")))?;
        Ok(Transcription {
            text: text.trim().to_string(),
        })
    }
}

/// A synthesis model a brain server serves.
pub struct BrainSynthesizer {
    url: String,
    key: String,
    model: String,
}

impl BrainSynthesizer {
    /// The synthesis model the servers of `endpoints` list as `model`.
    #[must_use]
    pub fn new(endpoints: &BrainEndpoints, model: &str) -> Self {
        Self {
            url: endpoint_for(endpoints, model),
            key: endpoints.key().to_string(),
            model: model.to_string(),
        }
    }
}

impl Synthesizer for BrainSynthesizer {
    fn speak(&self, text: &str, speaker: &SpeakerProfile) -> Result<Clip, PolicyError> {
        let failed = |reason: String| PolicyError::Synthesis {
            reason: format!("{} at {}: {reason}", self.model, self.url),
        };
        let mut body = json!({"model": self.model, "input": text, "response_format": "wav", "seed": speaker.seed()});
        if let Some(voice) = speaker.voice() {
            body["input_references"] = reference_parts(voice).map_err(failed)?;
        }
        let reply = post_for_bytes(
            &format!("{}/audio/speech", self.url),
            &self.key,
            &body,
            SPEECH_TIMEOUT,
        )
        .map_err(failed)?;
        let clip = Clip::from_wav(&reply.body).map_err(|e| failed(e.to_string()))?;
        // The rate is stated twice; a disagreement means the audio is not what
        // the server says it made.
        if let Some(stated) = reply
            .headers
            .get("x-sample-rate")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<u32>().ok())
        {
            if stated != clip.sample_rate() {
                return Err(failed(format!(
                    "X-Sample-Rate says {stated} Hz, the WAV header {} Hz",
                    clip.sample_rate()
                )));
            }
        }
        let drawn = reply
            .headers
            .get("x-seed")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<u64>().ok());
        if drawn != Some(speaker.seed()) {
            return Err(failed(format!(
                "asked for seed {}, the server says it drew {}",
                speaker.seed(),
                drawn.map_or_else(|| "none".to_string(), |d| d.to_string())
            )));
        }
        Ok(clip)
    }
}

/// The `input_references` of a speaker imitating `voice`: the recording as an
/// `input_audio` part, and its transcript as a `text` part when known.
fn reference_parts(voice: &VoiceReference) -> Result<serde_json::Value, String> {
    let recording = read_recording(voice.recording())?;
    let mut parts = vec![json!({
        "type": "input_audio",
        "input_audio": {"data": base64::engine::general_purpose::STANDARD.encode(recording)},
    })];
    if let Some(text) = voice.transcript() {
        parts.push(json!({"type": "text", "text": text}));
    }
    Ok(serde_json::Value::Array(parts))
}

fn read_recording(path: &Path) -> Result<Vec<u8>, String> {
    let shown = path.display();
    let size = std::fs::metadata(path)
        .map_err(|e| format!("the voice recording {shown}: {e}"))?
        .len();
    if size > MAX_REFERENCE_BYTES {
        return Err(format!("the voice recording {shown} is {size} bytes, more than the {MAX_REFERENCE_BYTES} a speaker may imitate"));
    }
    std::fs::read(path).map_err(|e| format!("the voice recording {shown}: {e}"))
}
