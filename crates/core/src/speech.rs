// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements speech interfaces whose synthetic voices are
// always labelled as such, for its clients. If your team needs expertise in
// spoken dialogue systems, you can procure our services by sending an email to
// info@swedishembedded.com.

//! How a speaking persona is described: a speaker is never offered without
//! saying what it is.
//!
//! A persona speaks with a synthetic voice. No recording of Samuel Adams or
//! Thomas Jefferson exists, so no voice can be theirs: it is a theatrical
//! portrayal and must be labelled as one wherever it is heard or recorded. A
//! [`SpeakerProfile`] therefore cannot be built, or read from a file, without
//! a [`Portrayal`].
//!
//! A speaker may also imitate a recorded voice ([`VoiceReference`]): a few
//! seconds of someone speaking, and what they say. Its portrayal then says
//! whose voice it imitates; the synthetic-theatrical label, which states that
//! no recording exists, does not describe it.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The label every theatrical voice carries.
const THEATRICAL_LABEL: &str =
    "Synthetic theatrical portrayal; no recording of this person's voice exists";

/// A [`Portrayal`] label that is blank.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("a portrayal label must say what the voice is; it was blank")]
pub struct BlankPortrayal;

/// What a synthetic voice declares itself to be. Never blank.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Portrayal(String);

impl Portrayal {
    /// A portrayal stating `label`, which must not be blank.
    pub fn new(label: impl Into<String>) -> Result<Self, BlankPortrayal> {
        let label = label.into();
        if label.trim().is_empty() {
            return Err(BlankPortrayal);
        }
        Ok(Portrayal(label.trim().to_string()))
    }

    /// The label for a synthetic voice given to a person nobody recorded.
    #[must_use]
    pub fn synthetic_theatrical() -> Self {
        Portrayal(THEATRICAL_LABEL.to_string())
    }

    /// The words the voice declares itself with.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for Portrayal {
    type Error = BlankPortrayal;

    fn try_from(label: String) -> Result<Self, Self::Error> {
        Portrayal::new(label)
    }
}

impl From<Portrayal> for String {
    fn from(portrayal: Portrayal) -> String {
        portrayal.0
    }
}

/// A recorded voice a speaker imitates: the recording, and what it says when
/// that is known (a synthesizer that reads the transcript imitates delivery
/// and room as well as timbre).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoiceReference {
    recording: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    transcript: Option<String>,
}

impl VoiceReference {
    /// The voice of `recording`, whose words are `transcript` when given; a
    /// blank transcript counts as none.
    #[must_use]
    pub fn new(recording: impl Into<PathBuf>, transcript: Option<String>) -> Self {
        VoiceReference {
            recording: recording.into(),
            transcript: transcript.filter(|t| !t.trim().is_empty()),
        }
    }

    /// The recording to imitate.
    #[must_use]
    pub fn recording(&self) -> &Path {
        &self.recording
    }

    /// What the recording says, when known.
    #[must_use]
    pub fn transcript(&self) -> Option<&str> {
        self.transcript.as_deref()
    }
}

/// A synthetic speaker: the seed that renders it, the portrayal it declares
/// and, optionally, the recorded voice it imitates. The same seed with the
/// same synthesizer (and recording) renders the same voice.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeakerProfile {
    seed: u64,
    portrayal: Portrayal,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    voice: Option<VoiceReference>,
}

impl SpeakerProfile {
    /// A speaker rendered from `seed`, declaring `portrayal`.
    #[must_use]
    pub fn new(seed: u64, portrayal: Portrayal) -> Self {
        SpeakerProfile {
            seed,
            portrayal,
            voice: None,
        }
    }

    /// This speaker imitating the recorded `voice`.
    #[must_use]
    pub fn imitating(mut self, voice: VoiceReference) -> Self {
        self.voice = Some(voice);
        self
    }

    /// This speaker rendered from `seed` instead: the same portrayal and
    /// recorded voice, another take.
    #[must_use]
    pub fn with_seed(&self, seed: u64) -> Self {
        SpeakerProfile {
            seed,
            ..self.clone()
        }
    }

    /// The seed the voice is rendered from.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// What the voice declares itself to be.
    #[must_use]
    pub fn portrayal(&self) -> &Portrayal {
        &self.portrayal
    }

    /// The recorded voice it imitates, if any.
    #[must_use]
    pub fn voice(&self) -> Option<&VoiceReference> {
        self.voice.as_ref()
    }
}

/// The sentence a session opens with, so a listener never takes the voice for
/// the historical person's own.
#[must_use]
pub fn spoken_disclosure(persona: &str, speaker: &SpeakerProfile) -> String {
    format!(
        "You are speaking with a simulation of {}. {}.",
        persona.trim(),
        speaker.portrayal().label()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_portrayal_cannot_be_blank() {
        assert!(Portrayal::new("").is_err());
        assert!(Portrayal::new("  \t").is_err());
        assert!(Portrayal::new("A synthetic voice").is_ok());
    }

    #[test]
    fn the_theatrical_portrayal_says_no_recording_of_the_person_exists() {
        let label = Portrayal::synthetic_theatrical().label().to_lowercase();
        assert!(
            label.contains("synthetic") && label.contains("no recording"),
            "{label}"
        );
    }

    #[test]
    fn a_speaker_read_without_a_portrayal_is_refused() {
        assert!(serde_json::from_str::<SpeakerProfile>(r#"{"seed": 7}"#).is_err());
        assert!(
            serde_json::from_str::<SpeakerProfile>(r#"{"seed": 7, "portrayal": "  "}"#).is_err()
        );
        let ok: SpeakerProfile =
            serde_json::from_str(r#"{"seed": 7, "portrayal": "Synthetic"}"#).unwrap();
        assert_eq!(ok.seed(), 7);
    }

    #[test]
    fn a_speaker_round_trips_through_json() {
        let speaker = SpeakerProfile::new(3, Portrayal::synthetic_theatrical());
        let back: SpeakerProfile =
            serde_json::from_str(&serde_json::to_string(&speaker).unwrap()).unwrap();
        assert_eq!(back, speaker);
    }

    #[test]
    fn a_recorded_voice_survives_a_new_take_and_a_record_without_one_still_reads() {
        let voice = VoiceReference::new("me.wav", Some("what I said".into()));
        let speaker = SpeakerProfile::new(
            3,
            Portrayal::new("Synthetic speech in my own voice").unwrap(),
        )
        .imitating(voice.clone());
        let take = speaker.with_seed(4);
        assert_eq!(
            (take.seed(), take.voice(), take.portrayal()),
            (4, Some(&voice), speaker.portrayal())
        );
        let back: SpeakerProfile =
            serde_json::from_str(&serde_json::to_string(&speaker).unwrap()).unwrap();
        assert_eq!(back, speaker);
        let old: SpeakerProfile =
            serde_json::from_str(r#"{"seed": 7, "portrayal": "Synthetic"}"#).unwrap();
        assert_eq!(old.voice(), None);
        assert_eq!(
            VoiceReference::new("a.wav", Some("  ".into())).transcript(),
            None,
            "a blank transcript is none"
        );
    }

    #[test]
    fn the_spoken_disclosure_names_the_persona_and_the_portrayal() {
        let speaker = SpeakerProfile::new(1, Portrayal::synthetic_theatrical());
        let text = spoken_disclosure("Thomas Jefferson", &speaker);
        assert!(text.contains("Thomas Jefferson"), "{text}");
        assert!(text.contains(speaker.portrayal().label()), "{text}");
    }
}
