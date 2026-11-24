// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for synthetic voices that stay the
// same voice from one session to the next for its clients. If your team needs
// expertise in speech synthesis for conversational agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The seed of a persona's synthetic voice.
//!
//! A voice is an identity: the same sentence is spoken in very different
//! deliveries from different seeds, and takes, lessons and bundles made in one
//! voice mean nothing in another. The seed is therefore decided once, when the
//! first output of a run is made, recorded in `voice.json` beside the outputs,
//! and read back by every later command that writes to the same place. A
//! first run that is given no `--seed` draws one and prints it.
//!
//! A voice may instead imitate a recording (`--voice FILE`, with what it says
//! in `--voice-text`): the speaker then declares the `--portrayal` it is
//! given, because the theatrical label states that no recording exists. The
//! recording is named on every run; only the seed is recorded.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use clap::Args;
use splinter_sdk::seed::FrozenSeed;
use splinter_sdk::vocabulary::speech::{Portrayal, SpeakerProfile, VoiceReference};

/// The file that records a voice seed in an output directory.
const MANIFEST: &str = "voice.json";

/// Where the voice of the outputs in `dir` is recorded.
pub fn manifest_in(dir: &Path) -> PathBuf {
    dir.join(MANIFEST)
}

/// Where the voice of the output file `out` is recorded: beside it.
pub fn manifest_beside(out: &Path) -> PathBuf {
    out.with_extension("voice.json")
}

/// The seed of the voice recorded at `manifest`, else `given`, else a fresh
/// draw (printed). A `given` that differs from the recorded seed is refused:
/// it would describe another voice than the outputs already made.
pub fn seed(manifest: &Path, given: Option<u64>) -> Result<u64> {
    Ok(FrozenSeed::decide(manifest, "seed", given, None)?.record()?)
}

/// The persona's synthetic voice for the outputs recorded at `manifest`.
pub fn speaker(manifest: &Path, given: Option<u64>) -> Result<SpeakerProfile> {
    Ok(SpeakerProfile::new(
        seed(manifest, given)?,
        Portrayal::synthetic_theatrical(),
    ))
}

/// The `--seed` of a command that speaks in the persona's voice.
#[derive(Args)]
pub struct VoiceSeed {
    /// The seed the persona's synthetic voice is rendered from. The voice of
    /// an output directory (or of an output file, in `NAME.voice.json` beside
    /// it) is recorded there and read back: without it a first run draws a
    /// seed and prints it, and a seed that differs from the recorded one is
    /// refused.
    #[arg(long)]
    seed: Option<u64>,
    /// A recording of the voice to speak in (WAV, Ogg/Opus or WebM/Opus, a
    /// few seconds of one speaker). Needs `--portrayal`.
    #[arg(long, value_name = "FILE")]
    voice: Option<PathBuf>,
    /// What the `--voice` recording says: a synthesizer that reads it
    /// imitates the delivery as well as the timbre.
    #[arg(long, value_name = "TEXT", requires = "voice")]
    voice_text: Option<String>,
    /// What a voice imitating `--voice` declares itself to be, said wherever
    /// it is heard or recorded (for example "Synthetic speech in the voice of
    /// the person recorded, with their consent").
    #[arg(long, value_name = "LABEL", requires = "voice")]
    portrayal: Option<String>,
}

impl VoiceSeed {
    /// A voice for a command that writes no output place of its own: its
    /// report states the seed, drawn and printed when absent.
    ///
    /// # Errors
    /// `--voice` without a `--portrayal` that says what the voice is.
    pub fn unrecorded_speaker(&self) -> Result<SpeakerProfile> {
        self.dress(SpeakerProfile::new(
            splinter_sdk::seed::resolve("seed", self.seed),
            Portrayal::synthetic_theatrical(),
        ))
    }

    /// The voice recorded at `manifest`.
    ///
    /// # Errors
    /// The recorded seed differs from the one given, or the record cannot be
    /// read or written.
    pub fn speaker(&self, manifest: &Path) -> Result<SpeakerProfile> {
        self.dress(speaker(manifest, self.seed)?)
    }

    /// `speaker` imitating `--voice` under its `--portrayal`, when one is given.
    fn dress(&self, speaker: SpeakerProfile) -> Result<SpeakerProfile> {
        let Some(recording) = &self.voice else {
            return Ok(speaker);
        };
        let Some(label) = self.portrayal.as_deref() else {
            bail!(
                "--voice needs --portrayal: say what a voice imitating {} is wherever it is heard",
                recording.display()
            );
        };
        let portrayal = Portrayal::new(label).map_err(|e| anyhow::anyhow!("--portrayal: {e}"))?;
        Ok(SpeakerProfile::new(speaker.seed(), portrayal)
            .imitating(VoiceReference::new(recording, self.voice_text.clone())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_voice_drawn_for_the_first_output_is_read_back_by_the_next_and_another_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = manifest_in(dir.path());
        let drawn = seed(&manifest, None).unwrap();
        assert_eq!(seed(&manifest, None).unwrap(), drawn);
        let elsewhere = tempfile::tempdir().unwrap();
        assert_ne!(
            seed(&manifest_in(elsewhere.path()), None).unwrap(),
            drawn,
            "another output place draws its own voice"
        );
        assert_eq!(speaker(&manifest, None).unwrap().seed(), drawn);
        assert!(seed(&manifest, Some(drawn.wrapping_add(1))).is_err());
    }

    #[test]
    fn a_recorded_voice_needs_a_portrayal_and_keeps_the_seed() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = manifest_in(dir.path());
        let drawn = splinter_sdk::seed::resolve("seed", None);
        let args = |portrayal: Option<&str>| VoiceSeed {
            seed: Some(drawn),
            voice: Some("me.wav".into()),
            voice_text: Some("what I said".into()),
            portrayal: portrayal.map(str::to_string),
        };
        assert!(args(None).speaker(&manifest).is_err());
        let speaker = args(Some("Synthetic speech in my recorded voice"))
            .speaker(&manifest)
            .unwrap();
        assert_eq!(speaker.seed(), drawn);
        assert_eq!(
            speaker.portrayal().label(),
            "Synthetic speech in my recorded voice"
        );
        let voice = speaker.voice().unwrap();
        assert_eq!(
            (voice.recording(), voice.transcript()),
            (Path::new("me.wav"), Some("what I said"))
        );
    }

    #[test]
    fn a_given_seed_is_the_voice_of_a_new_output() {
        let dir = tempfile::tempdir().unwrap();
        let given = splinter_sdk::seed::resolve("seed", None);
        assert_eq!(seed(&manifest_in(dir.path()), Some(given)).unwrap(), given);
        assert_eq!(seed(&manifest_in(dir.path()), None).unwrap(), given);
    }
}
