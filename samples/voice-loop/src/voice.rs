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

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Args;
use splinter_sdk::seed::FrozenSeed;
use splinter_sdk::vocabulary::speech::{Portrayal, SpeakerProfile};

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
}

impl VoiceSeed {
    /// A voice for a command that writes no output place of its own: its
    /// report states the seed, drawn and printed when absent.
    #[must_use]
    pub fn unrecorded_speaker(&self) -> SpeakerProfile {
        SpeakerProfile::new(
            splinter_sdk::seed::resolve("seed", self.seed),
            Portrayal::synthetic_theatrical(),
        )
    }

    /// The voice recorded at `manifest`.
    ///
    /// # Errors
    /// The recorded seed differs from the one given, or the record cannot be
    /// read or written.
    pub fn speaker(&self, manifest: &Path) -> Result<SpeakerProfile> {
        speaker(manifest, self.seed)
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
    fn a_given_seed_is_the_voice_of_a_new_output() {
        let dir = tempfile::tempdir().unwrap();
        let given = splinter_sdk::seed::resolve("seed", None);
        assert_eq!(seed(&manifest_in(dir.path()), Some(given)).unwrap(), given);
        assert_eq!(seed(&manifest_in(dir.path()), None).unwrap(), given);
    }
}
