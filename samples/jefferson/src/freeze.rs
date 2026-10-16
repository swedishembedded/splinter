// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements measured knowledge-injection samples for
// language models for its clients. If your team needs expertise in teaching a
// model a document corpus and proving what it kept, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The seed an output directory was frozen with.
//!
//! The corpus split, the exam, the scenarios and the materials in one
//! directory are all functions of one seed, and the exam in it is pinned by
//! content. The seed is recorded beside them the first time any command writes
//! there, and every later command reads it back, so a command that is given no
//! seed continues the directory's and one that is given another is refused.

use std::path::Path;

use anyhow::Result;
use splinter_sdk::seed::FrozenSeed;

/// The seed every directory made before seed manifests was frozen with. A new
/// directory is given a drawn one.
const LEGACY_SEED: u64 = 1;

/// The manifest, beside the files it describes.
const MANIFEST: &str = "SEED.json";

/// What the commands write into an output directory; one of these existing
/// without a manifest means the directory predates manifests.
const OUTPUTS: [&str; 7] = [
    "letters.jsonl",
    "sft.jsonl",
    "exam.jsonl",
    "scenarios.jsonl",
    "sft-scenarios.jsonl",
    "letters",
    "works",
];

/// The seed of the artifacts in `out`: the recorded one, else `given`, else
/// a fresh draw (printed, so it can be passed back); refuses a `given` that
/// differs from the recorded one. The caller records it with
/// [`FrozenSeed::record`] once its inputs are checked and it is about to write.
pub fn seed(out: &Path, given: Option<u64>) -> Result<FrozenSeed> {
    let predates = OUTPUTS.iter().any(|name| out.join(name).exists());
    Ok(FrozenSeed::decide(
        &out.join(MANIFEST),
        "seed",
        given,
        predates.then_some(LEGACY_SEED),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recorded(out: &Path, given: Option<u64>) -> u64 {
        seed(out, given).unwrap().record().unwrap()
    }

    #[test]
    fn a_second_command_without_a_seed_continues_the_directorys() {
        let dir = tempfile::tempdir().unwrap();
        // seed: any value but the default shows the recorded one is read back
        assert_eq!(recorded(dir.path(), Some(9)), 9);
        assert_eq!(recorded(dir.path(), None), 9);
    }

    #[test]
    fn a_new_directory_without_a_seed_records_a_drawn_one() {
        let dir = tempfile::tempdir().unwrap();
        let drawn = recorded(dir.path(), None);
        assert_ne!(drawn, LEGACY_SEED);
        let manifest = std::fs::read_to_string(dir.path().join(MANIFEST)).unwrap();
        assert!(manifest.contains("recorded"), "{manifest}");
        assert_eq!(recorded(dir.path(), None), drawn);
        assert!(seed(dir.path(), Some(drawn ^ 1)).is_err());
    }

    #[test]
    fn a_different_explicit_seed_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        recorded(dir.path(), Some(9));
        assert!(seed(dir.path(), Some(10)).is_err());
    }

    #[test]
    fn a_directory_made_before_manifests_is_recorded_as_seed_one() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("exam.jsonl"), "{}\n").unwrap();
        assert_eq!(recorded(dir.path(), None), 1);
        let manifest = std::fs::read_to_string(dir.path().join(MANIFEST)).unwrap();
        assert!(manifest.contains("legacy"), "{manifest}");
        assert!(seed(dir.path(), Some(2)).is_err());
    }
}
