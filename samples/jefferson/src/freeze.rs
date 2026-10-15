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

/// The seed a directory is frozen with when none is given, and the one every
/// directory made before seed manifests was frozen with.
const DEFAULT_SEED: u64 = 1;

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
/// the default; refuses a `given` that differs from the recorded one. The
/// caller records it with [`FrozenSeed::record`] once its inputs are checked
/// and it is about to write.
pub fn seed(out: &Path, given: Option<u64>) -> Result<FrozenSeed> {
    let predates = OUTPUTS.iter().any(|name| out.join(name).exists());
    Ok(FrozenSeed::decide(
        &out.join(MANIFEST),
        given,
        DEFAULT_SEED,
        predates.then_some(DEFAULT_SEED),
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
