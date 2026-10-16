// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for reproducible, auditable
// experiments on frozen data for its clients. If your team needs expertise in
// experiment provenance or held-out evaluation design, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Seeds for the programs built on the SDK.
//!
//! [`resolve`] gives a random draw its seed: the one the caller passed, else a
//! fresh one that is printed so the run can be reproduced. A frozen artifact
//! is different: its seed belongs to the artifact, see [`frozen`].
//!
//! # The seed of a frozen artifact
//!
//! A frozen exam, split or partition is a function of a seed. The seed is part
//! of the artifact, so it is recorded beside it the first time the artifact is
//! made and every later command reads it back: a rerun that offers no seed
//! gets the recorded one, and one that offers a different seed is refused
//! rather than silently describing another artifact.

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub use splinter_core::rng::Seed;

/// The seed of the random draw called `name` (the flag that sets it, without
/// its dashes): `given` when the caller passed one, otherwise a fresh seed,
/// which is printed to stderr so passing it back reproduces the draw.
#[must_use]
pub fn resolve(name: &str, given: Option<u64>) -> u64 {
    if let Some(seed) = given {
        return seed;
    }
    let seed = splinter_model::seed::draw();
    eprintln!("no --{name} given, using seed {seed} (pass --{name} {seed} to reproduce)");
    seed.value()
}

/// Why a frozen seed could not be resolved.
#[derive(Debug, thiserror::Error)]
pub enum SeedError {
    /// The artifact was made with another seed than the one asked for.
    #[error(
        "{manifest}: the artifact was frozen with seed {recorded} ({origin}) and --seed {given} \
         would make another; a frozen artifact is never regenerated, use another directory"
    )]
    Differs {
        /// The seed manifest that holds the recorded seed.
        manifest: PathBuf,
        /// The seed the artifact was made with.
        recorded: u64,
        /// How it came to be recorded: `recorded` or `legacy`.
        origin: &'static str,
        /// The seed the caller asked for.
        given: u64,
    },
    /// The manifest exists but does not say a seed.
    #[error("{manifest}: not a seed manifest: {reason}")]
    Malformed {
        /// The manifest that could not be read.
        manifest: PathBuf,
        /// What is wrong with it.
        reason: String,
    },
    /// The manifest could not be read or written.
    #[error("{manifest}: {source}")]
    Io {
        /// The manifest involved.
        manifest: PathBuf,
        /// The underlying failure.
        source: std::io::Error,
    },
}

/// How a manifest's seed came to be recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Origin {
    /// Recorded when the artifact was made.
    Recorded,
    /// The artifact predates manifests; this is the seed that was the default
    /// when it was made.
    Legacy,
}

impl Origin {
    fn label(self) -> &'static str {
        match self {
            Origin::Recorded => "recorded",
            Origin::Legacy => "legacy",
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    seed: u64,
    origin: Origin,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    note: Option<String>,
}

/// A frozen artifact's seed, decided but possibly not yet recorded.
///
/// Deciding the seed is separate from recording it so a command can check its
/// inputs between the two: a freeze that fails before it writes anything must
/// not leave a manifest behind that forbids the next, corrected attempt.
#[derive(Debug)]
#[must_use = "a decided seed that is never recorded leaves the artifact without a manifest"]
pub struct FrozenSeed {
    seed: u64,
    unrecorded: Option<(PathBuf, Manifest)>,
}

impl FrozenSeed {
    /// Decide the seed of the frozen artifact whose seed manifest is at
    /// `manifest`.
    ///
    /// * A manifest exists: its seed is the answer; `given` must be absent or
    ///   equal to it, otherwise [`SeedError::Differs`].
    /// * No manifest, `legacy` is `Some(seed)`: the artifact was made before
    ///   manifests existed with that seed. `given` must be absent or equal to
    ///   it; [`record`](Self::record) writes the manifest once, saying the seed
    ///   is a legacy one.
    /// * No manifest, `legacy` is `None`: the artifact is being made now; its
    ///   seed is `given`, else a fresh draw printed as the one `name` (the
    ///   flag, without its dashes) used, see [`resolve`]. Only this case
    ///   draws, so a read-back says nothing about a seed.
    pub fn decide(
        manifest: &Path,
        name: &str,
        given: Option<u64>,
        legacy: Option<u64>,
    ) -> Result<Self, SeedError> {
        Self::decide_with(manifest, given, legacy, || resolve(name, None))
    }

    /// [`decide`](Self::decide) with the source of a fresh seed given by the
    /// caller; `draw` runs only for an artifact being made now with no seed
    /// given.
    pub fn decide_with(
        manifest: &Path,
        given: Option<u64>,
        legacy: Option<u64>,
        draw: impl FnOnce() -> u64,
    ) -> Result<Self, SeedError> {
        match std::fs::read_to_string(manifest) {
            Ok(text) => {
                let recorded: Manifest =
                    serde_json::from_str(&text).map_err(|e| SeedError::Malformed {
                        manifest: manifest.to_path_buf(),
                        reason: e.to_string(),
                    })?;
                let seed = agree(manifest, recorded.seed, recorded.origin, given)?;
                Ok(Self {
                    seed,
                    unrecorded: None,
                })
            }
            Err(e) if e.kind() == ErrorKind::NotFound => {
                let (seed, origin, note) = match legacy {
                    Some(seed) => (seed, Origin::Legacy, Some(LEGACY_NOTE.to_string())),
                    None => (given.unwrap_or_else(draw), Origin::Recorded, None),
                };
                agree(manifest, seed, origin, given)?;
                Ok(Self {
                    seed,
                    unrecorded: Some((manifest.to_path_buf(), Manifest { seed, origin, note })),
                })
            }
            Err(source) => Err(SeedError::Io {
                manifest: manifest.to_path_buf(),
                source,
            }),
        }
    }

    /// The decided seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Write the manifest if the artifact has none, and return the seed. The
    /// manifest is never rewritten, so a freeze's seed cannot change after the
    /// fact. Two processes freezing at once are not coordinated: freezing is a
    /// single-operator step.
    pub fn record(self) -> Result<u64, SeedError> {
        if let Some((path, manifest)) = &self.unrecorded {
            let io = |source| SeedError::Io {
                manifest: path.clone(),
                source,
            };
            let body =
                serde_json::to_string_pretty(manifest).map_err(|e| io(std::io::Error::other(e)))?;
            write_once(path, &body).map_err(io)?;
            if manifest.origin == Origin::Legacy {
                eprintln!(
                    "{}: frozen before seed manifests existed; recorded as seed {}",
                    path.display(),
                    self.seed
                );
            }
        }
        Ok(self.seed)
    }
}

const LEGACY_NOTE: &str =
    "the artifact predates seed manifests; this is the seed that was the default when it was made";

/// Decide and record in one step: [`FrozenSeed::decide`] then
/// [`FrozenSeed::record`], for a command with nothing left to check between.
pub fn frozen(
    manifest: &Path,
    name: &str,
    given: Option<u64>,
    legacy: Option<u64>,
) -> Result<u64, SeedError> {
    FrozenSeed::decide(manifest, name, given, legacy)?.record()
}

fn agree(
    manifest: &Path,
    recorded: u64,
    origin: Origin,
    given: Option<u64>,
) -> Result<u64, SeedError> {
    match given {
        Some(given) if given != recorded => Err(SeedError::Differs {
            manifest: manifest.to_path_buf(),
            recorded,
            origin: origin.label(),
            given,
        }),
        _ => Ok(recorded),
    }
}

/// Write through a temporary file so a reader never sees half a manifest.
fn write_once(path: &Path, body: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let part = path.with_extension("part");
    std::fs::write(&part, format!("{body}\n"))?;
    std::fs::rename(&part, path)
}
