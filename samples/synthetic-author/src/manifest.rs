// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in proving which author, writer and
// settings a corpus was made with, you can procure our services by sending an
// email to info@swedishembedded.com.

//! What a corpus directory records about how it was made, and the author it
//! belongs to.
//!
//! The author is drawn from a seed once, when the corpus is begun. Every later
//! command that is given the directory reads the seed back from it and
//! refuses a different one, because grading a corpus against another author's
//! beliefs would give a wrong answer without a word.

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use splinter_sdk::seed::{resolve, SeedError};

/// The file in a corpus directory that records how the corpus was made.
pub const MANIFEST: &str = "manifest.json";
/// The file that names the author a corpus is written for. Corpora written
/// before manifests existed have this and no manifest.
const PERSONA: &str = "persona.json";

/// The file in a corpus directory that records how its decisions were read.
pub const MINE_RECORD: &str = "mine.json";

/// How the corpus's random draws were derived from the run seed, so that a
/// request's seed can be recomputed from the manifest.
pub const SEED_DERIVATION: &str =
    "splitmix64 of run_seed xor a salt per role, advanced by article and attempt";

/// Where the author seed came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SeedSource {
    /// Passed with `--seed`.
    Given,
    /// Drawn because no `--seed` was passed, and printed.
    Drawn,
}

/// The settings that decide what a corpus is. A corpus is only ever continued
/// with the same ones.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Recipe {
    /// The author's seed.
    pub author_seed: u64,
    /// The model that writes, as a model reference.
    pub writer: String,
    /// The model that reads each article, if there is one.
    pub reader: Option<String>,
    /// How the models decode.
    pub decoding: String,
    /// The seed the models' per-request seeds are derived from; `None` when
    /// every model is reached over a network and cannot be seeded.
    pub run_seed: Option<u64>,
    /// How many articles the corpus has.
    pub articles: usize,
    /// How many episodes each article tells.
    pub episodes_per_article: usize,
    /// The fewest words an article may have.
    pub min_words: usize,
}

/// The limits of the first run; later runs may spend more or less.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Limits {
    /// How many times an article is asked for before it is given up on.
    pub attempts: usize,
    /// The most requests the run may make.
    pub max_calls: usize,
    /// The most output tokens the run may write, estimated.
    pub max_output_tokens: u64,
    /// How many articles are written at a time.
    pub workers: usize,
}

/// How a corpus was made.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusManifest {
    /// Whether the author seed was given or drawn.
    pub seed_source: SeedSource,
    /// What decides the corpus.
    #[serde(flatten)]
    pub recipe: Recipe,
    /// How the per-request seeds follow from the run seed.
    pub seed_derivation: String,
    /// The limits of the run that began the corpus.
    pub limits: Limits,
    /// The revision of the code that began it.
    pub code_revision: String,
}

/// The revision of the code this program was built from.
#[must_use]
pub fn code_revision() -> &'static str {
    env!("SYNTHETIC_AUTHOR_REVISION")
}

fn path_of(dir: &Path) -> PathBuf {
    dir.join(MANIFEST)
}

/// The manifest of the corpus in `dir`, if it has one.
///
/// # Errors
/// The file exists and cannot be read or is not a manifest.
pub fn read(dir: &Path) -> anyhow::Result<Option<CorpusManifest>> {
    let path = path_of(dir);
    match std::fs::read_to_string(&path) {
        Ok(text) => Ok(Some(
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?,
        )),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

/// The author the corpus in `dir` was written for, with the file that says so
/// and how it came to be recorded (`recorded` by a manifest, `legacy` by the
/// persona file of a corpus that predates manifests).
fn recorded_author(dir: &Path) -> anyhow::Result<Option<(PathBuf, u64, &'static str)>> {
    if let Some(manifest) = read(dir)? {
        return Ok(Some((
            path_of(dir),
            manifest.recipe.author_seed,
            "recorded",
        )));
    }
    let path = dir.join(PERSONA);
    match std::fs::read_to_string(&path) {
        Ok(text) => {
            let doc: serde_json::Value = serde_json::from_str(&text)
                .with_context(|| format!("parsing {}", path.display()))?;
            let seed = doc["seed"]
                .as_u64()
                .with_context(|| format!("{}: no seed", path.display()))?;
            Ok(Some((path, seed, "legacy")))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

fn agree(
    file: PathBuf,
    recorded: u64,
    origin: &'static str,
    given: Option<u64>,
) -> anyhow::Result<u64> {
    match given {
        Some(given) if given != recorded => Err(SeedError::Differs {
            manifest: file,
            recorded,
            origin,
            given,
        }
        .into()),
        _ => Ok(recorded),
    }
}

/// The author seed for a command that works on the corpus in `dir`: the one
/// the corpus records, which `given` must equal when passed. A directory that
/// records no author (or does not exist) takes `given`.
///
/// # Errors
/// `given` differs from the recorded author, nothing is recorded and nothing
/// was given, or the record cannot be read.
pub fn author_seed(dir: &Path, given: Option<u64>) -> anyhow::Result<u64> {
    match recorded_author(dir)? {
        Some((file, recorded, origin)) => agree(file, recorded, origin, given),
        None => given.with_context(|| format!("{} records no author: pass --seed", dir.display())),
    }
}

/// The author seed for a command that shows an author: the one the corpus in
/// `cache` records when there is a corpus (which `given` must equal when
/// passed); otherwise `given`, and otherwise a fresh seed, printed so passing
/// it back reproduces the author.
///
/// # Errors
/// As [`author_seed`] when `cache` is given.
pub fn author_seed_or_draw(cache: Option<&Path>, given: Option<u64>) -> anyhow::Result<u64> {
    match cache {
        Some(dir) => author_seed(dir, given),
        None => Ok(resolve("seed", given)),
    }
}

/// The author seed for beginning or continuing the corpus in `dir`: as
/// [`author_seed`], and when nothing is recorded and none is given, a fresh
/// one, printed so passing it back reproduces the author.
///
/// # Errors
/// As [`author_seed`].
pub fn corpus_author_seed(dir: &Path, given: Option<u64>) -> anyhow::Result<(u64, SeedSource)> {
    match recorded_author(dir)? {
        Some((file, recorded, origin)) => {
            Ok((agree(file, recorded, origin, given)?, SeedSource::Given))
        }
        None => Ok(match given {
            Some(seed) => (seed, SeedSource::Given),
            None => (resolve("seed", None), SeedSource::Drawn),
        }),
    }
}

/// The run seed for the corpus in `dir`: the one its manifest records, which
/// `given` must equal when passed; else `given`; else a fresh one, printed.
/// `None` when no model of the corpus can be seeded.
///
/// # Errors
/// `given` differs from the recorded run seed, or the manifest cannot be read.
pub fn run_seed(dir: &Path, given: Option<u64>, seedable: bool) -> anyhow::Result<Option<u64>> {
    if let Some(manifest) = read(dir)? {
        let recorded = manifest.recipe.run_seed;
        if let (Some(recorded), Some(given)) = (recorded, given) {
            agree(path_of(dir), recorded, "recorded", Some(given))?;
        }
        return Ok(recorded);
    }
    Ok(seedable.then(|| resolve("run-seed", given)))
}

/// Makes `manifest` the record of the corpus in `dir`. A corpus that has a
/// manifest is only continued with the same recipe; the manifest it has is
/// kept as it is, with the limits of the run that began it.
///
/// # Errors
/// The recipe differs from the recorded one (the differing fields are named),
/// or the manifest cannot be read or written.
pub fn begin_or_continue(dir: &Path, manifest: CorpusManifest) -> anyhow::Result<CorpusManifest> {
    let path = path_of(dir);
    if let Some(kept) = read(dir)? {
        let differing = kept.recipe.differences(&manifest.recipe);
        anyhow::ensure!(
            differing.is_empty(),
            "{}: the corpus was begun with another recipe (differs in {}); a corpus is only \
             continued with the settings it was begun with, use another cache directory",
            path.display(),
            differing.join(", "),
        );
        return Ok(kept);
    }
    std::fs::create_dir_all(dir)?;
    let part = path.with_extension("part");
    std::fs::write(&part, serde_json::to_vec_pretty(&manifest)?)
        .with_context(|| format!("writing {}", part.display()))?;
    std::fs::rename(&part, &path).with_context(|| format!("writing {}", path.display()))?;
    Ok(manifest)
}

impl Recipe {
    fn differences(&self, other: &Self) -> Vec<&'static str> {
        let mut named = Vec::new();
        let mut note = |differs: bool, name| {
            if differs {
                named.push(name);
            }
        };
        note(self.author_seed != other.author_seed, "author seed");
        note(self.writer != other.writer, "writer");
        note(self.reader != other.reader, "reader");
        note(self.decoding != other.decoding, "decoding");
        note(self.run_seed != other.run_seed, "run seed");
        note(self.articles != other.articles, "articles");
        note(
            self.episodes_per_article != other.episodes_per_article,
            "episodes per article",
        );
        note(self.min_words != other.min_words, "min words");
        named
    }
}

/// How the decisions of a corpus are read out of its articles: what a later
/// `mine` must continue with, as a reader of another kind or another seed
/// would make decisions that are not comparable with the ones already read.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MineRecord {
    /// The model that reads, as a model reference.
    pub miner: String,
    /// The seed the reader's per-request seeds are derived from; `None` when
    /// the reader is reached over a network and cannot be seeded.
    pub run_seed: Option<u64>,
    /// How the per-request seeds follow from the run seed.
    pub seed_derivation: String,
}

/// The record of how the decisions of the corpus in `dir` are read: the one
/// kept when there is one, which `miner` and `given` must agree with; else a
/// new one, with a fresh run seed (printed) when none is given and the reader
/// `seedable`, written to the directory.
///
/// # Errors
/// The corpus was mined with another reader or another run seed, or the
/// record cannot be read or written.
pub fn begin_or_continue_mining(
    dir: &Path,
    miner: &str,
    given: Option<u64>,
    seedable: bool,
) -> anyhow::Result<MineRecord> {
    let path = dir.join(MINE_RECORD);
    match std::fs::read_to_string(&path) {
        Ok(text) => {
            let kept: MineRecord = serde_json::from_str(&text)
                .with_context(|| format!("parsing {}", path.display()))?;
            anyhow::ensure!(
                kept.miner == miner,
                "{}: the decisions were read by {}, not {miner}; use another cache directory",
                path.display(),
                kept.miner,
            );
            if let (Some(recorded), Some(_)) = (kept.run_seed, given) {
                agree(path, recorded, "recorded", given)?;
            }
            Ok(kept)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let record = MineRecord {
                miner: miner.to_string(),
                run_seed: seedable.then(|| resolve("run-seed", given)),
                seed_derivation: SEED_DERIVATION.to_string(),
            };
            std::fs::create_dir_all(dir)?;
            let part = path.with_extension("part");
            std::fs::write(&part, serde_json::to_vec_pretty(&record)?)
                .with_context(|| format!("writing {}", part.display()))?;
            std::fs::rename(&part, &path).with_context(|| format!("writing {}", path.display()))?;
            Ok(record)
        }
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}
