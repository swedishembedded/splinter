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

use crate::family::{check_distinct, Roles};
use crate::persona::Persona;

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

/// Whose beliefs a corpus is written for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum PersonaSource {
    /// The author the author seed draws.
    #[default]
    Drawn,
    /// The author who holds every belief of the one `seed` draws the other
    /// way, in the world of that seed.
    OppositeOf {
        /// The seed of the author this one is the opposite of; it is also the
        /// corpus's author seed, which names the world.
        seed: u64,
    },
}

impl PersonaSource {
    // By reference because serde's `skip_serializing_if` hands it one.
    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn is_drawn(&self) -> bool {
        *self == Self::Drawn
    }
}

impl std::fmt::Display for PersonaSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Drawn => write!(f, "an author drawn from the seed"),
            Self::OppositeOf { seed } => write!(f, "the opposite of the author of seed {seed}"),
        }
    }
}

/// The settings that decide what a corpus is. A corpus is only ever continued
/// with the same ones.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Recipe {
    /// The seed that names the author's world: the situations, the
    /// combinations withheld and the examination. For the opposite of an
    /// author it is that author's seed.
    pub author_seed: u64,
    /// Whose beliefs the corpus is written for.
    #[serde(default, skip_serializing_if = "PersonaSource::is_drawn")]
    pub persona_source: PersonaSource,
    /// The model that writes, as a model reference.
    pub writer: String,
    /// The model that reads each article, if there is one.
    pub reader: Option<String>,
    /// Whether the check that the writer and the reader are of different model
    /// families was set aside.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub allow_same_family: bool,
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

/// An author and the world it is set in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Author {
    /// The beliefs.
    pub persona: Persona,
    /// The seed that names the world: what the situations, the withheld
    /// combinations and the examination are drawn from.
    pub seed: u64,
    /// Whose beliefs they are.
    pub persona_source: PersonaSource,
    /// Whether the seed was given or drawn.
    pub seed_source: SeedSource,
}

/// The persona a world seed and a source name.
fn persona_for(seed: u64, source: PersonaSource) -> anyhow::Result<Persona> {
    match source {
        PersonaSource::Drawn => Ok(Persona::generate(seed)),
        PersonaSource::OppositeOf { seed: of } => {
            anyhow::ensure!(
                of == seed,
                "the corpus is the opposite of the author of seed {of} but its author seed is {seed}"
            );
            Persona::opposite_of(of).with_context(|| {
                format!("the opposite of the author of seed {of} cannot be checked against")
            })
        }
    }
}

/// The author the corpus in `dir` was written for, read back from it: the one
/// its manifest names, which `given` must agree with when passed. A corpus
/// from before manifests is a drawn author.
///
/// # Errors
/// As [`author_seed`], or the recorded author cannot be had.
pub fn author(dir: &Path, given: Option<u64>) -> anyhow::Result<Author> {
    let seed = author_seed(dir, given)?;
    let (source, seed_source) = match read(dir)? {
        Some(m) => (m.recipe.persona_source, m.seed_source),
        None => (PersonaSource::Drawn, SeedSource::Given),
    };
    Ok(Author {
        persona: persona_for(seed, source)?,
        seed,
        persona_source: source,
        seed_source,
    })
}

/// The author for a command that shows one: the corpus's when there is a
/// corpus in `cache` (which `given` must agree with when passed); otherwise
/// the one `given` draws, and otherwise one from a fresh seed, printed so
/// passing it back reproduces the author.
///
/// # Errors
/// As [`author`] when `cache` is given.
pub fn author_or_draw(cache: Option<&Path>, given: Option<u64>) -> anyhow::Result<Author> {
    if let Some(dir) = cache {
        return author(dir, given);
    }
    let seed = resolve("seed", given);
    Ok(Author {
        persona: Persona::generate(seed),
        seed,
        persona_source: PersonaSource::Drawn,
        seed_source: if given.is_some() {
            SeedSource::Given
        } else {
            SeedSource::Drawn
        },
    })
}

/// The author for beginning or continuing the corpus in `dir`, from the seed
/// that draws it or the seed of the author it is the opposite of (not both).
/// A corpus is continued as the kind of author it was begun for.
///
/// # Errors
/// Both a seed and an author to oppose are named; they contradict what the
/// directory records; or the opposite cannot be checked against (the reason is
/// named).
pub fn corpus_author(
    dir: &Path,
    seed: Option<u64>,
    opposite_of: Option<u64>,
) -> anyhow::Result<Author> {
    anyhow::ensure!(
        seed.is_none() || opposite_of.is_none(),
        "--seed and --opposite-of both name the author: give one"
    );
    let wanted = opposite_of.map(|seed| PersonaSource::OppositeOf { seed });
    let kept = match read(dir)? {
        Some(m) => Some(m.recipe.persona_source),
        None => recorded_author(dir)?.map(|_| PersonaSource::Drawn),
    };
    if let (Some(kept), Some(wanted)) = (kept, wanted) {
        anyhow::ensure!(
            kept == wanted,
            "{}: the corpus was begun for {kept} and this run asks for {wanted} (persona source); \
             a corpus is only continued with the settings it was begun with, use another cache directory",
            dir.display(),
        );
    }
    let (seed, seed_source) = corpus_author_seed(dir, seed.or(opposite_of))?;
    let persona_source = wanted.or(kept).unwrap_or_default();
    Ok(Author {
        persona: persona_for(seed, persona_source)?,
        seed,
        persona_source,
        seed_source,
    })
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
        note(
            self.persona_source != other.persona_source,
            "persona source",
        );
        note(self.writer != other.writer, "writer");
        note(self.reader != other.reader, "reader");
        note(
            self.allow_same_family != other.allow_same_family,
            "allow same family",
        );
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
    /// Whether the check that the miner is of another model family than the
    /// writer and the reader of the corpus was set aside.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub allow_same_family: bool,
}

/// The record of how the decisions of the corpus in `dir` are read: the one
/// kept when there is one, which `miner` and `given` must agree with; else a
/// new one, with a fresh run seed (printed) when none is given and the reader
/// `seedable`, written to the directory. A new record is only made for a miner
/// of another model family than the writer and the reader the corpus records,
/// unless `allow_same_family`, which the record then says; a corpus from before
/// manifests records neither, so there is nothing to compare with.
///
/// # Errors
/// The corpus was mined with another reader or another run seed, the miner is
/// of the family of the writer or the reader, or the record cannot be read or
/// written.
pub fn begin_or_continue_mining(
    dir: &Path,
    miner: &str,
    given: Option<u64>,
    seedable: bool,
    allow_same_family: bool,
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
            if let (false, Some(corpus)) = (allow_same_family, read(dir)?) {
                check_distinct(&Roles {
                    writer: &corpus.recipe.writer,
                    reader: corpus.recipe.reader.as_deref(),
                    miner: Some(miner),
                })?;
            }
            let record = MineRecord {
                miner: miner.to_string(),
                run_seed: seedable.then(|| resolve("run-seed", given)),
                seed_derivation: SEED_DERIVATION.to_string(),
                allow_same_family,
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
