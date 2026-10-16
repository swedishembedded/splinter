// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements the measurement of fine-tuned language
// models for its clients. If your team needs expertise in pre-registering a
// replication study so that no choice can follow its results, you can
// procure our services by sending an email to info@swedishembedded.com.

//! The study protocol: everything that is fixed before any run, written once
//! and hashed.
//!
//! The protocol names the authors (their seeds are drawn when it is frozen and
//! read back from the file ever after), the training seeds of each author, the
//! three models that write, read and mine the corpus, the size of the corpus
//! and of the examination, the arms, how the adapters are trained, the primary
//! comparison, the family of comparisons and the correction applied over it,
//! and the decodings. Its digest is the blake3 address of its canonical JSON;
//! every manifest the study writes records it, and a stage refuses to continue
//! under a protocol whose digest is not the one the work was begun under.

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_sdk::vocabulary::digest::{canonical_json, Digest};

use super::arm_key::{ArmKey, FIRST_LISTED, MAJORITY};
use super::power::{Derivation, Observed};
use crate::dataset::Briefings;
use crate::exam::Version;
use crate::family::{check_distinct, Roles};
use crate::manifest::SeedSource;
use crate::training::{DEFAULT_ALPHA, DEFAULT_RANK, DEFAULT_RECORDS_PER_STEP};

/// The name of the protocol file in a study directory.
pub const PROTOCOL_FILE: &str = "protocol.json";

/// The format this module reads and writes.
pub const FORMAT: &str = "synthetic-author-study-protocol-1";

/// How many authors a study has unless it is told otherwise.
pub const DEFAULT_AUTHORS: usize = 8;
/// How many times each author's adapter is trained (each under a training
/// seed of its own) unless told otherwise.
pub const DEFAULT_REPLICATES: usize = 3;
/// Articles planned for each author's corpus.
pub const DEFAULT_ARTICLES: usize = 60;
/// Episodes each article tells.
pub const DEFAULT_EPISODES_PER_ARTICLE: usize = 3;
/// The fewest words an article may have.
pub const DEFAULT_MIN_WORDS: usize = 700;
/// Attempts at an article before it is given up on.
pub const DEFAULT_ATTEMPTS: usize = 3;
/// The most requests the corpus stage of one author may make.
pub const DEFAULT_MAX_CALLS: usize = 600;
/// The most output tokens the corpus stage of one author may write.
pub const DEFAULT_MAX_OUTPUT_TOKENS: u64 = 600_000;
/// Articles written at a time.
pub const DEFAULT_WORKERS: usize = 4;
/// The most tokens a reply to an exam item may run to.
pub const DEFAULT_MAX_TOKENS: u32 = 256;
/// Optimizer steps of each adapter.
pub const DEFAULT_STEPS: u32 = 240;
/// The learning rate of each adapter, fixed so that no seed is trained under
/// a trainer default nobody recorded.
pub const DEFAULT_LEARNING_RATE: f64 = 3e-4;
/// The temperature of the secondary, sampled decoding.
pub const DEFAULT_TEMPERATURE: f64 = 0.7;
/// How many times each item is asked under the sampled decoding.
pub const DEFAULT_DRAWS: usize = 8;
/// The level of the one-sided tests and of the family-wise error rate.
pub const DEFAULT_ALPHA_LEVEL: f64 = 0.05;
/// The power the examination is sized for.
pub const DEFAULT_POWER: f64 = 0.8;
/// How many of an author's training runs the ablation adapters are trained at.
pub const DEFAULT_ABLATION_REPLICATES: usize = 1;

/// What tuned against retrieval came to for the first author of the earlier
/// examination, from which the examination is sized: of 160 items, 37 on
/// which exactly one of the two was right, 23 of them for the adapter.
pub const EARLIER_OBSERVATION: Observed = Observed {
    items: 160,
    discordant: 37,
    candidate_wins: 23,
};
/// Where [`EARLIER_OBSERVATION`] comes from, as the sizing records it.
pub const EARLIER_SOURCE: &str =
    "tuned against retrieval for the first author of the earlier examination, greedy decoding";

/// How many tries a seed draw gets to avoid a repeat before it is an error.
const DRAW_TRIES: usize = 1000;

/// The three models of a corpus, by role.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRoles {
    /// Writes the essays.
    pub writer: String,
    /// Reads each draft for the gate.
    pub reader: String,
    /// Reads the decisions out of the kept essays.
    pub miner: String,
}

/// One author of the study.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthorPlan {
    /// The seed the author, its world and its examination are drawn from.
    pub seed: u64,
    /// The seeds the author's adapters are trained with.
    pub training_seeds: Vec<u64>,
}

/// Where the seeds of the study came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    /// The author seeds: drawn when the protocol was frozen, or given.
    pub authors: SeedSource,
    /// The training seeds.
    pub training: SeedSource,
}

/// The corpus each author's essays are written into.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusSettings {
    /// Articles planned: the corpus is planned with this many, and the
    /// examination draws no situation an article tells.
    pub articles: usize,
    /// Episodes each article tells.
    pub episodes_per_article: usize,
    /// The fewest words an article may have.
    pub min_words: usize,
    /// Attempts at an article before it is given up on.
    pub attempts: usize,
    /// The most requests one author's corpus may make.
    pub max_calls: usize,
    /// The most output tokens one author's corpus may write.
    pub max_output_tokens: u64,
    /// Articles written at a time.
    pub workers: usize,
}

/// The examination, frozen before any adapter is trained.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExamPlan {
    /// Which examination.
    pub version: Version,
    /// Items of each kind of difficulty.
    pub per_cell: usize,
    /// The most tokens a reply may run to.
    pub max_tokens: u32,
    /// How `per_cell` was sized.
    pub sizing: Derivation,
}

/// How each adapter is trained.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingPlan {
    /// Optimizer steps.
    pub steps: u32,
    /// LoRA rank.
    pub rank: u32,
    /// LoRA alpha.
    pub alpha: f64,
    /// The learning rate.
    pub learning_rate: f64,
    /// Records averaged into one optimizer step.
    pub records_per_step: u32,
    /// Whether the base is held at bf16.
    pub bf16: bool,
}

/// The adapters trained to ask which part of the records accounts for the
/// result.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ablations {
    /// The settings of the records, besides the full one.
    pub briefings: Vec<Briefings>,
    /// How many of an author's training seeds (the first ones) each ablation
    /// adapter is trained at.
    pub training_seeds: usize,
}

/// A pair of arms, the first tested against the second.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Comparison {
    /// The arm whose gain is tested.
    pub candidate: String,
    /// The arm, or constant policy, it is tested against.
    pub reference: String,
}

impl Comparison {
    /// The comparison of `candidate` against `reference`.
    #[must_use]
    pub fn new(candidate: &str, reference: &str) -> Self {
        Self {
            candidate: candidate.to_string(),
            reference: reference.to_string(),
        }
    }

    /// The label used in tables: `candidate over reference`.
    #[must_use]
    pub fn label(&self) -> String {
        format!("{} over {}", self.candidate, self.reference)
    }
}

/// The correction for testing many comparisons.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Multiplicity {
    /// Holm's step-down procedure over the protocol's whole family of
    /// comparisons, each tested across authors; the per-seed and per-author
    /// tests are reported beside them and are not part of the family.
    HolmOverComparisons {
        /// The family-wise error rate.
        alpha: f64,
    },
}

/// How the arms decode.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecodingPlan {
    /// The decoding of every confirmatory result.
    pub primary: PrimaryDecoding,
    /// The sampled decoding asked beside it, if any.
    pub secondary: Option<SampledDecoding>,
}

/// The decoding the conclusions rest on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PrimaryDecoding {
    /// The likeliest token every time: one answer per item.
    Greedy,
}

/// A sampled decoding.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledDecoding {
    /// The temperature.
    pub temperature: f64,
    /// Times each item is asked.
    pub draws: usize,
}

/// The whole protocol.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Protocol {
    /// Always [`FORMAT`].
    pub format: String,
    /// The authors, each with its training seeds.
    pub authors: Vec<AuthorPlan>,
    /// Where the seeds came from.
    pub provenance: Provenance,
    /// The models that write, read and mine.
    pub models: ModelRoles,
    /// The corpus.
    pub corpus: CorpusSettings,
    /// The examination.
    pub exam: ExamPlan,
    /// The arms of the examination to run, by name. `tuned` is always there.
    pub arms: Vec<String>,
    /// The ablation adapters.
    pub ablations: Ablations,
    /// How adapters are trained.
    pub training: TrainingPlan,
    /// The primary comparison, tested across authors; it is one of
    /// `comparisons`.
    pub primary: Comparison,
    /// The family of pre-registered comparisons.
    pub comparisons: Vec<Comparison>,
    /// The correction over `comparisons`.
    pub multiplicity: Multiplicity,
    /// The decodings.
    pub decoding: DecodingPlan,
}

fn plain_arms() -> Vec<String> {
    [
        "base",
        "named",
        "principles",
        "principles-verdicts",
        "retrieval",
        "retrieval-2",
        "retrieval-12",
        "retrieval-records-2",
        "retrieval-records-6",
        "retrieval-records-12",
        "tuned",
    ]
    .map(String::from)
    .to_vec()
}

fn standard_comparisons() -> Vec<Comparison> {
    let mut family: Vec<Comparison> = [
        "base",
        "named",
        "principles",
        "principles-verdicts",
        "retrieval",
        "retrieval-2",
        "retrieval-12",
        "retrieval-records-2",
        "retrieval-records-6",
        "retrieval-records-12",
        "tuned-reader-only",
        "tuned-fixed-wording-only",
        FIRST_LISTED,
        MAJORITY,
    ]
    .iter()
    .map(|reference| Comparison::new("tuned", reference))
    .collect();
    family.extend([
        Comparison::new("retrieval", "base"),
        Comparison::new("principles", "base"),
        Comparison::new("principles-verdicts", "principles"),
    ]);
    family
}

impl Protocol {
    /// The standard protocol for `models`, with `exam` as the examination's
    /// size and no authors yet: [`freeze`] draws them.
    #[must_use]
    pub fn standard(models: ModelRoles, sizing: Derivation) -> Self {
        let comparisons = standard_comparisons();
        Self {
            format: FORMAT.to_string(),
            authors: Vec::new(),
            provenance: Provenance {
                authors: SeedSource::Drawn,
                training: SeedSource::Drawn,
            },
            models,
            corpus: CorpusSettings {
                articles: DEFAULT_ARTICLES,
                episodes_per_article: DEFAULT_EPISODES_PER_ARTICLE,
                min_words: DEFAULT_MIN_WORDS,
                attempts: DEFAULT_ATTEMPTS,
                max_calls: DEFAULT_MAX_CALLS,
                max_output_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
                workers: DEFAULT_WORKERS,
            },
            exam: ExamPlan {
                version: Version::V2,
                per_cell: sizing.per_cell,
                max_tokens: DEFAULT_MAX_TOKENS,
                sizing,
            },
            arms: plain_arms(),
            ablations: Ablations {
                briefings: vec![Briefings::ReaderOnly, Briefings::FixedWordingOnly],
                training_seeds: DEFAULT_ABLATION_REPLICATES,
            },
            training: TrainingPlan {
                steps: DEFAULT_STEPS,
                rank: DEFAULT_RANK,
                alpha: f64::from(DEFAULT_ALPHA),
                learning_rate: DEFAULT_LEARNING_RATE,
                records_per_step: DEFAULT_RECORDS_PER_STEP,
                bf16: false,
            },
            primary: Comparison::new("tuned", "retrieval"),
            comparisons,
            multiplicity: Multiplicity::HolmOverComparisons {
                alpha: DEFAULT_ALPHA_LEVEL,
            },
            decoding: DecodingPlan {
                primary: PrimaryDecoding::Greedy,
                secondary: Some(SampledDecoding {
                    temperature: DEFAULT_TEMPERATURE,
                    draws: DEFAULT_DRAWS,
                }),
            },
        }
    }

    /// The digest the protocol is known by: the address of its canonical JSON.
    #[must_use]
    pub fn digest(&self) -> Digest {
        // A protocol is plain data: serializing it cannot fail.
        Digest::of(&canonical_json(self).unwrap_or_default())
    }

    /// Every arm the study asks, by key: the arms listed and, for each
    /// ablation, the tuned arm trained on its records.
    ///
    /// # Errors
    /// An arm is not one the examination has.
    pub fn arm_keys(&self) -> anyhow::Result<Vec<ArmKey>> {
        let mut keys = self
            .arms
            .iter()
            .map(|name| {
                ArmKey::parse(name)
                    .filter(|k| k.briefings() == Briefings::Both)
                    .with_context(|| format!("arm {name:?} is not one the study asks"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        keys.extend(self.ablations.briefings.iter().map(|&b| ArmKey::tuned(b)));
        Ok(keys)
    }

    /// The training seeds the adapters of `key` are trained at for `author`:
    /// every seed of the author for the full adapter, the first
    /// `ablations.training_seeds` for an ablation, none for an arm without an
    /// adapter.
    #[must_use]
    pub fn seeds_for(&self, author: &AuthorPlan, key: ArmKey) -> Vec<u64> {
        if !key.is_adapter() {
            return Vec::new();
        }
        let count = if key.briefings() == Briefings::Both {
            author.training_seeds.len()
        } else {
            self.ablations.training_seeds
        };
        author.training_seeds.iter().copied().take(count).collect()
    }

    /// Checks that the protocol says what a study can be run from.
    ///
    /// # Errors
    /// The first thing that is wrong, named: no authors or repeated seeds,
    /// models of one family, an arm or comparison that does not exist, a
    /// primary outside the family, or a size or level out of range.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.format == FORMAT,
            "protocol format {:?} is not {FORMAT:?}",
            self.format
        );
        self.validate_seeds()?;
        check_distinct(&Roles {
            writer: &self.models.writer,
            reader: Some(&self.models.reader),
            miner: Some(&self.models.miner),
        })
        .context("the writer, the gate reader and the miner must be three different families")?;
        anyhow::ensure!(
            self.corpus.articles > 0
                && self.corpus.episodes_per_article > 0
                && self.corpus.attempts > 0
                && self.corpus.workers > 0
                && self.exam.per_cell > 0
                && self.exam.max_tokens > 0
                && self.training.steps > 0
                && self.training.records_per_step > 0,
            "the corpus, examination and training sizes must be positive"
        );
        anyhow::ensure!(
            self.exam.per_cell == self.exam.sizing.per_cell,
            "the examination has {} items per kind and its recorded sizing derives {}",
            self.exam.per_cell,
            self.exam.sizing.per_cell
        );
        self.validate_arms()?;
        let Multiplicity::HolmOverComparisons { alpha } = self.multiplicity;
        anyhow::ensure!(alpha > 0.0 && alpha < 1.0, "alpha {alpha} is not a level");
        if let Some(sampled) = self.decoding.secondary {
            anyhow::ensure!(
                sampled.draws > 0 && sampled.temperature > 0.0,
                "the sampled decoding needs a positive temperature and at least one draw"
            );
        }
        Ok(())
    }

    fn validate_seeds(&self) -> anyhow::Result<()> {
        anyhow::ensure!(!self.authors.is_empty(), "the study has no authors");
        let mut seen = Vec::new();
        for author in &self.authors {
            anyhow::ensure!(
                !seen.contains(&author.seed),
                "author seed {} is listed twice",
                author.seed
            );
            seen.push(author.seed);
            anyhow::ensure!(
                !author.training_seeds.is_empty(),
                "author {} has no training seeds",
                author.seed
            );
            let mut own = author.training_seeds.clone();
            own.sort_unstable();
            own.dedup();
            anyhow::ensure!(
                own.len() == author.training_seeds.len(),
                "author {} repeats a training seed",
                author.seed
            );
            anyhow::ensure!(
                self.ablations.training_seeds <= author.training_seeds.len(),
                "ablations are trained at {} seeds and author {} has {}",
                self.ablations.training_seeds,
                author.seed,
                author.training_seeds.len()
            );
        }
        Ok(())
    }

    fn validate_arms(&self) -> anyhow::Result<()> {
        let keys = self.arm_keys()?;
        anyhow::ensure!(
            self.arms.iter().any(|a| a == "tuned"),
            "the arms must include tuned"
        );
        anyhow::ensure!(
            !self.ablations.briefings.contains(&Briefings::Both),
            "an ablation names a setting other than the full records"
        );
        let names: Vec<String> = keys.iter().map(|k| k.name()).collect();
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        anyhow::ensure!(sorted.len() == names.len(), "an arm is listed twice");
        for comparison in &self.comparisons {
            anyhow::ensure!(
                names.contains(&comparison.candidate),
                "{}: the candidate is not an arm of the study",
                comparison.label()
            );
            anyhow::ensure!(
                names.contains(&comparison.reference)
                    || [FIRST_LISTED, MAJORITY].contains(&comparison.reference.as_str()),
                "{}: the reference is neither an arm of the study nor a constant policy",
                comparison.label()
            );
            anyhow::ensure!(
                comparison.candidate != comparison.reference,
                "{} compares an arm with itself",
                comparison.label()
            );
        }
        anyhow::ensure!(
            self.comparisons.contains(&self.primary),
            "the primary comparison {} is not in the family",
            self.primary.label()
        );
        Ok(())
    }

    /// The top-level fields in which `self` and `other` differ, for a refusal.
    fn differences(&self, other: &Self) -> Vec<String> {
        let (Ok(Value::Object(a)), Ok(Value::Object(b))) =
            (serde_json::to_value(self), serde_json::to_value(other))
        else {
            return vec!["protocol".to_string()];
        };
        a.iter()
            .filter(|(key, value)| b.get(*key) != Some(*value))
            .map(|(key, _)| key.clone())
            .collect()
    }
}

/// Where the seeds of a list come from when a protocol is frozen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Seeds {
    /// Draw this many fresh ones.
    Draw(usize),
    /// Use these.
    Given(Vec<u64>),
}

impl Seeds {
    /// Whether `kept`, the seeds a frozen protocol has, are the ones this asks
    /// for: the given ones, or as many as were to be drawn.
    fn matches(&self, kept: &[u64]) -> bool {
        match self {
            Self::Given(given) => given == kept,
            Self::Draw(count) => *count == kept.len(),
        }
    }

    fn source(&self) -> SeedSource {
        match self {
            Self::Draw(_) => SeedSource::Drawn,
            Self::Given(_) => SeedSource::Given,
        }
    }

    /// The seeds, drawing with `draw` any that are not given; none repeats
    /// one in `taken` or one drawn before it.
    fn resolve(&self, draw: &mut dyn FnMut() -> u64, taken: &[u64]) -> anyhow::Result<Vec<u64>> {
        match self {
            Self::Given(seeds) => Ok(seeds.clone()),
            Self::Draw(count) => {
                let mut seeds: Vec<u64> = Vec::with_capacity(*count);
                for _ in 0..*count {
                    let fresh = (0..DRAW_TRIES)
                        .map(|_| draw())
                        .find(|s| !seeds.contains(s) && !taken.contains(s))
                        .context("the seed source keeps repeating seeds")?;
                    seeds.push(fresh);
                }
                Ok(seeds)
            }
        }
    }
}

/// What to freeze: a protocol without authors and where its seeds come from.
#[derive(Clone, Debug)]
pub struct Spec {
    /// The protocol, with `authors` empty.
    pub template: Protocol,
    /// How many authors, or which seeds.
    pub author_seeds: Seeds,
    /// How many training seeds each author has, or which seeds every author
    /// is trained with.
    pub training_seeds: Seeds,
}

/// A protocol that has been written.
#[derive(Clone, Debug)]
pub struct Frozen {
    /// The protocol.
    pub protocol: Protocol,
    /// Its digest.
    pub digest: Digest,
    /// Whether this call wrote it (otherwise it was already there).
    pub created: bool,
}

/// The path of the protocol in the study directory `root`.
#[must_use]
pub fn path(root: &Path) -> PathBuf {
    root.join(PROTOCOL_FILE)
}

/// Reads and validates the protocol frozen in `root`.
///
/// # Errors
/// There is none, it cannot be parsed, or it is not valid.
pub fn read(root: &Path) -> anyhow::Result<Frozen> {
    let file = path(root);
    let text = std::fs::read_to_string(&file).with_context(|| {
        format!(
            "reading {}: freeze the protocol first (`study freeze`)",
            file.display()
        )
    })?;
    let protocol: Protocol =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", file.display()))?;
    protocol
        .validate()
        .with_context(|| format!("{} is not a valid protocol", file.display()))?;
    Ok(Frozen {
        digest: protocol.digest(),
        protocol,
        created: false,
    })
}

/// Freezes the protocol `spec` describes into `root`: draws the seeds with
/// `draw`, writes the file whole or not at all, and returns it. If `root`
/// already holds a protocol, that one is returned untouched provided `spec`
/// describes the same protocol (the seeds it would have drawn are not drawn);
/// a protocol that differs is refused and nothing is written, because a frozen
/// protocol is never changed or redrawn.
///
/// # Errors
/// The protocol is not valid, differs from the frozen one (the differing
/// fields are named), or cannot be written.
pub fn freeze(root: &Path, spec: &Spec, draw: &mut dyn FnMut() -> u64) -> anyhow::Result<Frozen> {
    if path(root).exists() {
        let frozen = read(root)?;
        let mut same = spec.template.clone();
        same.authors = frozen.protocol.authors.clone();
        same.provenance = frozen.protocol.provenance;
        let kept: Vec<u64> = frozen.protocol.authors.iter().map(|a| a.seed).collect();
        let seeds_agree = spec.author_seeds.matches(&kept)
            && frozen
                .protocol
                .authors
                .iter()
                .all(|a| spec.training_seeds.matches(&a.training_seeds));
        let mut differing = same.differences(&frozen.protocol);
        if !seeds_agree {
            differing.push("authors".to_string());
        }
        anyhow::ensure!(
            differing.is_empty(),
            "{} is frozen with digest {} and this protocol differs from it ({}); a frozen \
             protocol is never changed or redrawn, use another study directory",
            path(root).display(),
            frozen.digest,
            differing.join(", "),
        );
        return Ok(frozen);
    }
    let author_seeds = spec.author_seeds.resolve(draw, &[])?;
    let mut protocol = spec.template.clone();
    for seed in author_seeds {
        let training = spec.training_seeds.resolve(draw, &[seed])?;
        protocol.authors.push(AuthorPlan {
            seed,
            training_seeds: training,
        });
    }
    protocol.provenance = Provenance {
        authors: spec.author_seeds.source(),
        training: spec.training_seeds.source(),
    };
    protocol.validate()?;
    std::fs::create_dir_all(root).with_context(|| format!("creating {}", root.display()))?;
    let file = path(root);
    let part = file.with_extension("part");
    std::fs::write(&part, serde_json::to_vec_pretty(&protocol)?)
        .with_context(|| format!("writing {}", part.display()))?;
    std::fs::rename(&part, &file).with_context(|| format!("writing {}", file.display()))?;
    Ok(Frozen {
        digest: protocol.digest(),
        protocol,
        created: true,
    })
}
