// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in proving that a result belongs to
// the exact items and model that produced it, you can procure our services by
// sending an email to info@swedishembedded.com.

//! What binds an examination's outcomes to the items they answer and the
//! model that answered.
//!
//! An item id such as `t0c1-007` names a different situation when the number
//! of planned articles or items per cell changes, and says nothing about the
//! model. So every arm of an examination has a record in `exam/manifest.json`
//! of everything its outcomes depend on, and every outcome carries the digest
//! of its item and of its model. Outcomes are only ever appended under the
//! record they were begun under; a run that differs is refused with both
//! digests named, and joins across arms or authors fail on a mismatch
//! instead of silently pairing different items.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use splinter_sdk::vocabulary::digest::{canonical_json, Digest};

use crate::arms::Outcome;
use crate::exam::{Item, Version};
use crate::prompts::render;

/// The file in the examination directory that holds the arm records.
pub const MANIFEST: &str = "manifest.json";

/// The digest of one item: its id, the prompt as the model reads it, the
/// tool definition it is offered, the label that is right and, when the
/// examination says it, whether the corpus trained on its combination.
#[must_use]
pub fn item_digest(item: &Item) -> Digest {
    let position = item
        .options
        .iter()
        .position(|&o| o == item.expected)
        .unwrap_or(0);
    let mut shown = serde_json::json!({
        "id": item.id,
        "prompt": render(&item.briefing, &item.options, &item.schema),
        "tool": item.schema.definition(&item.options),
        "expected": item.schema.label(item.expected, position),
    });
    // Only the second examination says it, so the first keeps its digests.
    if let Some(trained) = item.combo_trained {
        shown["combination_trained"] = trained.into();
    }
    digest_of(&shown)
}

/// The digest of an examination's items, in order.
#[must_use]
pub fn items_digest(items: &[Item]) -> Digest {
    let each: Vec<String> = items
        .iter()
        .map(|i| item_digest(i).as_str().to_string())
        .collect();
    digest_of(&each)
}

fn digest_of<T: Serialize + ?Sized>(value: &T) -> Digest {
    // A JSON value tree or a list of strings: serialization cannot fail.
    Digest::of(&canonical_json(value).unwrap_or_default())
}

/// `sha256:<hex>` of the file at `path`: the spelling brain reports for an
/// adapter, so the two compare directly.
///
/// # Errors
/// The file cannot be read.
pub fn file_digest(path: &Path) -> std::io::Result<String> {
    let file = std::fs::File::open(path)?;
    Ok(Digest::sha256_of_reader(file)?.to_string())
}

/// The digest naming a base checkpoint, computed as brain computes it so that
/// it equals the base digest in the adapter's card and training record: a
/// file's own digest, or for a directory the digest of its `config.json` and
/// weight files' digests in name order.
///
/// # Errors
/// The checkpoint cannot be read.
pub fn base_digest(path: &Path) -> std::io::Result<String> {
    if !path.is_dir() {
        return file_digest(path);
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(path)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                n == "config.json"
                    || n.ends_with(".safetensors")
                    || n.ends_with(".bin")
                    || n.ends_with(".index.json")
            })
        })
        .collect();
    files.sort();
    let mut combined = String::new();
    for file in &files {
        let name = file
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        combined.push_str(&format!("{name}={}\n", file_digest(file)?));
    }
    Ok(Digest::sha256_of(combined.as_bytes()).to_string())
}

/// What a model is, for the purpose of its answers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelIdentity {
    /// Digest of the base checkpoint.
    pub base_digest: String,
    /// Digest of the adapter file, if one is attached.
    pub adapter_digest: Option<String>,
    /// How the model decodes: `greedy`, or `sample:<temperature>`.
    pub decoding: String,
    /// The seed of a sampled decoding; greedy decoding draws nothing.
    pub decoding_seed: Option<u64>,
}

impl ModelIdentity {
    /// The digest an outcome carries for its model.
    #[must_use]
    pub fn digest(&self) -> Digest {
        digest_of(self)
    }
}

/// Everything the outcomes of one arm depend on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArmRecord {
    /// The arm.
    pub arm: String,
    /// Which examination the items are of. Records of the first version do
    /// not name it, which keeps their digests as they were.
    #[serde(default = "first_exam", skip_serializing_if = "is_first")]
    pub exam: Version,
    /// Digest of the items the arm is asked.
    pub items_digest: String,
    /// How many articles the corpus was planned with.
    pub articles: usize,
    /// Items of each kind of difficulty.
    pub per_cell: usize,
    /// The most tokens a reply may run to.
    pub max_tokens: u32,
    /// Digest of what the arm is told besides the item: its system turn, and
    /// for the retrieval arm the corpus it draws paragraphs from.
    pub context_digest: String,
    /// The model that answers.
    pub model: ModelIdentity,
    /// How many times each item is asked, for a sampled arm; a greedy arm
    /// asks once and does not say.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draws: Option<usize>,
    /// Whether the record was written for outcomes that already existed,
    /// from the flags given, rather than begun with them.
    #[serde(default)]
    pub adopted: bool,
}

impl ArmRecord {
    /// The digest of the record: equal exactly when the outcomes are
    /// interchangeable. Whether it was adopted is how it came to be written
    /// and does not enter.
    #[must_use]
    pub fn digest(&self) -> Digest {
        let mut same = self.clone();
        same.adopted = false;
        digest_of(&same)
    }

    /// The fields in which `self` differs from `other`, for the refusal.
    fn differences(&self, other: &Self) -> Vec<&'static str> {
        let mut named = Vec::new();
        let mut note = |differs: bool, name| {
            if differs {
                named.push(name);
            }
        };
        note(self.exam != other.exam, "exam");
        note(self.items_digest != other.items_digest, "items");
        note(self.articles != other.articles, "articles");
        note(self.per_cell != other.per_cell, "per_cell");
        note(self.max_tokens != other.max_tokens, "max_tokens");
        note(self.context_digest != other.context_digest, "context");
        note(self.model.base_digest != other.model.base_digest, "base");
        note(
            self.model.adapter_digest != other.model.adapter_digest,
            "adapter",
        );
        note(self.model.decoding != other.model.decoding, "decoding");
        note(
            self.model.decoding_seed != other.model.decoding_seed,
            "decoding seed",
        );
        note(self.draws != other.draws, "draws");
        named
    }
}

/// The examination a record that does not name one is of.
fn first_exam() -> Version {
    Version::V1
}

/// Whether `exam` is the examination as first reported.
// By reference because serde's `skip_serializing_if` hands it one.
#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_first(exam: &Version) -> bool {
    *exam == Version::V1
}

/// The records of every arm examined in one directory.
#[derive(Debug, Default, Serialize, Deserialize)]
struct Manifest {
    arms: BTreeMap<String, ArmRecord>,
}

fn read_manifest(path: &Path) -> anyhow::Result<Manifest> {
    match std::fs::read_to_string(path) {
        Ok(text) => {
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Manifest::default()),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

/// The record kept for `arm` in the examination directory `dir`, if any.
///
/// # Errors
/// The manifest cannot be read or is not one.
pub fn recorded(dir: &Path, arm: &str) -> anyhow::Result<Option<ArmRecord>> {
    Ok(read_manifest(&dir.join(MANIFEST))?.arms.remove(arm))
}

/// Whether `record` is now the one `outcomes` are appended under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bound {
    /// The arm has no outcomes yet; its record was written.
    Begun,
    /// The arm's record is this one already.
    Same,
    /// Outcomes existed without a record; one was written from the flags
    /// given, and they were not checked against it.
    Adopted,
}

/// Binds the outcomes in `outcomes` to `record` in the examination directory
/// `dir`: writes the record when the arm has none and no outcomes, accepts a
/// record equal to the one kept, and refuses anything else. Outcomes that
/// exist without any record are taken over only with `adopt`.
///
/// # Errors
/// The record differs from the one the outcomes were made under (both
/// digests are named), the outcomes have no record and `adopt` is not set,
/// or the manifest cannot be read or written.
pub fn bind(dir: &Path, outcomes: &Path, record: &ArmRecord, adopt: bool) -> anyhow::Result<Bound> {
    let path = dir.join(MANIFEST);
    let mut manifest = read_manifest(&path)?;
    let has_outcomes = std::fs::metadata(outcomes).is_ok_and(|m| m.len() > 0);
    let bound = match manifest.arms.get(&record.arm) {
        Some(kept) if kept.digest() == record.digest() => return Ok(Bound::Same),
        Some(kept) => anyhow::bail!(
            "{}: the outcomes of arm {} were made under manifest {} and this run is {} (differs in {}); \
             outcomes under another manifest are never appended to, use another cache directory",
            outcomes.display(),
            record.arm,
            kept.digest(),
            record.digest(),
            kept.differences(record).join(", "),
        ),
        None if has_outcomes && !adopt => anyhow::bail!(
            "{}: outcomes exist for arm {} but {} records no manifest for it (they predate manifests); \
             pass --adopt to record this run's flags as their manifest",
            outcomes.display(),
            record.arm,
            path.display(),
        ),
        None if has_outcomes => Bound::Adopted,
        None => Bound::Begun,
    };
    let mut written = record.clone();
    written.adopted = bound == Bound::Adopted;
    manifest.arms.insert(record.arm.clone(), written);
    std::fs::create_dir_all(dir)?;
    write_whole(&path, &serde_json::to_vec_pretty(&manifest)?)?;
    if bound == Bound::Adopted {
        eprintln!(
            "{}: adopted the existing {} outcomes under manifest {} written from the flags given; \
             they were not checked against it",
            outcomes.display(),
            record.arm,
            record.digest(),
        );
    }
    Ok(bound)
}

fn write_whole(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let part = path.with_extension("part");
    std::fs::write(&part, bytes).with_context(|| format!("writing {}", part.display()))?;
    std::fs::rename(&part, path).with_context(|| format!("renaming to {}", path.display()))
}

/// Checks that `outcomes`, read from `source`, were made by the model
/// `record` names. An outcome without digests (written before outcomes
/// carried them) is accepted only for an arm whose record was adopted.
///
/// # Errors
/// An outcome names another model than the record's, or has no digest and the
/// arm was not adopted.
pub fn verify_outcomes(
    record: &ArmRecord,
    outcomes: &[Outcome],
    source: &Path,
) -> anyhow::Result<()> {
    let model = record.model.digest();
    for outcome in outcomes {
        if outcome.model.is_empty() && record.adopted {
            continue;
        }
        anyhow::ensure!(
            outcome.model == model.as_str(),
            "{}: outcome {} was made by model {:?} and the manifest of arm {} says {model}",
            source.display(),
            outcome.id,
            outcome.model,
            record.arm,
        );
    }
    Ok(())
}

/// Checks that `outcomes`, read from `source`, answer `items`: the items are
/// the ones the record was made for, and every outcome answers the item its
/// id names now.
///
/// # Errors
/// The items differ from the record's (both digests are named), or an outcome
/// answers another item than its id names now.
pub fn verify_items(
    record: &ArmRecord,
    items: &[Item],
    outcomes: &[Outcome],
    source: &Path,
) -> anyhow::Result<()> {
    let now = items_digest(items);
    anyhow::ensure!(
        record.items_digest == now.as_str(),
        "{}: the outcomes of arm {} were made for items {} and the items these flags make are {now}",
        source.display(),
        record.arm,
        record.items_digest,
    );
    for outcome in outcomes {
        let Some(item) = items.iter().find(|i| i.id == outcome.id) else {
            continue;
        };
        if outcome.item.is_empty() && record.adopted {
            continue;
        }
        let digest = item_digest(item);
        anyhow::ensure!(
            outcome.item == digest.as_str(),
            "{}: outcome {} answers item {:?} and the item that id names now is {digest}",
            source.display(),
            outcome.id,
            outcome.item,
        );
    }
    Ok(())
}

/// Checks that the arms in `records` were all asked the same items, so that
/// comparing them pairs like with like.
///
/// # Errors
/// Two arms were asked different items (both digests are named).
pub fn verify_same_items(records: &[ArmRecord]) -> anyhow::Result<()> {
    if let Some(first) = records.first() {
        for other in &records[1..] {
            anyhow::ensure!(
                other.items_digest == first.items_digest,
                "arm {} was asked items {} and arm {} items {}; they cannot be compared",
                first.arm,
                first.items_digest,
                other.arm,
                other.items_digest,
            );
        }
    }
    Ok(())
}
