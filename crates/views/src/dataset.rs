// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A projection written as a dataset file, with its manifest beside it.
//!
//! Each [`Objective`] maps to one file format:
//!
//! * SFT, and classification rendered as SFT, are written as brain's
//!   `generic-messages-v2` ([`Format::GenericMessagesV2`]): one
//!   conversation per line, `{"messages":[{"role","content","train"}...],
//!   "metadata":{...}}`, where `train` marks what is supervised and
//!   `metadata` is carried but never rendered. The file is written beside
//!   its destination, parsed with brain's own dataset parser (through
//!   `splinter-policy`), and only then moved into place - so a dataset the
//!   trainer would refuse is never reported written.
//! * Preference pairs, contrastive triples, rewarded trajectories and raw
//!   text have no trainer in brain's public SDK. The writer refuses them
//!   with [`ViewError::ObjectiveNotTrainable`] unless the caller sets
//!   [`WriteOptions::export_only`], and then writes Splinter's export
//!   format ([`Format::SplinterExportV1`], [`EXPORT_FORMAT`]): one JSON
//!   object per line, `{"format":"splinter-export-v1","shape":<shape>,
//!   <the shape's fields>,"metadata":{...}}`, the shapes and their fields
//!   being [`RecordBody`]'s variants in snake case (`preference`: `prompt`,
//!   `chosen`, `rejected`; `contrastive`: `query`, `positive`,
//!   `negatives`; `rewarded`: `messages`, `reward`; `text`: `text`).
//!   Messages are `generic-messages-v2` messages. No file brain would
//!   misread is ever written: the export format is not the chat format,
//!   and brain's parser refuses it.
//!
//! A trainable objective is always written in brain's format, with or
//! without `export_only`: one objective, one file shape.
//!
//! Beside every dataset `<name>`, at [`manifest_path`] (`<name>.manifest.json`),
//! lies its [`Manifest`] in canonical JSON (sorted keys, compact): the
//! format, the dataset's digest, the view, objective, strip policy and
//! minimum strength, the experiences, tasks and sources its records came
//! from, and the counts of records and of exclusions by reason. The
//! manifest is content-addressed: [`Dataset::manifest`] is the digest of
//! its bytes, and it names the dataset by digest in turn. The dataset is
//! moved into place first, then the manifest; a dataset whose manifest is
//! missing or names another digest was interrupted and is not complete.

use std::collections::{BTreeMap, HashSet};
use std::hash::Hash;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_store::annotation::Strength;
use splinter_store::digest::{canonical_json, Digest};
use splinter_store::experience::ExperienceId;

use crate::{
    Exclusion, Objective, Projection, Record, RecordBody, RecordMetadata, Strip, ViewError,
};

/// The `format` every line of an export-only file carries.
pub const EXPORT_FORMAT: &str = "splinter-export-v1";

/// A dataset file's format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Format {
    /// Brain's chat dataset format.
    #[serde(rename = "generic-messages-v2")]
    GenericMessagesV2,
    /// Splinter's export-only format, which no trainer reads.
    #[serde(rename = "splinter-export-v1")]
    SplinterExportV1,
}

/// How [`write_dataset`] may write.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WriteOptions {
    /// Write an objective brain cannot train in Splinter's export format
    /// instead of refusing it.
    pub export_only: bool,
}

/// A dataset file, as written and validated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dataset {
    /// Where it is.
    pub path: PathBuf,
    /// Its format.
    pub format: Format,
    /// The digest of its bytes.
    pub digest: Digest,
    /// Records in it.
    pub records: usize,
    /// Supervised messages across them, as brain's parser counted them;
    /// `None` for a file brain does not parse.
    pub trained_messages: Option<usize>,
    /// The digest of its manifest's bytes.
    pub manifest: Digest,
}

/// What a dataset holds and how it was projected.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// The dataset's format.
    pub format: Format,
    /// The digest of the dataset's bytes.
    pub dataset: Digest,
    /// The view that projected it.
    pub view: String,
    /// The objective its records serve.
    pub objective: Objective,
    /// The strip policy applied; `None` for a view with no student input.
    pub strip: Option<Strip>,
    /// The minimum decision strength required; `None` for a view that
    /// reads no verdict.
    pub min_strength: Option<Strength>,
    /// The experiences the records came from, in first-use order.
    pub experiences: Vec<ExperienceId>,
    /// The tasks the records came from, in first-use order.
    pub tasks: Vec<Digest>,
    /// The source contents the records hold, in first-use order.
    pub sources: Vec<Digest>,
    /// Records and exclusions.
    pub counts: Counts,
}

/// How many records a projection yielded and how many candidates it left
/// out, by reason.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Counts {
    /// Records written.
    pub records: usize,
    /// Candidates left out, by reason.
    pub excluded: BTreeMap<Exclusion, usize>,
}

/// Where the manifest of the dataset at `dataset` lies.
#[must_use]
pub fn manifest_path(dataset: &Path) -> PathBuf {
    let mut name = dataset.file_name().unwrap_or_default().to_os_string();
    name.push(".manifest.json");
    dataset.with_file_name(name)
}

/// One line of the chat format: exactly the fields brain's parser accepts.
#[derive(Serialize)]
struct ChatLine<'a> {
    messages: &'a [splinter_lab::WireMessage],
    metadata: &'a RecordMetadata,
}

/// One line of the export format.
#[derive(Serialize)]
struct ExportLine<'a> {
    format: &'static str,
    #[serde(flatten)]
    body: &'a RecordBody,
    metadata: &'a RecordMetadata,
}

/// Writes `projection`'s records to `path` in the format its objective maps
/// to, and its manifest beside it. Refuses an empty projection, a record
/// whose shape does not serve the objective, an objective brain cannot
/// train unless `options.export_only`, and a chat dataset brain's parser
/// rejects or that supervises nothing; in every case nothing is left at
/// `path`.
pub fn write_dataset(
    path: &Path,
    projection: &Projection,
    options: WriteOptions,
) -> Result<Dataset, ViewError> {
    let objective = projection.objective;
    let format = if objective.trainable_by_brain() {
        Format::GenericMessagesV2
    } else if options.export_only {
        Format::SplinterExportV1
    } else {
        return Err(ViewError::ObjectiveNotTrainable { objective });
    };
    if projection.records.is_empty() {
        return Err(ViewError::Empty);
    }
    let mut text = String::new();
    for (index, record) in projection.records.iter().enumerate() {
        text.push_str(&line(format, objective, index, record)?);
        text.push('\n');
    }
    let io = |path: &Path| {
        let path = path.to_path_buf();
        move |source| ViewError::Io { path, source }
    };
    let pending = path.with_extension("pending");
    splinter_store::write_atomic(&pending, &text).map_err(io(&pending))?;
    let trained_messages = match format {
        Format::GenericMessagesV2 => match validate(path, &pending) {
            Ok(trained) => Some(trained),
            Err(e) => {
                // The refusal is the error worth reporting; a leftover
                // pending file is harmless and replaced by the next write.
                let _ = std::fs::remove_file(&pending);
                return Err(e);
            }
        },
        Format::SplinterExportV1 => None,
    };
    let digest = Digest::of(text.as_bytes());
    let manifest = canonical_json(&manifest(projection, format, &digest))?;
    std::fs::rename(&pending, path).map_err(io(path))?;
    let manifest_file = manifest_path(path);
    // Canonical JSON is UTF-8 by construction, so nothing is substituted.
    let manifest_text = String::from_utf8_lossy(&manifest);
    splinter_store::write_atomic(&manifest_file, &manifest_text).map_err(io(&manifest_file))?;
    Ok(Dataset {
        path: path.to_path_buf(),
        format,
        digest,
        records: projection.records.len(),
        trained_messages,
        manifest: Digest::of(&manifest),
    })
}

/// `record` as one line of `format`, refused when its shape does not serve
/// `objective`.
fn line(
    format: Format,
    objective: Objective,
    index: usize,
    record: &Record,
) -> Result<String, ViewError> {
    if !record.body.serves(objective) {
        return Err(ViewError::Shape {
            index,
            shape: record.body.shape(),
            objective,
        });
    }
    Ok(match (format, &record.body) {
        (Format::GenericMessagesV2, RecordBody::Chat { messages }) => {
            serde_json::to_string(&ChatLine {
                messages,
                metadata: &record.metadata,
            })?
        }
        // Only a chat record serves a trainable objective, so a chat
        // format line is always a chat record; anything else is exported.
        _ => serde_json::to_string(&ExportLine {
            format: EXPORT_FORMAT,
            body: &record.body,
            metadata: &record.metadata,
        })?,
    })
}

/// Brain's parser's verdict on the chat dataset written at `pending` for
/// `path`: its supervised message count, refused when it supervises
/// nothing.
fn validate(path: &Path, pending: &Path) -> Result<usize, ViewError> {
    let invalid = |reason: String| ViewError::Invalid {
        path: path.to_path_buf(),
        reason,
    };
    let summary =
        splinter_policy::train::validate_dataset(pending).map_err(|e| invalid(format!("{e:#}")))?;
    if summary.trained_messages == 0 {
        return Err(invalid("no message is supervised".into()));
    }
    Ok(summary.trained_messages)
}

fn manifest(projection: &Projection, format: Format, dataset: &Digest) -> Manifest {
    let mut experiences = FirstUse::default();
    let mut tasks = FirstUse::default();
    let mut sources = FirstUse::default();
    for metadata in projection.records.iter().map(|r| &r.metadata) {
        metadata
            .experiences
            .iter()
            .for_each(|id| experiences.add(id));
        metadata.task.iter().for_each(|task| tasks.add(task));
        metadata
            .sources
            .iter()
            .for_each(|source| sources.add(source));
    }
    Manifest {
        format,
        dataset: dataset.clone(),
        view: projection.view.clone(),
        objective: projection.objective,
        strip: projection.strip.clone(),
        min_strength: projection.min_strength,
        experiences: experiences.order,
        tasks: tasks.order,
        sources: sources.order,
        counts: Counts {
            records: projection.records.len(),
            excluded: projection.excluded.clone(),
        },
    }
}

/// Distinct values in the order they were first added.
struct FirstUse<T> {
    order: Vec<T>,
    seen: HashSet<T>,
}

impl<T> Default for FirstUse<T> {
    fn default() -> Self {
        Self {
            order: Vec::new(),
            seen: HashSet::new(),
        }
    }
}

impl<T: Clone + Eq + Hash> FirstUse<T> {
    fn add(&mut self, value: &T) {
        if self.seen.insert(value.clone()) {
            self.order.push(value.clone());
        }
    }
}
