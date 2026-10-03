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
//!   `splinter-model`), and only then moved into place - so a dataset the
//!   trainer would refuse is never reported written.
//! * DPO preference pairs are written as brain's `generic-preference-v1`
//!   ([`Format::GenericPreferenceV1`]): one pair per line,
//!   `{"prompt":[messages],"chosen":{message},"rejected":{message},
//!   "metadata":{...}}`, the messages in the `generic-messages-v2` shape
//!   without `train` - supervision follows from position: the prompt is
//!   never trained on, the two candidates always are. It is checked by
//!   brain's own preference parser the same way before it is moved into
//!   place.
//! * Contrastive triples, rewarded trajectories and raw text have no
//!   trainer in brain's public SDK. The writer refuses them with
//!   [`ViewError::ObjectiveNotTrainable`] unless the caller sets
//!   [`WriteOptions::export_only`], and then writes Splinter's export
//!   format ([`Format::SplinterExportV1`], [`EXPORT_FORMAT`]): one JSON
//!   object per line, `{"format":"splinter-export-v1","shape":<shape>,
//!   <the shape's fields>,"metadata":{...}}`, the shapes and their fields
//!   being [`RecordBody`]'s variants in snake case (`contrastive`: `query`,
//!   `positive`, `negatives`; `rewarded`: `messages`, `reward`; `text`:
//!   `text`). Messages are `generic-messages-v2` messages. No file brain
//!   would misread is ever written: the export format is neither of
//!   brain's, and brain's parsers refuse it.
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
use splinter_core::annotation::Strength;
use splinter_core::digest::{canonical_json, Digest};
use splinter_core::experience::ExperienceId;

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
    /// Brain's preference pair format.
    #[serde(rename = "generic-preference-v1")]
    GenericPreferenceV1,
    /// Splinter's export-only format, which no trainer reads.
    #[serde(rename = "splinter-export-v1")]
    SplinterExportV1,
}

/// A backend's verdict on a dataset file before it is put in place: the
/// parser that will read it at training time, run early so a file it would
/// refuse is never stored.
pub trait DatasetCheck {
    /// `Ok` when the file at `pending`, holding `records` records in
    /// `format`, is one the backend reads whole; the reason it is not
    /// otherwise.
    fn check(&self, format: Format, pending: &Path, records: usize) -> Result<(), String>;
}

/// The check that accepts every file: for a caller with no backend to
/// consult.
#[derive(Clone, Copy, Debug, Default)]
pub struct Unchecked;

impl DatasetCheck for Unchecked {
    fn check(&self, _: Format, _: &Path, _: usize) -> Result<(), String> {
        Ok(())
    }
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
    /// Supervised messages across them; `None` for a file that is not a chat
    /// dataset.
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

/// One line of the chat format.
#[derive(Serialize)]
struct ChatLine<'a> {
    messages: &'a [splinter_core::chat::WireMessage],
    metadata: &'a RecordMetadata,
}

/// One line of the preference format.
#[derive(Serialize)]
struct PreferenceLine<'a> {
    prompt: Vec<Turn<'a>>,
    chosen: Turn<'a>,
    rejected: Turn<'a>,
    metadata: &'a RecordMetadata,
}

/// A `generic-messages-v2` message without `train`, as a preference line
/// carries it.
#[derive(Serialize)]
struct Turn<'a> {
    role: &'a str,
    content: &'a str,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    tool_calls: &'a [splinter_core::chat::WireToolCall],
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a str>,
}

impl<'a> From<&'a splinter_core::chat::WireMessage> for Turn<'a> {
    fn from(message: &'a splinter_core::chat::WireMessage) -> Self {
        Self {
            role: &message.role,
            content: &message.content,
            tool_calls: &message.tool_calls,
            tool_call_id: message.tool_call_id.as_deref(),
        }
    }
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
/// to ([`Objective::line_format`]), and its manifest beside it. Refuses an
/// empty projection, a record whose shape does not serve the objective, and
/// a dataset `check` refuses; in every case nothing is left at `path`.
pub fn write_dataset(
    path: &Path,
    projection: &Projection,
    check: &dyn DatasetCheck,
) -> Result<Dataset, ViewError> {
    let objective = projection.objective;
    let format = objective.line_format();
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
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(io(parent))?;
    }
    let pending = path.with_extension("pending");
    std::fs::write(&pending, &text).map_err(io(&pending))?;
    if let Err(reason) = check.check(format, &pending, projection.records.len()) {
        // The refusal is the error worth reporting; a leftover pending file
        // is harmless and replaced by the next write.
        let _ = std::fs::remove_file(&pending);
        return Err(invalid(path, reason));
    }
    let trained_messages = (format == Format::GenericMessagesV2).then(|| {
        projection
            .records
            .iter()
            .map(|record| match &record.body {
                RecordBody::Chat { messages } => messages.iter().filter(|m| m.train).count(),
                _ => 0,
            })
            .sum()
    });
    let digest = Digest::of(text.as_bytes());
    let manifest = canonical_json(&manifest(projection, format, &digest))?;
    std::fs::rename(&pending, path).map_err(io(path))?;
    let manifest_file = manifest_path(path);
    // Canonical JSON is UTF-8 by construction, so nothing is substituted.
    let manifest_text = String::from_utf8_lossy(&manifest);
    std::fs::write(&manifest_file, manifest_text.as_bytes()).map_err(io(&manifest_file))?;
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
        (
            Format::GenericPreferenceV1,
            RecordBody::Preference {
                prompt,
                chosen,
                rejected,
            },
        ) => serde_json::to_string(&PreferenceLine {
            prompt: prompt.iter().map(Turn::from).collect(),
            chosen: chosen.into(),
            rejected: rejected.into(),
            metadata: &record.metadata,
        })?,
        // A record serving an objective brain trains is written in that
        // objective's format, so the shapes above are the only ones brain's
        // formats see; anything else is exported.
        _ => serde_json::to_string(&ExportLine {
            format: EXPORT_FORMAT,
            body: &record.body,
            metadata: &record.metadata,
        })?,
    })
}

fn invalid(path: &Path, reason: String) -> ViewError {
    ViewError::Invalid {
        path: path.to_path_buf(),
        reason,
    }
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
