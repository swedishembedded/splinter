// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A projection written as a dataset file, with its manifest beside it.
//!
//! Each [`Objective`] maps to one file format:
//!
//! * SFT, and classification rendered as SFT, are written as brain's
//!   `generic-messages-v2` ([`Format::GenericMessagesV2`]): one
//!   conversation per line, `{"messages":[{"role","content","train"}...],
//!   "tools":[...],"metadata":{...}}`, where `train` marks what is
//!   supervised, `tools` (present only for a conversation held under tool
//!   schemas) is what the chat template renders into the prompt, and
//!   `metadata` is carried but never rendered. The file is written beside
//!   its destination, parsed with brain's own dataset parser (through
//!   `splinter-model`), and only then moved into place - so a dataset the
//!   trainer would refuse is never reported written.
//! * Preference pairs are written as brain's `generic-preference-v1`
//!   ([`Format::GenericPreferenceV1`]): one pair per line,
//!   `{"prompt":[messages],"chosen":C,"rejected":C,"tools":[...],
//!   "metadata":{...}}`, the messages in the `generic-messages-v2` shape
//!   without `train` - supervision follows from position: the prompt is
//!   never trained on, a candidate always is. A candidate `C` is one
//!   assistant turn (an object) or a continuation (an array of turns that
//!   opens and ends with an assistant turn: its assistant turns are trained
//!   on, the tool results between them are context).
//! * Judged completions (KTO) are written as brain's `generic-feedback-v1`
//!   ([`Format::GenericFeedbackV1`]): `{"prompt":[messages],"completion":C,
//!   "desirable":bool,"tools":[...],"metadata":{...}}`, `C` as above.
//!
//!   The attempts a pair or a completion comes from ([`crate::RolloutIdentity`])
//!   are written into its `metadata` as `rollout`: brain carries `metadata`
//!   and never reads it.
//! * A raw text corpus (continued pretraining) is written as chat lines: each
//!   text is one assistant turn with nothing before it, supervised whole, so
//!   the loss is the next-token loss over every token of the text.
//! * Contrastive triples and rewarded trajectories have no
//!   trainer in brain's public SDK. The writer refuses them with
//!   [`ViewError::ObjectiveNotTrainable`] unless the caller sets
//!   [`WriteOptions::export_only`], and then writes Splinter's export
//!   format ([`Format::SplinterExportV1`], [`EXPORT_FORMAT`]): one JSON
//!   object per line, `{"format":"splinter-export-v1","shape":<shape>,
//!   <the shape's fields>,"metadata":{...}}`, the shapes and their fields
//!   being [`RecordBody`]'s variants in snake case (`contrastive`: `query`,
//!   `positive`, `negatives`; `rewarded`: `messages`, `reward`). Messages are `generic-messages-v2` messages. No file brain
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

use crate::TeacherMessage;
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
    /// Brain's judged-completion format (KTO).
    #[serde(rename = "generic-feedback-v1")]
    GenericFeedbackV1,
    /// Splinter's export-only format, which no trainer reads.
    #[serde(rename = "splinter-export-v1")]
    SplinterExportV1,
    /// Brain's subject-timeline format: one subject's irregular history and
    /// its outcomes per line. Not a projection of experiences: a record
    /// dataset an application builds from its sources.
    #[serde(rename = "timeline-v1")]
    TimelineV1,
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
    /// The groups of attempts whose rewards the records carry (a
    /// policy-gradient dataset); empty for any other.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<crate::GroupReport>,
    /// How the trainer is to turn the records' rewards into advantages
    /// ([`crate::Projection::advantage`]); `None` for a dataset whose records
    /// carry no reward.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub advantage: Option<crate::AdvantageKind>,
    /// The system prompt the records open with when it is not the default;
    /// `None` when it is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// Whether the records are packed trajectories (see
    /// [`crate::Projection::packed`]), which training keeps as one sequence
    /// each; a manifest written before this existed is not packed.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub packed: bool,
    /// Whether every record comes from an experience an independent verdict
    /// passed (see [`crate::Projection::outcome_verified_only`]); a manifest
    /// written before this existed was not built that way.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub outcome_verified_only: bool,
    /// The terms the records' data came under, combined over their sources
    /// (the most restrictive of each axis, a source stating none counting as
    /// unknown); `None` when no source stated any, which consumers treat as
    /// unknown (a manifest written before terms existed keeps its address).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terms: Option<splinter_core::terms::Terms>,
    /// How the held-out and monitoring sets of the records are drawn, when
    /// they carry a draw ([`crate::holdout`]); `None` for records taken from
    /// the newest end (a manifest written before the split drew by seed
    /// keeps its address).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split: Option<crate::holdout::SplitPolicy>,
    /// The curation policy the records were built under and what it did with
    /// what it flagged ([`crate::curation`]); its counts are in `counts`. `None`
    /// for a dataset no policy curated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curation: Option<crate::curation::Curation>,
    /// How the passes of each task were curated
    /// ([`crate::curation::PassCuration`]); its counts are in `counts`. `None`
    /// for a dataset whose passes were not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pass_curation: Option<crate::curation::PassCuration>,
    /// DAPO's over-length penalty the records' rewards carry, with its
    /// thresholds ([`crate::Overlong`]); `None` for rewards without one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overlong: Option<crate::Overlong>,
    /// The partial credit a `shaped-partial` dataset's failed attempts earned,
    /// with the naive floors it was measured above and the groups that had
    /// none ([`crate::PartialCredit`]); `None` for any other reward.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_credit: Option<crate::PartialCredit>,
    /// How the pairs of a preference dataset were chosen, the seed and the
    /// pairs from each source ([`crate::PairingPolicy`]); `None` for any
    /// other dataset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pairing: Option<crate::PairingPolicy>,
    /// The classes of a feedback dataset and the weight of each
    /// ([`crate::FeedbackBalance`]); `None` for any other dataset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feedback: Option<crate::FeedbackBalance>,
    /// How many rewarded actions carry the tokens the policy sampled, and why
    /// the others do not ([`crate::TokenReport`]); empty for a dataset whose
    /// view attaches none.
    #[serde(default, skip_serializing_if = "crate::TokenReport::is_empty")]
    pub tokens: crate::TokenReport,
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

/// The tokenizers the lines of a dataset's `text` name in their
/// `metadata.rollout.tokenizer_digest` (pairs and judged completions from
/// grouped attempts), in first-use order: a dataset holds the attempts of one
/// tokenizer, so more than one is a dataset to refuse. A line that is not
/// JSON names none.
#[must_use]
pub fn named_tokenizers(text: &str) -> Vec<String> {
    let mut named: Vec<String> = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let digest = serde_json::from_str::<serde_json::Value>(line)
            .ok()
            .and_then(|v| {
                v.pointer("/metadata/rollout/tokenizer_digest")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string)
            });
        if let Some(digest) = digest.filter(|d| !named.contains(d)) {
            named.push(digest);
        }
    }
    named
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
    messages: Vec<ChatMessage<'a>>,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    tools: &'a [serde_json::Value],
    #[serde(skip_serializing_if = "Option::is_none")]
    reward: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    group: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rollout: Option<&'a str>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    baseline_only: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    teacher_context: Option<TeacherContextLine<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokenizer_digest: Option<&'a str>,
    metadata: &'a RecordMetadata,
}

/// A message of a chat line: the message, and for a supervised one the tokens
/// the policy sampled for it (brain's `token_ids`, `behavior_logprobs` and
/// `prompt_token_ids`).
#[derive(Serialize)]
struct ChatMessage<'a> {
    #[serde(flatten)]
    message: &'a splinter_core::chat::WireMessage,
    #[serde(flatten)]
    sampled: Option<&'a crate::SampledTurn>,
}

/// The messages of a chat line, each supervised one with its sampled turn:
/// `sampled` holds one entry per supervised message, in order.
fn chat_messages<'a>(
    index: usize,
    messages: &'a [splinter_core::chat::WireMessage],
    sampled: Option<&'a crate::SampledTurns>,
) -> Result<Vec<ChatMessage<'a>>, ViewError> {
    let supervised = messages.iter().filter(|m| m.train).count();
    if let Some(sampled) = sampled.filter(|s| s.turns.len() != supervised) {
        return Err(ViewError::Parameter {
            name: "sampled",
            reason: format!(
                "record {index} supervises {supervised} messages and holds sampled tokens for {}",
                sampled.turns.len()
            ),
        });
    }
    let mut turns = sampled.iter().flat_map(|s| s.turns.iter());
    Ok(messages
        .iter()
        .map(|message| ChatMessage {
            message,
            sampled: if message.train {
                turns.next().and_then(Option::as_ref)
            } else {
                None
            },
        })
        .collect())
}

/// The teacher context of a chat line as brain reads it: the privileged
/// turns and the index of the line's message they are inserted before.
#[derive(Serialize)]
struct TeacherContextLine<'a> {
    at: usize,
    messages: &'a [TeacherMessage],
}

/// One line of the preference format.
#[derive(Serialize)]
struct PreferenceLine<'a> {
    prompt: Vec<Turn<'a>>,
    chosen: Candidate<'a>,
    rejected: Candidate<'a>,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    tools: &'a [serde_json::Value],
    metadata: LineMetadata<'a>,
}

/// One line of the judged-completion format.
#[derive(Serialize)]
struct FeedbackLine<'a> {
    prompt: Vec<Turn<'a>>,
    completion: Candidate<'a>,
    desirable: bool,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    tools: &'a [serde_json::Value],
    metadata: LineMetadata<'a>,
}

/// A record's metadata as a pair or completion line carries it: the record's
/// own, and the attempts it comes from.
#[derive(Serialize)]
struct LineMetadata<'a> {
    #[serde(flatten)]
    record: &'a RecordMetadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    rollout: Option<&'a crate::RolloutIdentity>,
}

/// A candidate of a pair or a judged completion: one turn is written as the
/// turn itself (the form every earlier pair has), a continuation as the
/// array of its turns.
#[derive(Serialize)]
#[serde(untagged)]
enum Candidate<'a> {
    Turn(Turn<'a>),
    Continuation(Vec<Turn<'a>>),
}

impl<'a> Candidate<'a> {
    fn of(turns: &'a [splinter_core::chat::WireMessage]) -> Self {
        match turns {
            [one] => Self::Turn(one.into()),
            many => Self::Continuation(many.iter().map(Turn::from).collect()),
        }
    }
}

/// A `generic-messages-v2` message without `train`, as a preference or
/// feedback line carries it.
#[derive(Serialize)]
struct Turn<'a> {
    role: &'a str,
    content: &'a str,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    tool_calls: &'a [splinter_core::chat::WireToolCall],
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<&'a str>,
}

impl<'a> From<&'a splinter_core::chat::WireMessage> for Turn<'a> {
    fn from(message: &'a splinter_core::chat::WireMessage) -> Self {
        Self {
            role: &message.role,
            content: &message.content,
            tool_calls: &message.tool_calls,
            tool_call_id: message.tool_call_id.as_deref(),
            reasoning_content: message.reasoning_content.as_deref(),
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
        let excluded = projection
            .excluded
            .iter()
            .map(|(reason, n)| format!("{reason:?}: {n}"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(ViewError::Empty { excluded });
    }
    let mut text = String::new();
    for (index, record) in projection.records.iter().enumerate() {
        text.push_str(&line(format, objective, index, record)?);
        text.push('\n');
    }
    let trained_messages = (format == Format::GenericMessagesV2).then(|| {
        projection
            .records
            .iter()
            .map(|record| match &record.body {
                RecordBody::Chat { messages, .. } => messages.iter().filter(|m| m.train).count(),
                // A text is written as one supervised assistant turn.
                RecordBody::Text { .. } => 1,
                _ => 0,
            })
            .sum()
    });
    let (digest, manifest) = place_dataset(
        path,
        &text,
        format,
        projection.records.len(),
        check,
        |digest| Ok(canonical_json(&manifest(projection, format, digest))?),
    )?;
    Ok(Dataset {
        path: path.to_path_buf(),
        format,
        digest,
        records: projection.records.len(),
        trained_messages,
        manifest,
    })
}

/// Places `text`, holding `records` records of `format`, at `path`, and the
/// manifest `manifest` makes from the text's digest beside it: the file is
/// written beside its destination, `check` reads it as the trainer will, and
/// only then is it moved into place, then the manifest. A refusal leaves
/// nothing at `path`. Returns the digests of the file and of its manifest.
pub(crate) fn place_dataset(
    path: &Path,
    text: &str,
    format: Format,
    records: usize,
    check: &dyn DatasetCheck,
    manifest: impl FnOnce(&Digest) -> Result<Vec<u8>, ViewError>,
) -> Result<(Digest, Digest), ViewError> {
    let io = |path: &Path| {
        let path = path.to_path_buf();
        move |source| ViewError::Io { path, source }
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(io(parent))?;
    }
    let pending = path.with_extension("pending");
    std::fs::write(&pending, text).map_err(io(&pending))?;
    if let Err(reason) = check.check(format, &pending, records) {
        // The refusal is the error worth reporting; a leftover pending file
        // is harmless and replaced by the next write.
        let _ = std::fs::remove_file(&pending);
        return Err(invalid(path, reason));
    }
    let digest = Digest::of(text.as_bytes());
    let manifest = match manifest(&digest) {
        Ok(manifest) => manifest,
        Err(error) => {
            let _ = std::fs::remove_file(&pending);
            return Err(error);
        }
    };
    std::fs::rename(&pending, path).map_err(io(path))?;
    let manifest_file = manifest_path(path);
    // Canonical JSON is UTF-8 by construction, so nothing is substituted.
    let manifest_text = String::from_utf8_lossy(&manifest);
    std::fs::write(&manifest_file, manifest_text.as_bytes()).map_err(io(&manifest_file))?;
    Ok((digest, Digest::of(&manifest)))
}

/// `record` as the line [`write_dataset`] writes for a projection of
/// `objective`: the text a trainer reads, for a caller that sizes a record as
/// the trainer will. Refused when the record's shape does not serve
/// `objective`.
pub fn dataset_line(objective: Objective, record: &Record) -> Result<String, ViewError> {
    line(objective.line_format(), objective, 0, record)
}

/// The teacher context of a chat line: the privileged turns go right after
/// the conversation's first user turn, so every answer of the line is scored
/// with them in view. A conversation with no user turn has nowhere to put
/// them and is refused rather than guessed.
fn teacher_line<'a>(
    index: usize,
    messages: &[splinter_core::chat::WireMessage],
    context: &'a [TeacherMessage],
) -> Result<Option<TeacherContextLine<'a>>, ViewError> {
    if context.is_empty() {
        return Ok(None);
    }
    let first_user =
        messages
            .iter()
            .position(|m| m.role == "user")
            .ok_or(ViewError::Parameter {
                name: "teacher_context",
                reason: format!(
                    "record {index} has a teacher context and no user turn to place it after"
                ),
            })?;
    Ok(Some(TeacherContextLine {
        at: first_user + 1,
        messages: context,
    }))
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
        (
            Format::GenericMessagesV2,
            RecordBody::Chat {
                messages,
                tools,
                outcome,
                teacher_context,
            },
        ) => serde_json::to_string(&ChatLine {
            messages: chat_messages(
                index,
                messages,
                outcome.as_ref().and_then(|o| o.sampled.as_ref()),
            )?,
            tokenizer_digest: outcome
                .as_ref()
                .and_then(|o| o.sampled.as_ref())
                .map(|s| s.tokenizer_digest.as_str()),
            tools,
            reward: outcome.as_ref().map(|o| o.reward),
            group: outcome.as_ref().map(|o| o.group.as_str()),
            rollout: outcome.as_ref().and_then(|o| o.rollout.as_deref()),
            baseline_only: outcome.as_ref().is_some_and(|o| o.baseline_only),
            teacher_context: teacher_line(index, messages, teacher_context)?,
            metadata: &record.metadata,
        })?,
        (
            Format::GenericPreferenceV1,
            RecordBody::Preference {
                prompt,
                chosen,
                rejected,
                tools,
                identity,
            },
        ) => serde_json::to_string(&PreferenceLine {
            prompt: prompt.iter().map(Turn::from).collect(),
            chosen: Candidate::of(chosen),
            rejected: Candidate::of(rejected),
            tools,
            metadata: LineMetadata {
                record: &record.metadata,
                rollout: identity.as_ref(),
            },
        })?,
        (
            Format::GenericFeedbackV1,
            RecordBody::Feedback {
                prompt,
                completion,
                desirable,
                tools,
                identity,
            },
        ) => serde_json::to_string(&FeedbackLine {
            prompt: prompt.iter().map(Turn::from).collect(),
            completion: Candidate::of(completion),
            desirable: *desirable,
            tools,
            metadata: LineMetadata {
                record: &record.metadata,
                rollout: Some(identity),
            },
        })?,
        // Continued pretraining on a text: the text is one assistant turn
        // with nothing before it, supervised whole, which is the next-token
        // loss over every token of it. The chat template frames the turn
        // exactly as it frames any answer.
        (Format::GenericMessagesV2, RecordBody::Text { text }) => {
            let message = splinter_core::chat::WireMessage {
                role: "assistant".into(),
                content: text.clone(),
                images: Vec::new(),
                tool_calls: Vec::new(),
                tool_call_id: None,
                reasoning_content: None,
                train: true,
                loss_weight: None,
            };
            serde_json::to_string(&ChatLine {
                messages: chat_messages(index, std::slice::from_ref(&message), None)?,
                tokenizer_digest: None,
                tools: &[],
                reward: None,
                group: None,
                rollout: None,
                baseline_only: false,
                teacher_context: None,
                metadata: &record.metadata,
            })?
        }
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
        groups: projection.groups.clone(),
        advantage: projection.advantage,
        system_prompt: projection.system_prompt.clone(),
        packed: projection.packed,
        outcome_verified_only: projection.outcome_verified_only,
        terms: projection.terms.clone(),
        split: projection.split_policy(),
        curation: projection.curation.clone(),
        pass_curation: projection.pass_curation,
        overlong: projection.overlong,
        partial_credit: projection.partial_credit.clone(),
        pairing: projection.pairing.clone(),
        feedback: projection.feedback,
        tokens: projection.tokens.clone(),
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
