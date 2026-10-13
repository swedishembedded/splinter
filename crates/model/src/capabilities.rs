// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gated self-improvement loops where a model
// update is adopted only on held-out evidence. If your team needs expertise
// in evaluation-gated fine-tuning, you can procure our services by sending
// an email to info@swedishembedded.com.

//! What the model backend trains, and the check of a dataset file against
//! its own parser.
//!
//! A view projects an objective's records and writes them in that
//! objective's line format; whether a trainer reads that format is the
//! backend's to say, so the answer lives here and not with the views.
//!
//! It also holds the length check of a chat dataset against the training row.

use std::path::{Path, PathBuf};

use splinter_data::{DatasetCheck, Format, Objective};

use crate::error::PolicyError;
use crate::train::{validate_dataset, validate_preference_dataset};

/// The dataset formats a backend trains from a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainingCapabilities {
    /// Supervised fine-tuning on chat records, which also trains
    /// classification rendered as SFT.
    pub chat_sft: bool,
    /// Preference optimisation on (chosen, rejected) pairs.
    pub preference: bool,
    /// Time-to-event training on subject timelines (`timeline-v1`).
    pub timeline: bool,
}

impl TrainingCapabilities {
    /// What brain's public SDK trains from a dataset file: chat fine-tuning,
    /// preference fine-tuning and subject timelines. It has no trainer that
    /// reads contrastive triples for the policy model, rewarded trajectories
    /// or a raw text corpus.
    pub const BRAIN: Self = Self {
        chat_sft: true,
        preference: true,
        timeline: true,
    };

    /// Whether a file in `format` can be trained from.
    #[must_use]
    pub fn supports(self, format: Format) -> bool {
        match format {
            Format::GenericMessagesV2 => self.chat_sft,
            Format::GenericPreferenceV1 => self.preference,
            Format::SplinterExportV1 => false,
            Format::TimelineV1 => self.timeline,
        }
    }

    /// The format `objective` is trained from, or why it cannot be.
    pub fn require(self, objective: Objective) -> Result<Format, PolicyError> {
        let format = objective.line_format();
        if self.supports(format) {
            Ok(format)
        } else {
            Err(PolicyError::ObjectiveNotTrainable { objective })
        }
    }
}

/// brain's own parsers as the check of a dataset file: the wire schema and
/// supervision boundaries of a chat file, which must supervise something,
/// and a preference file, which must be read whole.
#[derive(Clone, Copy, Debug, Default)]
pub struct BrainDatasetCheck;

impl DatasetCheck for BrainDatasetCheck {
    fn check(&self, format: Format, pending: &Path, records: usize) -> Result<(), String> {
        match format {
            Format::GenericMessagesV2 => {
                let summary = validate_dataset(pending).map_err(|e| format!("{e:#}"))?;
                if summary.trained_messages == 0 {
                    return Err("no message is supervised".into());
                }
                Ok(())
            }
            Format::GenericPreferenceV1 => {
                let summary = validate_preference_dataset(pending).map_err(|e| format!("{e:#}"))?;
                if summary.pairs == records {
                    Ok(())
                } else {
                    Err(format!(
                        "{records} pair(s) were written but brain's parser read {}",
                        summary.pairs
                    ))
                }
            }
            Format::SplinterExportV1 => Ok(()),
            Format::TimelineV1 => {
                let subjects = crate::timeline::read_jsonl(pending).map_err(|e| e.to_string())?;
                if subjects.len() == records {
                    Ok(())
                } else {
                    Err(format!(
                        "{records} subject(s) were written but brain's parser read {}",
                        subjects.len()
                    ))
                }
            }
        }
    }
}

/// A chat record too long for the training row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OverlongRecord {
    /// The dataset file it is in.
    pub dataset: PathBuf,
    /// Its 1-based line in the file.
    pub line: usize,
    /// What its metadata names it by (its first experience, else its task),
    /// when it has either.
    pub origin: Option<String>,
    /// Its length in tokens.
    pub tokens: usize,
}

impl std::fmt::Display for OverlongRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.dataset.display(), self.line)?;
        if let Some(origin) = &self.origin {
            write!(f, " ({origin})")?;
        }
        write!(f, " is {} tokens", self.tokens)
    }
}

/// The records of `dataset` longer than `max_block` tokens, in file order.
/// A record's length is its conversation, with the tools it offers, rendered
/// by brain's chat renderer and tokenized by the tokenizer of the checkpoint
/// in `model_dir`. The generation prompt the renderer appends can make it a
/// few tokens longer than the training row, never shorter, and the trainer's
/// own cap still decides what it accepts. Needs no device.
pub fn overlong_records(
    dataset: &Path,
    model_dir: &Path,
    max_block: u32,
) -> Result<Vec<OverlongRecord>, PolicyError> {
    let unreadable = |reason: String| PolicyError::Dataset {
        path: dataset.to_path_buf(),
        reason,
    };
    let tokenizer = brain::chat_tokens::ChatTokenizer::from_model_dir(model_dir)
        .map_err(|e| unreadable(e.to_string()))?;
    let text = std::fs::read_to_string(dataset).map_err(|e| unreadable(e.to_string()))?;
    let mut overlong = Vec::new();
    for (index, line) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let line_no = index + 1;
        let record: serde_json::Value =
            serde_json::from_str(line).map_err(|e| unreadable(format!("line {line_no}: {e}")))?;
        let request = chat_request_of(&record)
            .map_err(|reason| unreadable(format!("line {line_no}: {reason}")))?;
        let tokens = tokenizer
            .prompt_ids(&request)
            .map_err(|e| unreadable(format!("line {line_no}: {e}")))?
            .len();
        if tokens > max_block as usize {
            overlong.push(OverlongRecord {
                dataset: dataset.to_path_buf(),
                line: line_no,
                origin: origin_of(&record),
                tokens,
            });
        }
    }
    Ok(overlong)
}

/// A `generic-messages-v2` record as the request brain renders for its
/// conversation: its messages (an assistant turn's tool calls and a tool
/// result's call id included) and the tools it offers.
fn chat_request_of(record: &serde_json::Value) -> Result<brain::ChatRequest, String> {
    use brain::chat::{ToolCall, ToolSchema};
    let text = |v: &serde_json::Value, key: &str| {
        v.get(key)
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_owned()
    };
    let messages = record
        .get("messages")
        .and_then(serde_json::Value::as_array)
        .ok_or("no messages array")?
        .iter()
        .map(|m| {
            let content = text(m, "content");
            Ok(match text(m, "role").as_str() {
                "system" => brain::ChatMessage::system(content),
                "user" => brain::ChatMessage::user(content),
                "tool" => brain::ChatMessage::tool(text(m, "tool_call_id"), content),
                "assistant" => {
                    let calls = m
                        .get("tool_calls")
                        .and_then(serde_json::Value::as_array)
                        .map(|calls| {
                            calls
                                .iter()
                                .map(|c| {
                                    let function = c.get("function").unwrap_or(c);
                                    ToolCall {
                                        id: text(c, "id"),
                                        name: text(function, "name"),
                                        arguments: text(function, "arguments"),
                                    }
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    brain::ChatMessage::assistant(content).with_tool_calls(calls)
                }
                other => return Err(format!("unknown role {other:?}")),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let tools = record
        .get("tools")
        .and_then(serde_json::Value::as_array)
        .map(|tools| {
            tools
                .iter()
                .map(|t| {
                    let function = t.get("function").unwrap_or(t);
                    ToolSchema::new(
                        text(function, "name"),
                        text(function, "description"),
                        function.get("parameters").cloned().unwrap_or_default(),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(brain::ChatRequest::new(messages).tools(tools))
}

/// What a record's metadata names it by: its first experience, else its task.
fn origin_of(record: &serde_json::Value) -> Option<String> {
    let metadata = record.get("metadata")?;
    let first = metadata
        .get("experiences")
        .and_then(serde_json::Value::as_array)
        .and_then(|e| e.first());
    first
        .or_else(|| metadata.get("task"))
        .map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_owned))
}
