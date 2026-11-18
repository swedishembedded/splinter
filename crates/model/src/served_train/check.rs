// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-data pipelines that check a dataset
// against the model that will train on it before any card is claimed, for its
// clients. If your team needs expertise in validating and sizing fine-tuning
// data against a served model, you can procure our services by sending an
// email to info@swedishembedded.com.

//! A dataset read by the base that will train on it, without training: brain's
//! `check_dataset` action of `brain/train`.
//!
//! The service reads the file with the base's own tokenizer and chat template,
//! exactly as its trainer would, and reports what a run would see: the records
//! it refuses (by line, with the rule of the training-record contract each
//! broke), the examples and their lengths in the base's tokens, and the
//! examples longer than the training row. With [`CheckSpec::per_record`] it
//! also lists every record's own length, a page at a time; [`check_dataset`]
//! reads every page, so the report holds every record.
//!
//! Nothing here tokenizes: a length is the service's count, the one the
//! trainer refuses a record by.

use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use splinter_core::digest::Digest;

use crate::endpoint::BrainEndpoints;
use crate::error::PolicyError;

/// The most records one page of a per-record listing asks for: brain's own
/// bound on a page.
pub const PAGE: usize = 200_000;

/// The record format a dataset is read as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DatasetKind {
    /// `generic-messages-v2` chat records: supervised, policy-gradient and
    /// full-parameter training.
    Chat,
    /// `generic-preference-v1` pairs: preference optimisation.
    Preference,
    /// `generic-feedback-v1` judged completions: KTO.
    Feedback,
    /// `timeline-v1` subjects: each line validated, no token counted.
    Timeline,
}

impl DatasetKind {
    fn name(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Preference => "preference",
            Self::Feedback => "feedback",
            Self::Timeline => "timeline",
        }
    }
}

/// What a dataset is checked for.
#[derive(Clone, Debug)]
pub struct CheckSpec<'a> {
    /// The base whose tokenizer and template read the records, as the model
    /// store names it (`Qwen/Qwen3-4B`); a timeline dataset has none.
    pub base: Option<&'a str>,
    /// How the records are read.
    pub kind: DatasetKind,
    /// Rendered as a run that trains the model to reason would render it.
    pub thinking: bool,
    /// A record that supervises several turns is one example, not one per
    /// turn.
    pub pack_answers: bool,
    /// The training row: longer examples are counted and their lines named.
    pub max_block: u32,
    /// Also list every record's own length.
    pub per_record: bool,
}

impl<'a> CheckSpec<'a> {
    /// A chat check against `base` at brain's widest row, listing every
    /// record.
    #[must_use]
    pub fn chat(base: &'a str, thinking: bool, pack_answers: bool) -> Self {
        Self {
            base: Some(base),
            kind: DatasetKind::Chat,
            thinking,
            pack_answers,
            max_block: MAX_BLOCK,
            per_record: true,
        }
    }
}

/// The widest training row brain takes, in tokens.
pub const MAX_BLOCK: u32 = 1_048_576;

/// A spread of per-example token counts.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Distribution {
    /// Summed over the examples.
    pub total: u64,
    /// The shortest example.
    pub min: u64,
    /// The mean example.
    pub mean: f64,
    /// The median example (nearest rank).
    pub p50: u64,
    /// The 90th percentile.
    pub p90: u64,
    /// The 99th percentile.
    pub p99: u64,
    /// The longest example: the row a run over the dataset needs.
    pub max: u64,
}

/// A record the trainer refuses.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct Refusal {
    /// Its 1-based line in the file.
    pub line: usize,
    /// Why, in brain's words.
    pub error: String,
    /// The rule of the training-record contract it broke, when it broke one.
    #[serde(default)]
    pub rule: Option<String>,
    /// The record field the rule names, as a JSON path.
    #[serde(default)]
    pub field: Option<String>,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.error)
    }
}

/// One valid record's length, in the base's tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct RecordLength {
    /// Its 1-based line in the file.
    pub line: usize,
    /// Its longest example: the row it needs.
    pub tokens: usize,
    /// Of that example, the tokens read but not trained on.
    pub prompt: usize,
    /// Of that example, the supervised tokens.
    pub trained: usize,
    /// The examples the record trains as.
    pub examples: usize,
}

/// What the service found in a dataset.
#[derive(Clone, Debug, PartialEq)]
pub struct DatasetReport {
    /// Non-blank lines.
    pub records: usize,
    /// Records that parse and encode.
    pub valid_records: usize,
    /// What the valid records train as.
    pub examples: usize,
    /// Per example, in the base's tokens.
    pub tokens: Distribution,
    /// Per example, the supervised tokens.
    pub supervised_tokens: Distribution,
    /// Examples that would train on nothing.
    pub examples_without_supervision: usize,
    /// Examples longer than the row checked against.
    pub over_max_block: usize,
    /// The first lines holding one.
    pub over_max_block_lines: Vec<usize>,
    /// Records the trainer refuses.
    pub invalid_total: usize,
    /// The first of them.
    pub invalid: Vec<Refusal>,
    /// With [`CheckSpec::per_record`]: every valid record's length, in file
    /// order.
    pub per_record: Vec<RecordLength>,
    /// With [`CheckSpec::per_record`]: every refused record, in file order
    /// (`invalid` lists the first of them only).
    pub refused: Vec<Refusal>,
}

impl DatasetReport {
    /// The longest example in tokens: the row a run over the dataset needs.
    #[must_use]
    pub fn longest(&self) -> u32 {
        u32::try_from(self.tokens.max).unwrap_or(u32::MAX)
    }

    /// The tokens of the record on `line`, when it is valid and was listed.
    #[must_use]
    pub fn tokens_at(&self, line: usize) -> Option<usize> {
        self.per_record
            .binary_search_by_key(&line, |r| r.line)
            .ok()
            .map(|i| self.per_record[i].tokens)
    }

    /// `Err` naming the refused records when any record is refused.
    pub fn refuse_invalid(&self, dataset: &Path) -> Result<(), PolicyError> {
        if self.invalid_total == 0 {
            return Ok(());
        }
        let named: Vec<String> = self.invalid.iter().map(ToString::to_string).collect();
        Err(PolicyError::Dataset {
            path: dataset.to_path_buf(),
            reason: format!(
                "{} record(s) the trainer refuses: {}",
                self.invalid_total,
                named.join("; ")
            ),
        })
    }
}

/// The page of a report the service answers.
#[derive(Deserialize)]
struct Page {
    records: usize,
    valid_records: usize,
    examples: usize,
    tokens: Distribution,
    supervised_tokens: Distribution,
    examples_without_supervision: usize,
    over_max_block: usize,
    over_max_block_lines: Vec<usize>,
    invalid_total: usize,
    invalid: Vec<Refusal>,
    #[serde(default)]
    per_record: Vec<(usize, usize, usize, usize, usize)>,
    #[serde(default)]
    per_record_refused: Vec<Refusal>,
    #[serde(default)]
    per_record_next: Option<usize>,
}

/// Checks `dataset` as `spec` says against the service `brain`. The dataset
/// is published to the artifact store and named by its digest.
///
/// # Errors
/// No artifact store is configured, the service refused the check (an
/// unknown base, say), or it answered something that is not a report.
pub fn check_dataset(
    brain: &BrainEndpoints,
    dataset: &Path,
    spec: &CheckSpec<'_>,
) -> Result<DatasetReport, PolicyError> {
    let digest = brain.artifact_store()?.publish(dataset)?;
    check_published(brain, &digest, dataset, spec)
}

/// [`check_dataset`] of a dataset already in the artifact store, named by
/// `digest`; `label` names it in an error.
pub fn check_published(
    brain: &BrainEndpoints,
    digest: &Digest,
    label: &Path,
    spec: &CheckSpec<'_>,
) -> Result<DatasetReport, PolicyError> {
    let dataset = label;
    let mut params = Map::new();
    if let Some(base) = spec.base {
        params.insert("base".into(), json!(base));
    }
    params.insert("kind".into(), json!(spec.kind.name()));
    params.insert("dataset_digest".into(), json!(digest.hex()));
    if spec.kind == DatasetKind::Chat {
        params.insert("thinking".into(), json!(spec.thinking));
        params.insert("pack_answers".into(), json!(spec.pack_answers));
    }
    params.insert("max_block".into(), json!(spec.max_block));
    // A timeline dataset counts no tokens, so it has no lengths to list.
    let listed = spec.per_record && spec.kind != DatasetKind::Timeline;
    params.insert("per_record".into(), json!(listed));
    let mut report: Option<DatasetReport> = None;
    let mut offset = 0usize;
    loop {
        if listed {
            params.insert("offset".into(), json!(offset));
            params.insert("limit".into(), json!(PAGE));
        }
        let outputs = super::run_action(
            brain,
            dataset,
            offset as u64,
            "check_dataset",
            &Value::Object(params.clone()),
            None,
            &mut |_| {},
        )?;
        let page: Page = serde_json::from_value(outputs).map_err(|e| PolicyError::Endpoint {
            url: brain.urls().join(", "),
            reason: format!("check_dataset answered no report: {e}"),
        })?;
        let lengths = page
            .per_record
            .iter()
            .map(|&(line, tokens, prompt, trained, examples)| RecordLength {
                line,
                tokens,
                prompt,
                trained,
                examples,
            });
        let next = page.per_record_next;
        match report.as_mut() {
            Some(kept) => {
                kept.per_record.extend(lengths);
                kept.refused.extend(page.per_record_refused);
            }
            None => {
                report = Some(DatasetReport {
                    records: page.records,
                    valid_records: page.valid_records,
                    examples: page.examples,
                    tokens: page.tokens,
                    supervised_tokens: page.supervised_tokens,
                    examples_without_supervision: page.examples_without_supervision,
                    over_max_block: page.over_max_block,
                    over_max_block_lines: page.over_max_block_lines,
                    invalid_total: page.invalid_total,
                    invalid: page.invalid,
                    per_record: lengths.collect(),
                    refused: page.per_record_refused,
                });
            }
        }
        match next {
            Some(next) if listed && next > offset => offset = next,
            _ => break,
        }
    }
    report.ok_or_else(|| PolicyError::Endpoint {
        url: brain.urls().join(", "),
        reason: "check_dataset answered nothing".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_is_found_by_its_line_and_a_refusal_names_its_line() {
        let report = DatasetReport {
            records: 3,
            valid_records: 2,
            examples: 2,
            tokens: Distribution {
                max: 9,
                ..Distribution::default()
            },
            supervised_tokens: Distribution::default(),
            examples_without_supervision: 0,
            over_max_block: 0,
            over_max_block_lines: vec![],
            invalid_total: 1,
            invalid: vec![Refusal {
                line: 2,
                error: "not JSON".into(),
                rule: None,
                field: None,
            }],
            per_record: vec![
                RecordLength {
                    line: 1,
                    tokens: 5,
                    prompt: 3,
                    trained: 2,
                    examples: 1,
                },
                RecordLength {
                    line: 3,
                    tokens: 9,
                    prompt: 4,
                    trained: 5,
                    examples: 1,
                },
            ],
            refused: vec![],
        };
        assert_eq!(report.tokens_at(3), Some(9));
        assert_eq!(report.tokens_at(2), None);
        assert_eq!(report.longest(), 9);
        let error = report
            .refuse_invalid(Path::new("d.jsonl"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("line 2: not JSON"), "{error}");
    }
}
