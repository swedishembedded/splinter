// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning from the sessions a person holds
// with an agent, for its clients. If your team needs expertise in
// contamination-free evaluation of fine-tuned models, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The guard that keeps sealed probes out of training.
//!
//! A probe is a question written before training, with the answer it is
//! measured against, and sealed: nothing that reads sessions, writes tasks
//! or trains may have seen it. [`SealedProbes::check`] is the check that a
//! record is clean. A record is refused when it contains a probe's question,
//! or shares a run of [`GRAM_WORDS`] words with a probe (its question or its
//! answer) that the claim the record teaches does not itself contain: the
//! claim's own statement is what the model is to learn, so the words it
//! shares with its probe are the point; any other shared run means the
//! probe's wording reached training by another road.
//!
//! A probe file is JSON Lines, or one JSON array, of objects with a
//! `question`, optionally a `reference` (the answer) and a `name` to be
//! called by; a probe without a name is called by its file and position.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use splinter_data::{Projection, Record, RecordBody};
use splinter_orchestrator::error::{io, OrchestratorError};

/// The words in a run that two texts must share to count as the same text.
pub const GRAM_WORDS: usize = 8;

/// A probe as its file writes it.
#[derive(Deserialize)]
struct ProbeLine {
    question: String,
    #[serde(default)]
    reference: String,
    #[serde(default)]
    name: Option<String>,
}

/// One sealed probe.
#[derive(Clone, Debug)]
struct Probe {
    name: String,
    question: String,
    grams: BTreeSet<Vec<String>>,
}

/// Why a record was refused.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
pub enum LeakKind {
    /// The record contains the probe's question.
    Question,
    /// The record shares a run of words with the probe that its claim does
    /// not contain.
    Overlap {
        /// The shared run.
        words: String,
    },
}

/// A record that contains a probe, and which.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Leak {
    /// The probe, by its name.
    pub probe: String,
    /// What was shared.
    #[serde(flatten)]
    pub kind: LeakKind,
}

impl std::fmt::Display for Leak {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            LeakKind::Question => write!(
                f,
                "contains the question of the sealed probe {}",
                self.probe
            ),
            LeakKind::Overlap { words } => write!(
                f,
                "shares {words:?} with the sealed probe {} beyond what its claim states",
                self.probe
            ),
        }
    }
}

/// The sealed probes a run is held to.
#[derive(Clone, Debug, Default)]
pub struct SealedProbes {
    probes: Vec<Probe>,
}

/// The lower-cased words of `text`, runs of letters and digits.
fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn grams(text: &str) -> BTreeSet<Vec<String>> {
    let words = words(text);
    words.windows(GRAM_WORDS).map(<[String]>::to_vec).collect()
}

impl SealedProbes {
    /// The probes `files` hold; refused, naming the file and line, when one
    /// is not a probe.
    pub fn load(files: &[PathBuf]) -> Result<Self, OrchestratorError> {
        let mut probes = Vec::new();
        for file in files {
            let text = std::fs::read_to_string(file).map_err(io(file))?;
            for (position, line) in parse(file, &text)?.into_iter().enumerate() {
                let name = line
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("{}:{}", file.display(), position + 1));
                probes.push(Self::probe(name, &line.question, &line.reference));
            }
        }
        Ok(Self { probes })
    }

    /// The probes of `(name, question, reference)`.
    #[must_use]
    pub fn of<'a>(probes: impl IntoIterator<Item = (&'a str, &'a str, &'a str)>) -> Self {
        Self {
            probes: probes
                .into_iter()
                .map(|(name, question, reference)| Self::probe(name.into(), question, reference))
                .collect(),
        }
    }

    fn probe(name: String, question: &str, reference: &str) -> Probe {
        Probe {
            name,
            question: words(question).join(" "),
            grams: grams(&format!("{question}\n{reference}")),
        }
    }

    /// Whether there is no probe to hold anything to.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.probes.is_empty()
    }

    /// Why a record made of `texts` is refused, or `None` when it is clean.
    /// `allowed` are the texts of the claim the record teaches: runs of words
    /// they contain are not held against it.
    #[must_use]
    pub fn check(&self, texts: &[&str], allowed: &[&str]) -> Option<Leak> {
        if self.probes.is_empty() {
            return None;
        }
        let joined: Vec<String> = texts
            .iter()
            .map(|t| format!(" {} ", words(t).join(" ")))
            .collect();
        let allowed: BTreeSet<Vec<String>> = allowed.iter().flat_map(|t| grams(t)).collect();
        let record: BTreeSet<Vec<String>> = texts.iter().flat_map(|t| grams(t)).collect();
        for probe in &self.probes {
            if !probe.question.is_empty()
                && joined
                    .iter()
                    .any(|j| j.contains(&format!(" {} ", probe.question)))
            {
                return Some(Leak {
                    probe: probe.name.clone(),
                    kind: LeakKind::Question,
                });
            }
            if let Some(shared) = probe
                .grams
                .iter()
                .find(|g| record.contains(*g) && !allowed.contains(*g))
            {
                return Some(Leak {
                    probe: probe.name.clone(),
                    kind: LeakKind::Overlap {
                        words: shared.join(" "),
                    },
                });
            }
        }
        None
    }

    /// Removes from `projection` every record that contains a probe,
    /// holding each to the claim texts `allowed` returns for it; the
    /// records removed are returned with why. The system turn is the same
    /// in every record and is not part of what a record teaches.
    pub fn remove_leaks(
        &self,
        projection: &mut Projection,
        allowed: impl Fn(&Record) -> Vec<String>,
    ) -> Vec<(Record, Leak)> {
        let mut refused = Vec::new();
        let records = std::mem::take(&mut projection.records);
        for record in records {
            let leak = {
                let texts = record_texts(&record);
                let allowed = allowed(&record);
                let allowed: Vec<&str> = allowed.iter().map(String::as_str).collect();
                self.check(&texts, &allowed)
            };
            match leak {
                Some(leak) => refused.push((record, leak)),
                None => projection.records.push(record),
            }
        }
        refused
    }

    /// The first line of the JSON Lines dataset `path` that contains a
    /// probe, with its position (from one), or `None`: for the records a
    /// run mixes in that it did not write itself.
    pub fn check_file(&self, path: &Path) -> Result<Option<(usize, Leak)>, OrchestratorError> {
        let text = std::fs::read_to_string(path).map_err(io(path))?;
        for (position, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
            let value: serde_json::Value =
                serde_json::from_str(line).map_err(|source| OrchestratorError::Json {
                    what: format!("record {} of {}", position + 1, path.display()),
                    source,
                })?;
            let turns: Vec<&str> = value["messages"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|m| m["role"].as_str() != Some("system"))
                .filter_map(|m| m["content"].as_str())
                .collect();
            if let Some(leak) = self.check(&turns, &[]) {
                return Ok(Some((position + 1, leak)));
            }
        }
        Ok(None)
    }
}

/// What a record teaches, as texts: every turn but the system one.
fn record_texts(record: &Record) -> Vec<&str> {
    match &record.body {
        RecordBody::Chat { messages } | RecordBody::Rewarded { messages, .. } => messages
            .iter()
            .filter(|m| m.role != "system")
            .map(|m| m.content.as_str())
            .collect(),
        RecordBody::Preference {
            prompt,
            chosen,
            rejected,
        } => prompt
            .iter()
            .filter(|m| m.role != "system")
            .chain([chosen, rejected])
            .map(|m| m.content.as_str())
            .collect(),
        RecordBody::Text { text } => vec![text.as_str()],
        RecordBody::Contrastive {
            query,
            positive,
            negatives,
        } => [query, positive]
            .into_iter()
            .chain(negatives)
            .map(String::as_str)
            .collect(),
    }
}

fn parse(file: &Path, text: &str) -> Result<Vec<ProbeLine>, OrchestratorError> {
    let refuse = |at: String, why: String| {
        OrchestratorError::Refused(format!("sealed probes {}{at}: {why}", file.display()))
    };
    if text.trim_start().starts_with('[') {
        return serde_json::from_str(text).map_err(|e| refuse(String::new(), e.to_string()));
    }
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(n, line)| {
            serde_json::from_str(line)
                .map_err(|e| refuse(format!(" line {}", n + 1), e.to_string()))
        })
        .collect()
}
