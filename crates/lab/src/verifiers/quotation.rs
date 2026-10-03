// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifiers that hold a model's quotations to
// the source for its clients. If your team needs expertise in catching a
// model that invents what a document says, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The quotation verifier: every passage an answer presents in quotation marks
//! is in the task's own source text, and the answer gives the advice the
//! reference records.
//!
//! A model asked what a person wrote will quote, and a model that does not
//! know will quote something that sounds right. Nothing a model says about its
//! own quotations is evidence; this verifier reads the answer and the source
//! text and nothing else. It passes only an answer that quotes at least one
//! passage of a length that is a claim to a passage, quotes only what the
//! source says (words, not punctuation or line breaks), and reproduces enough
//! of the reference for the answer to be that advice.
//!
//! The source text comes from an [`EvidenceText`], so the check does not know
//! where text is kept: [`StoredEvidence`] reads the task's evidence from the
//! source store.

use std::collections::HashSet;

use serde_json::json;
use splinter_core::annotation::{Producer, Strength};
use splinter_core::digest::Digest;
use splinter_core::experience::{Experience, Task};
use splinter_record::sources::SourceStore;

use super::{single_reference, Finding, Verifier, VerifyError};

/// Where the source text of a task comes from.
pub trait EvidenceText: Send + Sync {
    /// The text of every source part `task`'s evidence is in; empty when none
    /// can be had.
    ///
    /// # Errors
    /// The text exists but cannot be read.
    fn of(&self, task: &Task) -> Result<Vec<String>, VerifyError>;
}

/// The task's evidence parts, read from the source store.
pub struct StoredEvidence {
    sources: SourceStore,
}

impl StoredEvidence {
    /// Evidence read from `sources`.
    #[must_use]
    pub fn new(sources: SourceStore) -> Self {
        Self { sources }
    }
}

impl EvidenceText for StoredEvidence {
    fn of(&self, task: &Task) -> Result<Vec<String>, VerifyError> {
        let mut seen: HashSet<Digest> = HashSet::new();
        let mut texts = Vec::new();
        for span in &task.evidence {
            if seen.insert(span.source.clone()) {
                let bytes = self.sources.read_blob(&span.source)?;
                texts.push(String::from_utf8_lossy(&bytes).into_owned());
            }
        }
        Ok(texts)
    }
}

/// How strict the check is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuotationPolicy {
    /// The fewest words a quoted passage has to run to count as a claim to a
    /// passage; a shorter quoted phrase is a term, not a quotation.
    pub min_words: usize,
    /// The share of the reference's runs of words the answer has to
    /// reproduce, in `0.0..=1.0`; `0.0` does not look at the reference.
    pub min_reference_recall: f64,
}

/// The longest run of words recall is counted over.
const RUN: usize = 8;

/// Passes an answer whose quotations are all in the source and which gives
/// the reference advice; see the module documentation.
pub struct QuotationVerifier {
    producer: Producer,
    evidence: Box<dyn EvidenceText>,
    policy: QuotationPolicy,
}

impl QuotationVerifier {
    /// A verifier reading source text from `evidence`, its verdicts carrying
    /// `producer`.
    #[must_use]
    pub fn new(
        producer: Producer,
        evidence: Box<dyn EvidenceText>,
        policy: QuotationPolicy,
    ) -> Self {
        Self {
            producer,
            evidence,
            policy,
        }
    }
}

/// Lower-case alphanumeric words of `text`: what two printings of a passage
/// agree on once spacing, punctuation and capitals are set aside.
#[must_use]
pub fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// The passages of `answer` in double quotation marks, straight or curly,
/// that run at least `min_words` words.
#[must_use]
pub fn quotations(answer: &str, min_words: usize) -> Vec<String> {
    let mut found = Vec::new();
    let mut inside: Option<String> = None;
    for c in answer.chars() {
        match (&mut inside, c) {
            (None, '"' | '\u{201c}') => inside = Some(String::new()),
            (Some(open), '"' | '\u{201d}') => {
                if words(open).len() >= min_words {
                    found.push(std::mem::take(open));
                }
                inside = None;
            }
            (Some(open), c) => open.push(c),
            (None, _) => {}
        }
    }
    found
}

/// Texts as runs of normalised words, so a passage is looked up by whole
/// words, whichever text prints it and however its lines were broken. A
/// passage is found only inside one text, never across two.
pub struct TextIndex {
    haystacks: Vec<String>,
}

impl TextIndex {
    /// An index over `texts`.
    #[must_use]
    pub fn new<'a>(texts: impl IntoIterator<Item = &'a str>) -> Self {
        Self {
            haystacks: texts
                .into_iter()
                .map(|text| format!(" {} ", words(text).join(" ")))
                .collect(),
        }
    }

    /// Whether `passage` occurs in one of the texts, word for word.
    #[must_use]
    pub fn contains(&self, passage: &str) -> bool {
        let w = words(passage);
        if w.is_empty() {
            return false;
        }
        let needle = format!(" {} ", w.join(" "));
        self.haystacks.iter().any(|h| h.contains(&needle))
    }
}

fn runs(text: &str) -> HashSet<String> {
    let w = words(text);
    let size = RUN.min(w.len());
    if size == 0 {
        return HashSet::new();
    }
    w.windows(size).map(|r| r.join(" ")).collect()
}

impl Verifier for QuotationVerifier {
    fn producer(&self) -> Producer {
        self.producer.clone()
    }

    fn strength(&self) -> Strength {
        Strength::Formal
    }

    /// The evidence names counts and the digest of the answer, never a
    /// quotation or the reference.
    fn verify(&self, task: &Task, exp: &Experience) -> Result<Finding, VerifyError> {
        if !(0.0..=1.0).contains(&self.policy.min_reference_recall) {
            return Err(VerifyError::Parameter {
                name: "min_reference_recall",
                reason: "a share lies between 0 and 1".into(),
            });
        }
        let texts = self.evidence.of(task)?;
        if texts.iter().all(|t| t.trim().is_empty()) {
            return Ok(Finding::abstain(
                "the task's source text is not available",
                json!({}),
            ));
        }
        let Some(answer) = exp.final_output.as_deref() else {
            return Ok(Finding::decided(false, json!({ "answer": false })));
        };
        let index = TextIndex::new(texts.iter().map(String::as_str));
        let quoted = quotations(answer, self.policy.min_words);
        let invented = quoted.iter().filter(|q| !index.contains(q)).count();

        let recall = if self.policy.min_reference_recall > 0.0 {
            let reference = match single_reference(task) {
                Ok(reference) => reference,
                Err(abstain) => return Ok(abstain),
            };
            let wanted = runs(reference);
            if wanted.is_empty() {
                return Ok(Finding::abstain("the reference has no words", json!({})));
            }
            let got = runs(answer);
            wanted.iter().filter(|r| got.contains(*r)).count() as f64 / wanted.len() as f64
        } else {
            1.0
        };

        let passed =
            !quoted.is_empty() && invented == 0 && recall >= self.policy.min_reference_recall;
        Ok(Finding::decided(
            passed,
            json!({
                "comparison": "quotations against the task's source text",
                "quotations": quoted.len(),
                "not_in_source": invented,
                "min_words": self.policy.min_words,
                "reference_recall": recall,
                "min_reference_recall": self.policy.min_reference_recall,
                "answer": Digest::of(answer.as_bytes()),
            }),
        ))
    }
}
