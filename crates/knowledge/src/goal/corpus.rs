// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that decide which documents a
// person or an agent should study to reach a goal, for its clients. If your
// team needs expertise in goal-directed selection of learning material you
// can procure our services by sending an email to info@swedishembedded.com.

//! A raw corpus: documents with the little provenance they carry, cut into
//! the sections selection chooses among.
//!
//! A document is the knowledge objects' [`Document`]: Markdown with an
//! optional header of `authority`, `version` and `date` between `---`
//! fences, which says how far it is believed against another document.
//! Nothing here knows what a section is worth; that is [`super::select`]'s
//! decision.

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::sections::{sections, MARKDOWN};
use splinter_core::text::words;

pub use crate::objects::{Document, Provenance, NEUTRAL_AUTHORITY};

/// Where a section is: its document and its position among the document's
/// sections.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SectionId {
    /// The document.
    pub document: String,
    /// The section's position in it.
    pub section: usize,
}

impl fmt::Display for SectionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}#{}", self.document, self.section)
    }
}

/// A section that selection may choose.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    /// Where it is.
    pub id: SectionId,
    /// What its document says about itself.
    pub provenance: Provenance,
    /// Where it lies in its document's text, in bytes.
    pub range: Range<usize>,
    /// Its text.
    pub text: String,
}

impl Candidate {
    /// What it costs to learn it: its words.
    #[must_use]
    pub fn words(&self) -> usize {
        self.text.split_whitespace().count()
    }
}

/// Every section of every document, in corpus order.
#[must_use]
pub fn candidates(documents: &[Document]) -> Vec<Candidate> {
    documents
        .iter()
        .flat_map(|doc| {
            sections(&doc.text, MARKDOWN)
                .into_iter()
                .enumerate()
                .filter_map(|(at, s)| {
                    Some(Candidate {
                        id: SectionId {
                            document: doc.id.clone(),
                            section: at,
                        },
                        provenance: doc.provenance.clone(),
                        range: s.range.clone(),
                        text: s.text(&doc.text)?.to_string(),
                    })
                })
        })
        .collect()
}

/// How rare each word is across the sections of a corpus.
///
/// A word in nearly every section says nothing about which section is
/// wanted; one in a few names a topic. Both clustering probes into gaps and
/// ranking sections against them use the same weights.
#[derive(Clone, Debug, Default)]
pub struct Vocabulary {
    sections: usize,
    holders: HashMap<String, usize>,
}

impl Vocabulary {
    /// The vocabulary of `candidates`.
    #[must_use]
    pub fn of(candidates: &[Candidate]) -> Self {
        let mut holders: HashMap<String, usize> = HashMap::new();
        for c in candidates {
            let unique: BTreeSet<String> = words(&c.text).into_iter().collect();
            for w in unique {
                *holders.entry(w).or_default() += 1;
            }
        }
        Self {
            sections: candidates.len(),
            holders,
        }
    }

    /// The share of sections that have `word`; `None` when none does.
    #[must_use]
    pub fn share(&self, word: &str) -> Option<f64> {
        Some(*self.holders.get(word)? as f64 / self.sections.max(1) as f64)
    }

    /// The squared inverse-document-frequency weight of `word`; `None` for a
    /// word no section has, which no section can be chosen for, and for a
    /// number, which names one item and not what it is about.
    #[must_use]
    pub fn weight(&self, word: &str) -> Option<f64> {
        if word.chars().all(|c| c.is_ascii_digit()) {
            return None;
        }
        let holders = *self.holders.get(word)? as f64;
        Some(
            (((self.sections as f64) + 1.0) / (holders + 0.5))
                .ln()
                .powi(2),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn front_matter_is_provenance_and_the_body_is_cut_into_sections() {
        let text =
            "---\ndate: 2026-03-01\nversion: 3\nauthority: 3\n---\n# A\nfirst\n\n# B\nsecond\n";
        let doc = Document::parse("d1", text);
        assert_eq!(doc.provenance.date.as_deref(), Some("2026-03-01"));
        assert_eq!((doc.provenance.version, doc.provenance.authority), (3, 3));
        let found = candidates(std::slice::from_ref(&doc));
        assert_eq!(found.len(), 2);
        assert_eq!(found[1].id.to_string(), "d1#1");
    }

    #[test]
    fn a_document_without_front_matter_is_neutral() {
        let doc = Document::parse("d", "# A\nbody\n");
        assert_eq!(doc.provenance.authority, NEUTRAL_AUTHORITY);
        assert_eq!(doc.provenance.date, None);
    }
}
