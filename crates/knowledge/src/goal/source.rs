// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that decide which documents a
// person or an agent should study to reach a goal, for its clients. If your
// team needs expertise in goal-directed selection of learning material you
// can procure our services by sending an email to info@swedishembedded.com.

//! Where selection gets facts and conflict rulings: an admitted pack of
//! knowledge objects.
//!
//! Selection does not read facts or decide which of two statements stands.
//! The material compiler's [`objects`](crate::objects) do: a proposer says
//! what a document states, code admits only what the document's own quote
//! bears out, and where admitted objects disagree the documents' headers
//! decide (authority, then date, then version) or, if they cannot, the
//! disagreement is flagged. A [`PackSource`] lays that over the sections:
//! an object belongs to the section its quote lies in, its fact is the
//! section's fact, and its resolution is the section's ruling.
//!
//! [`PackSource::from_sentences`] proposes from the two sentence forms the
//! synthetic corpora use; a model proposer's pack goes through
//! [`PackSource::new`] the same way.

use std::collections::{BTreeSet, HashMap};

use serde_json::json;

use super::corpus::{Candidate, Document, SectionId};
use super::facts::{Fact, FactKey, Form, PatternFacts};
use super::select::Verdict;
use crate::objects::{admit, Object, Pack, ProposedObject};

/// What selection asks about a section.
pub trait FactSource {
    /// The facts the section states.
    fn facts(&self, section: &Candidate) -> Vec<Fact>;

    /// Whether another section's statement beats one of this section's,
    /// as a verdict that leaves it out; `None` when none does.
    fn ruling(&self, section: &Candidate) -> Option<Verdict>;

    /// Whether anything at all was made of the section: a section no object
    /// can be made of gives a gap nothing to be closed with.
    fn has_content(&self, section: &Candidate) -> bool;
}

/// A pack's objects laid over the sections of the corpus it came from.
pub struct PackSource {
    objects: BTreeSet<SectionId>,
    facts: HashMap<SectionId, Vec<Fact>>,
    rulings: HashMap<SectionId, Verdict>,
}

impl PackSource {
    /// The source of `pack` over `candidates`.
    #[must_use]
    pub fn new(pack: &Pack, candidates: &[Candidate]) -> Self {
        let mut by_object: HashMap<&str, SectionId> = HashMap::new();
        let mut objects: BTreeSet<SectionId> = BTreeSet::new();
        let mut facts: HashMap<SectionId, Vec<Fact>> = HashMap::new();
        for object in &pack.objects {
            let Some(section) = section_of(object, pack, candidates) else {
                continue;
            };
            by_object.insert(object.id.as_str(), section.clone());
            objects.insert(section.clone());
            if let Some(fact) = fact_of(object) {
                facts.entry(section).or_default().push(fact);
            }
        }
        let mut rulings: HashMap<SectionId, Verdict> = HashMap::new();
        for resolution in &pack.resolutions {
            let Some(winner) = by_object.get(resolution.winner.as_str()) else {
                continue;
            };
            for id in &resolution.superseded {
                let Some(loser) = by_object.get(id.as_str()) else {
                    continue;
                };
                if loser == winner {
                    continue;
                }
                let by = winner.clone();
                let verdict = match resolution.bases.get(id).map(String::as_str) {
                    Some("authority") => Verdict::Contradicted { by },
                    _ => Verdict::Superseded { by },
                };
                rulings.entry(loser.clone()).or_insert(verdict);
            }
        }
        for flag in &pack.flags {
            let sections: BTreeSet<&SectionId> = flag
                .objects
                .iter()
                .filter_map(|o| by_object.get(o.as_str()))
                .collect();
            for section in &sections {
                if let Some(other) = sections.iter().find(|s| *s != section) {
                    rulings
                        .entry((*section).clone())
                        .or_insert(Verdict::Contested {
                            with: (*other).clone(),
                        });
                }
            }
        }
        Self {
            objects,
            facts,
            rulings,
        }
    }

    /// A pack proposed from the sentence forms of the corpus's documents and
    /// admitted, laid over `candidates`. A key stays in the words the quote
    /// uses: admission requires it, so statements that paraphrase each other
    /// are not recognised as disagreeing.
    #[must_use]
    pub fn from_sentences(documents: &[Document], candidates: &[Candidate]) -> Self {
        let proposals: Vec<ProposedObject> =
            documents.iter().flat_map(sentence_proposals).collect();
        Self::new(&admit("goal", documents, &proposals), candidates)
    }
}

impl FactSource for PackSource {
    fn facts(&self, section: &Candidate) -> Vec<Fact> {
        self.facts.get(&section.id).cloned().unwrap_or_default()
    }

    fn ruling(&self, section: &Candidate) -> Option<Verdict> {
        self.rulings.get(&section.id).cloned()
    }

    fn has_content(&self, section: &Candidate) -> bool {
        self.objects.contains(&section.id)
    }
}

/// One `fact` proposal for each sentence of `document` that states one.
fn sentence_proposals(document: &Document) -> Vec<ProposedObject> {
    use super::facts::FactExtractor;
    let mut found = Vec::new();
    for sentence in document.text.split_inclusive(['.', '\n']) {
        let quote = sentence.trim();
        let Some(fact) = PatternFacts.facts(quote).into_iter().next() else {
            continue;
        };
        found.push(ProposedObject {
            id: format!("{}/{}", document.id, found.len()),
            kind: "fact".into(),
            document: document.id.clone(),
            quote: quote.to_string(),
            data: json!({
                "key": format!("{} | {}", fact.key.attribute, fact.key.entity),
                "value": fact.value,
                "form": match fact.key.form { Form::Property => "property", Form::Behaviour => "behaviour" },
            }),
        });
    }
    found
}

fn fact_of(object: &Object) -> Option<Fact> {
    if object.kind != "fact" {
        return None;
    }
    let (attribute, entity) = object.data["key"].as_str()?.split_once('|')?;
    let form = match object.data["form"].as_str() {
        Some("behaviour") => Form::Behaviour,
        _ => Form::Property,
    };
    Some(Fact {
        key: FactKey {
            form,
            attribute: attribute.trim().to_lowercase(),
            entity: entity.trim().to_lowercase(),
        },
        value: object.data["value"].as_str()?.to_string(),
    })
}

/// The section of `candidates` the object's quote lies in.
fn section_of(object: &Object, pack: &Pack, candidates: &[Candidate]) -> Option<SectionId> {
    let document = pack.documents.iter().find(|d| d.id == object.document)?;
    let at = if document.text.is_ascii() {
        object.span[0]
    } else {
        // Spans are in characters; sections are in bytes.
        document.text.char_indices().nth(object.span[0])?.0
    };
    candidates
        .iter()
        .find(|c| c.id.document == object.document && c.range.contains(&at))
        .map(|c| c.id.clone())
}
