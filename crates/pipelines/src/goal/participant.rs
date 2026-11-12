// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that decide which documents a
// person or an agent should study to reach a goal, for its clients. If your
// team needs expertise in goal-directed selection of learning material you
// can procure our services by sending an email to info@swedishembedded.com.

//! The participant being taught: something that answers questions and can
//! take material in.
//!
//! [`InContext`] learns by remembering sections and answers by showing a
//! [`Reader`] the few remembered sections that bear on the question. That
//! is learning without weights, which is what a run on a machine with no
//! free card can do; a participant that trains an adapter on the compiled
//! material implements [`Participant`] the same way.

use splinter_knowledge::goal::corpus::Candidate;
use splinter_knowledge::goal::select::Examinee;
use splinter_knowledge::retrieve::{Bm25, Passage};

/// How many remembered sections a question is answered from.
pub const SHOWN_SECTIONS: usize = 3;

/// A question with the material shown with it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Query {
    /// What is asked.
    pub question: String,
    /// The sections shown, best match first; empty for closed-book.
    pub material: Vec<String>,
}

/// Answers queries: a model, or a stand-in for one.
pub trait Reader {
    /// One answer per query, in order.
    fn read_all(&self, queries: &[Query]) -> Vec<String>;
}

/// A participant that can be asked and taught.
pub trait Participant: Examinee {
    /// Takes `sections` in.
    fn learn(&mut self, sections: &[Candidate]);
}

/// Learns by remembering; answers by reading what it remembers.
pub struct InContext<R> {
    reader: R,
    memory: Vec<Candidate>,
}

impl<R: Reader> InContext<R> {
    /// A participant that has learned nothing.
    #[must_use]
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            memory: Vec::new(),
        }
    }

    /// The sections it remembers.
    #[must_use]
    pub fn memory(&self) -> &[Candidate] {
        &self.memory
    }
}

impl<R: Reader> Examinee for InContext<R> {
    fn answer_all(&self, questions: &[String]) -> Vec<String> {
        let passages: Vec<Passage> = self
            .memory
            .iter()
            .map(|c| Passage::of_text(&c.id.document, c.id.section, &c.text))
            .collect();
        let index = Bm25::new(&passages);
        let queries: Vec<Query> = questions
            .iter()
            .map(|question| Query {
                question: question.clone(),
                material: index
                    .search(question, SHOWN_SECTIONS)
                    .into_iter()
                    .map(|hit| self.memory[hit.passage].text.clone())
                    .collect(),
            })
            .collect();
        self.reader.read_all(&queries)
    }
}

impl<R: Reader> Participant for InContext<R> {
    fn learn(&mut self, sections: &[Candidate]) {
        self.memory.extend(sections.iter().cloned());
    }
}

/// A stand-in reader for tests and for runs with no model: it answers from
/// the first shown section that states the asked fact, else from what it
/// "knew" before, else says it does not know. It does not weigh a section's
/// date or authority; when two shown sections disagree the better ranked
/// one wins, as it would for a reader that trusts what it is shown first.
pub struct ScriptedReader {
    known: Vec<splinter_knowledge::goal::facts::Fact>,
}

impl ScriptedReader {
    /// A reader that knew `known` before it was shown anything.
    #[must_use]
    pub fn knowing(known: Vec<splinter_knowledge::goal::facts::Fact>) -> Self {
        Self { known }
    }
}

impl Reader for ScriptedReader {
    fn read_all(&self, queries: &[Query]) -> Vec<String> {
        use splinter_knowledge::goal::facts::{FactExtractor, PatternFacts};
        queries
            .iter()
            .map(|q| {
                let shown = q
                    .material
                    .iter()
                    .flat_map(|m| PatternFacts.facts(m))
                    .find(|f| f.question() == q.question);
                let own = || self.known.iter().find(|f| f.question() == q.question);
                shown
                    .or_else(|| own().cloned())
                    .map_or_else(|| "unknown".to_string(), |f| f.value)
            })
            .collect()
    }
}
