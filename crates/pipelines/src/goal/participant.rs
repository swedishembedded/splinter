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
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::rc::Rc;

use splinter_knowledge::goal::corpus::SectionId;
use splinter_knowledge::goal::hard::canonical;
use splinter_knowledge::retrieve::{fuse, Bm25, Dense, Embedder, Hit, Passage};

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
    provenance: bool,
    /// How the sections shown with a question are found.
    retrieval: Retrieval,
    /// What embeds, when retrieval is by meaning.
    embedder: Option<Rc<dyn Embedder>>,
    /// The vectors of remembered sections, made once.
    vectors: RefCell<HashMap<SectionId, Vec<f32>>>,
}

/// How the sections shown with a question are found among the remembered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Retrieval {
    /// By the question's words (BM25).
    Lexical,
    /// By closeness in meaning (an embedder).
    Semantic,
    /// Both, fused by reciprocal rank, so a section either finds counts and
    /// one both find counts most.
    Fused,
}

impl<R: Reader> InContext<R> {
    /// A participant that has learned nothing.
    #[must_use]
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            memory: Vec::new(),
            provenance: false,
            retrieval: Retrieval::Lexical,
            embedder: None,
            vectors: RefCell::new(HashMap::new()),
        }
    }

    /// Finds the sections shown with a question as `retrieval` says, with
    /// `embedder` for the part that goes by meaning. A failed embedding shows
    /// the question no section; the embedder is the one to ask what failed.
    #[must_use]
    pub fn retrieving(mut self, retrieval: Retrieval, embedder: Rc<dyn Embedder>) -> Self {
        self.retrieval = retrieval;
        self.embedder = Some(embedder);
        self
    }

    /// The remembered sections that bear on `question`, best first.
    fn recall(&self, question: &str, bm25: &Bm25) -> Vec<usize> {
        let lexical = || bm25.search(question, SHOWN_SECTIONS);
        let semantic = |embedder: &Rc<dyn Embedder>| -> Vec<Hit> {
            let mut vectors = self.vectors.borrow_mut();
            let missing: Vec<usize> = (0..self.memory.len())
                .filter(|i| !vectors.contains_key(&self.memory[*i].id))
                .collect();
            for batch in missing.chunks(32) {
                let texts: Vec<&str> = batch
                    .iter()
                    .map(|i| self.memory[*i].text.as_str())
                    .collect();
                match embedder.embed(&texts) {
                    Ok(made) if made.len() == texts.len() => {
                        for (i, v) in batch.iter().zip(made) {
                            vectors.insert(self.memory[*i].id.clone(), v);
                        }
                    }
                    _ => return Vec::new(),
                }
            }
            let dense =
                Dense::from_vectors(self.memory.iter().map(|c| vectors[&c.id].clone()).collect());
            dense
                .search(question, &**embedder, SHOWN_SECTIONS)
                .unwrap_or_default()
        };
        let hits = match (self.retrieval, &self.embedder) {
            (Retrieval::Semantic, Some(e)) => semantic(e),
            (Retrieval::Fused, Some(e)) => fuse(&[lexical(), semantic(e)], SHOWN_SECTIONS),
            _ => lexical(),
        };
        hits.into_iter().map(|h| h.passage).collect()
    }

    /// Shows the reader each section under a header with its document's
    /// authority, date and version, as [`header`] writes it, so that a
    /// reader that can weigh them is able to settle a disagreement itself.
    #[must_use]
    pub fn showing_provenance(mut self) -> Self {
        self.provenance = true;
        self
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
                material: self
                    .recall(question, &index)
                    .into_iter()
                    .map(|at| {
                        let section = &self.memory[at];
                        if self.provenance {
                            format!("{}\n{}", header(section), section.text)
                        } else {
                            section.text.clone()
                        }
                    })
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

/// The line that heads a section shown with its provenance.
#[must_use]
pub fn header(section: &Candidate) -> String {
    let p = &section.provenance;
    format!(
        "[authority {}, date {}, version {}]",
        p.authority,
        p.date.as_deref().unwrap_or("unknown"),
        p.version
    )
}

/// A stand-in reader for tests and for runs with no model: it answers from
/// the shown section that states the asked fact, else from what it "knew"
/// before, else says it does not know. When shown sections disagree it takes
/// the one whose [`header`] stands highest (authority, then date, then
/// version) and, where the sections carry no header, the better ranked one,
/// as a reader would that trusts what it is shown first.
pub struct ScriptedReader {
    known: Vec<splinter_knowledge::goal::facts::Fact>,
    /// Variants and what they stand for: the meaning a model would read.
    lexicon: BTreeMap<String, String>,
}

impl ScriptedReader {
    /// A reader that knew `known` before it was shown anything.
    #[must_use]
    pub fn knowing(known: Vec<splinter_knowledge::goal::facts::Fact>) -> Self {
        Self {
            known,
            lexicon: BTreeMap::new(),
        }
    }

    /// Reads paraphrases as the words they stand for, as a model reads them.
    #[must_use]
    pub fn understanding(mut self, lexicon: BTreeMap<String, String>) -> Self {
        self.lexicon = lexicon;
        self
    }
}

/// Whether two questions ask the same once variants are read as what they
/// stand for.
fn same(a: &str, b: &str, lexicon: &BTreeMap<String, String>) -> bool {
    canonical(a, lexicon) == canonical(b, lexicon)
}

/// What a [`header`] line says: authority, date, version; all zero or empty
/// when the section has no header.
fn standing(section: &str) -> (u8, String, u32) {
    let Some(line) = section
        .lines()
        .next()
        .filter(|l| l.starts_with("[authority "))
    else {
        return (0, String::new(), 0);
    };
    let field = |name: &str| {
        line.trim_matches(['[', ']'])
            .split(", ")
            .find_map(|part| part.strip_prefix(name))
            .map(str::trim)
    };
    (
        field("authority ")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
        field("date ").unwrap_or("").to_string(),
        field("version ").and_then(|v| v.parse().ok()).unwrap_or(0),
    )
}

impl Reader for ScriptedReader {
    fn read_all(&self, queries: &[Query]) -> Vec<String> {
        use splinter_knowledge::goal::facts::{FactExtractor, PatternFacts};
        queries
            .iter()
            .map(|q| {
                // The first of the best-standing sections that states it:
                // `max_by_key` keeps the last of equals, so rank in reverse.
                let shown = q
                    .material
                    .iter()
                    .enumerate()
                    .flat_map(|(rank, m)| {
                        PatternFacts
                            .facts(m)
                            .into_iter()
                            .filter(|f| same(&f.question(), &q.question, &self.lexicon))
                            .map(move |f| (standing(m), std::cmp::Reverse(rank), f))
                    })
                    .max_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)))
                    .map(|(_, _, f)| f);
                let own = || {
                    self.known
                        .iter()
                        .find(|f| same(&f.question(), &q.question, &self.lexicon))
                };
                shown
                    .or_else(|| own().cloned())
                    .map_or_else(|| "unknown".to_string(), |f| f.value)
            })
            .collect()
    }
}

/// An embedder that understands only a lexicon: a text becomes the counts of
/// its words after variants are read as what they stand for, hashed into a
/// fixed number of dimensions. It stands in for a model's sense of meaning
/// in runs with no model, and knows exactly the synonyms it was given.
pub struct LexiconEmbedder {
    lexicon: BTreeMap<String, String>,
}

/// Words every question and passage has; they say nothing of meaning.
const FUNCTION_WORDS: [&str; 14] = [
    "what", "is", "the", "of", "a", "does", "do", "when", "client", "this", "about", "an", "to",
    "and",
];

/// The dimensions of a [`LexiconEmbedder`] vector.
const LEXICON_DIMENSIONS: usize = 512;

impl LexiconEmbedder {
    /// An embedder that reads `lexicon`.
    #[must_use]
    pub fn new(lexicon: BTreeMap<String, String>) -> Self {
        Self { lexicon }
    }
}

impl Embedder for LexiconEmbedder {
    fn name(&self) -> String {
        "lexicon".to_string()
    }

    fn embed(
        &self,
        texts: &[&str],
    ) -> Result<Vec<Vec<f32>>, splinter_knowledge::retrieve::EmbedError> {
        Ok(texts
            .iter()
            .map(|text| {
                let mut v = vec![0.0f32; LEXICON_DIMENSIONS];
                for word in canonical(text, &self.lexicon)
                    .split(|c: char| !c.is_alphanumeric())
                    .filter(|w| !w.is_empty() && !FUNCTION_WORDS.contains(w))
                {
                    let hash = word.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
                        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
                    });
                    v[(hash % LEXICON_DIMENSIONS as u64) as usize] += 1.0;
                }
                v
            })
            .collect())
    }
}

impl Reader for Box<dyn Reader> {
    fn read_all(&self, queries: &[Query]) -> Vec<String> {
        (**self).read_all(queries)
    }
}
