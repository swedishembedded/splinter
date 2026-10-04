// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements retrieval over a person's writing that
// finds the passages a task or a principle bears on, for its clients. If
// your team needs expertise in hybrid lexical and semantic search over a
// large corpus, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Retrieval: the passages of the sources that bear on a query.
//!
//! The sources become [`Passage`]s - each section of each text part with
//! enough words to say something - and three small parts rank them:
//!
//! * [`Bm25`], lexical: rare words weigh more than common ones, a long
//!   passage does not win by length. Exact names, places and dates are found
//!   by it where an embedding blurs them.
//! * [`Dense`], semantic: cosine over the vectors an [`Embedder`] makes. The
//!   embedder is a trait; computing embeddings is model work that belongs to
//!   the model runtime, not to retrieval.
//! * [`fuse`], reciprocal-rank fusion of any number of rankings, so a passage
//!   both kinds of search find beats one only either finds.
//!
//! A [`Library`] puts the two to work: meaning first, the words an embedding
//! blurs filling the tail. A [`Reranker`] reads the candidates it finds
//! beside the question and moves forward the ones that bear on it.
//!
//! A ranking is a list of [`Hit`]s by passage index, best first, and is
//! deterministic: ties break by the passage's position.

mod dense;
mod fuse;
mod lexical;
mod library;
mod rerank;

pub use dense::{Dense, EmbedError, Embedder};
pub use fuse::fuse;
pub use lexical::Bm25;
pub use library::{FindError, Library};
pub use rerank::{RerankError, Reranker};

use splinter_core::digest::Digest;
use splinter_core::source::SourceId;
use splinter_store::sources::SourceStore;

use crate::sections::MARKDOWN;
use crate::tasks::{GenerateError, SourceText};

/// The fewest words a section has to run to be a passage: shorter ones are
/// headings, salutations and signatures.
pub const MIN_PASSAGE_WORDS: usize = 12;

/// One section of one text part of a source, with the text it holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Passage {
    /// The source it is from; `None` for a passage that was not read from a
    /// stored source.
    pub source: Option<SourceId>,
    /// The part of the source.
    pub part: String,
    /// Its position among the part's sections.
    pub section: usize,
    /// The digest of the part's content the byte range indexes into; `None`
    /// when the passage was not read from a stored source.
    pub content: Option<Digest>,
    /// Where in the part's content it lies, in bytes.
    pub range: std::ops::Range<usize>,
    /// Its text.
    pub text: String,
}

impl Passage {
    /// A passage of `text` that is not from a stored source.
    #[must_use]
    pub fn of_text(part: &str, section: usize, text: &str) -> Self {
        Self {
            source: None,
            part: part.to_string(),
            section,
            content: None,
            range: 0..text.len(),
            text: text.to_string(),
        }
    }
}

/// One ranked passage: its index in the slice the ranking was built over,
/// and the score the ranking gave it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    /// The passage's index.
    pub passage: usize,
    /// Its score; comparable only within one ranking.
    pub score: f64,
}

/// The passages of every text part of `sources`, in source, part and section
/// order.
pub fn passages(store: &SourceStore, sources: &[SourceId]) -> Result<Vec<Passage>, GenerateError> {
    let mut found = Vec::new();
    for id in sources {
        let source = store.get_source(id)?;
        for part in source
            .parts
            .iter()
            .filter(|p| p.media_type.starts_with("text/") || p.media_type == MARKDOWN)
        {
            let text = SourceText::load(store, id, &part.name)?;
            for (section, range) in text.sections().iter().enumerate() {
                let Some(body) = text.section_text(section) else {
                    continue;
                };
                if body.split_whitespace().count() < MIN_PASSAGE_WORDS {
                    continue;
                }
                found.push(Passage {
                    source: Some(id.clone()),
                    part: part.name.clone(),
                    section,
                    content: Some(text.content.clone()),
                    range: range.range.clone(),
                    text: body.to_string(),
                });
            }
        }
    }
    Ok(found)
}
