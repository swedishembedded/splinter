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

/// The words a passage grows to by taking in the sections after it: enough to
/// have a referent and to say something, short enough that an embedding of it
/// is about one thing.
pub const TARGET_PASSAGE_WORDS: usize = 120;

/// The most words a passage holds. A section longer than this is split at
/// sentence ends, and a passage never takes in a section that would pass it;
/// an embedder reads only so much of a passage.
pub const MAX_PASSAGE_WORDS: usize = 260;

/// A run of text to be merged into a passage: a section, or a part of a long
/// one.
struct Unit {
    /// The first section it belongs to.
    section: usize,
    range: std::ops::Range<usize>,
    words: usize,
}

/// The sentences of `body`, which starts at byte `offset` of its text, as
/// byte ranges: each ends at a full stop, question mark or exclamation mark
/// followed by space, or at the end.
fn sentence_ranges(body: &str, offset: usize) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0;
    let mut chars = body.char_indices().peekable();
    while let Some((at, c)) = chars.next() {
        let ends = matches!(c, '.' | '?' | '!')
            && chars.peek().is_none_or(|(_, next)| next.is_whitespace());
        if ends {
            let end = at + c.len_utf8();
            ranges.push(offset + start..offset + end);
            start = end;
            // The next sentence begins at the first thing after the space.
            while let Some((next, space)) = chars.peek().copied() {
                if !space.is_whitespace() {
                    break;
                }
                chars.next();
                start = next + space.len_utf8();
            }
        }
    }
    if start < body.len() && !body[start..].trim().is_empty() {
        ranges.push(offset + start..offset + body.trim_end().len());
    }
    ranges
}

/// The units of one section: itself, or, past [`MAX_PASSAGE_WORDS`], runs of
/// whole sentences of about [`TARGET_PASSAGE_WORDS`] each.
fn units_of(section: usize, range: &std::ops::Range<usize>, text: &str) -> Vec<Unit> {
    let body = &text[range.clone()];
    let words = body.split_whitespace().count();
    if words <= MAX_PASSAGE_WORDS {
        return vec![Unit {
            section,
            range: range.clone(),
            words,
        }];
    }
    let mut units: Vec<Unit> = Vec::new();
    for sentence in sentence_ranges(body, range.start) {
        let n = text[sentence.clone()].split_whitespace().count();
        match units.last_mut() {
            Some(last)
                if last.words < TARGET_PASSAGE_WORDS && last.words + n <= MAX_PASSAGE_WORDS =>
            {
                last.range.end = sentence.end;
                last.words += n;
            }
            _ => units.push(Unit {
                section,
                range: sentence,
                words: n,
            }),
        }
    }
    units
}

/// The passages of every text part of `sources`, in source, part and section
/// order: each part's sections taken in turn and merged forward until a
/// passage holds [`TARGET_PASSAGE_WORDS`] (a header goes with what follows
/// it, a signature with what precedes it), a section past
/// [`MAX_PASSAGE_WORDS`] split at sentence ends, and a fragment too short to
/// say anything ([`MIN_PASSAGE_WORDS`]) left out.
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
            let mut units: Vec<Unit> = Vec::new();
            for (section, range) in text.sections().iter().enumerate() {
                units.extend(units_of(section, &range.range, &text.text));
            }
            let mut group: Option<Unit> = None;
            let flush = |group: Option<Unit>, found: &mut Vec<Passage>| {
                if let Some(unit) = group.filter(|u| u.words >= MIN_PASSAGE_WORDS) {
                    found.push(Passage {
                        source: Some(id.clone()),
                        part: part.name.clone(),
                        section: unit.section,
                        content: Some(text.content.clone()),
                        text: text.text[unit.range.clone()].to_string(),
                        range: unit.range,
                    });
                }
            };
            for unit in units {
                group = match group.take() {
                    Some(mut held)
                        if held.words < TARGET_PASSAGE_WORDS
                            && held.words + unit.words <= MAX_PASSAGE_WORDS =>
                    {
                        held.range.end = unit.range.end;
                        held.words += unit.words;
                        Some(held)
                    }
                    other => {
                        flush(other, &mut found);
                        Some(unit)
                    }
                };
            }
            flush(group, &mut found);
        }
    }
    Ok(found)
}
