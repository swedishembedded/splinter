// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data built from a writer's own
// text without a model in the loop, for its clients. If your team needs
// expertise in training a model on a person's voice from what they wrote,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The voice view: the writer's own text, the whole corpus of it, as the
//! supervised answer to a request that says what the passage is.
//!
//! A conversation written by a teacher costs a model's time for every
//! answer, and a run over a large body of writing reaches a few per cent of
//! it. The writer's own text costs nothing to turn into records and has the
//! diction and reasoning no paraphrase has. Every text part of every source
//! in the corpus is a candidate, in a stable order that does not follow the
//! parts' names, so a limited projection covers a spread of the corpus.
//!
//! A part is cut into chunks of [`MIN_CHUNK_WORDS`] to [`MAX_CHUNK_WORDS`]
//! words (`chunks`): whole sentences, closed where a paragraph ends, a
//! heading always with the text it heads and no chunk of nothing but a
//! heading. Each chunk is one record whose answer is the chunk, word for
//! word, and whose request is written by code from what the text says of
//! itself - whom it is to, in which year, what it opens on (`prompt`) - so
//! the model learns to write about what it is asked, in the writer's words.
//! Told who the writer is ([`Voice::written_as`]), half the records are
//! asked under the persona prompt and half under the identity line alone, so
//! the voice is carried by the adapter and not only by the prompt.
//!
//! Two prints of one text are one family ([`Voice::keeping_one_print_of`]):
//! only one is projected, and no family supplies more than its share of the
//! tokens ([`Voice::bounding_families_to`]). Content projected twice is a
//! [`Exclusion::Duplicate`], content that is not text [`Exclusion::NotText`],
//! empty content [`Exclusion::Empty`], a text with too little to be a record
//! [`Exclusion::TooShort`]. There is no student input to strip and no
//! verdict to read.

mod chunks;
mod prompt;

use std::collections::{BTreeMap, HashSet};
use std::ops::Range;

use splinter_core::digest::Digest;
use splinter_core::prompt::persona_prompt;
use splinter_store::sources::SourceStore;

use super::source_text;
use crate::render::message;
use crate::{
    spread, Corpus, Exclusion, Objective, Projection, Provenance, RecordBody, View, ViewError,
};
use chunks::{chunks, Limits};

/// The name every voice record carries.
pub const NAME: &str = "voice";

/// A chunk closes at a paragraph end once it holds this many words.
pub const MIN_CHUNK_WORDS: usize = 250;

/// The most words a chunk holds.
pub const MAX_CHUNK_WORDS: usize = 650;

/// The most tokens a chunk holds: with the system and request turns before
/// it, within the 1024-token row a persona run's conversations need.
pub const MAX_CHUNK_TOKENS: usize = 880;

/// The fewest words of body a text or a chunk must hold to be a record.
pub const MIN_BODY_WORDS: usize = 25;

/// The most of the voice tokens one family may supply by default: a work of
/// many chunks is a tenth of the writer's text, not the half of it.
pub const DEFAULT_MAX_FAMILY_SHARE: f64 = 0.1;

/// How many characters count as one token when no tokenizer measures the
/// text: fewer than prose takes, so the bound holds for text that tokenizes
/// badly.
pub const CHARS_PER_TOKEN: usize = 3;

/// How a text's length is measured against [`MAX_CHUNK_TOKENS`].
pub type Measure<'m> = &'m (dyn Fn(&str) -> usize + 'm);

/// How a text of a media type is divided into sections, as byte ranges in
/// it: a chunk closes where a section does.
pub type Sectioner<'x> = &'x (dyn Fn(&str, &str) -> Vec<Range<usize>> + 'x);

/// The family of a text part's content: the name shared by the prints of one
/// text, `None` when it has none.
pub type FamilyOf<'f> = &'f (dyn Fn(&Digest) -> Option<String> + 'f);

/// The whole text as one section: the sectioner when none is given.
#[must_use]
pub fn whole_text(text: &str, _media_type: &str) -> Vec<Range<usize>> {
    std::iter::once(0..text.len()).collect()
}

/// Characters at [`CHARS_PER_TOKEN`], rounded up: the measure when no
/// tokenizer is given.
#[must_use]
pub fn chars_as_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(CHARS_PER_TOKEN)
}

/// The writer's own text as supervised answers.
#[derive(Clone, Copy)]
pub struct Voice<'s, 'm> {
    sources: &'s SourceStore,
    measure: Measure<'m>,
    sectioner: Sectioner<'m>,
    author: Option<&'m str>,
    family_of: Option<FamilyOf<'m>>,
    max_family_share: f64,
}

impl<'s, 'm> Voice<'s, 'm> {
    /// The view reading source content from `sources`, its chunks measured
    /// in characters, each part one section, no family bound.
    #[must_use]
    pub fn new(sources: &'s SourceStore) -> Self {
        Self {
            sources,
            measure: &chars_as_tokens,
            sectioner: &whole_text,
            author: None,
            family_of: None,
            max_family_share: 1.0,
        }
    }

    /// The same view with its chunks measured by `measure`: the tokens of
    /// the model to be trained.
    #[must_use]
    pub fn measured_by(self, measure: Measure<'m>) -> Self {
        Self { measure, ..self }
    }

    /// The same view with its parts divided into sections by `sectioner`.
    #[must_use]
    pub fn sectioned_by(self, sectioner: Sectioner<'m>) -> Self {
        Self { sectioner, ..self }
    }

    /// The same view writing its requests as `author`'s, half of them under
    /// the persona prompt of `author` and half under the line that says who
    /// the model is and nothing more.
    #[must_use]
    pub fn written_as(self, author: &'m str) -> Self {
        Self {
            author: Some(author),
            ..self
        }
    }

    /// The same view projecting one print of each text that `family_of`
    /// puts in a family: the print whose content is the family's name.
    #[must_use]
    pub fn keeping_one_print_of(self, family_of: FamilyOf<'m>) -> Self {
        Self {
            family_of: Some(family_of),
            ..self
        }
    }

    /// The same view in which no family supplies more than `share` of the
    /// tokens - or its even share, when there are fewer families than that
    /// leaves room for.
    #[must_use]
    pub fn bounding_families_to(self, share: f64) -> Self {
        Self {
            max_family_share: share,
            ..self
        }
    }

    fn limits() -> Limits {
        Limits {
            min_words: MIN_CHUNK_WORDS,
            max_words: MAX_CHUNK_WORDS,
            max_tokens: MAX_CHUNK_TOKENS,
            min_body_words: MIN_BODY_WORDS,
        }
    }

    /// The system turn the author's persona makes of a record answering
    /// `answer`: half the records, by the answer's own digest so the half
    /// does not move with what else is projected, are asked under the
    /// identity line alone. `None` when the author is not known.
    fn system_for(&self, answer: &str) -> Option<String> {
        let author = self.author?;
        let digest = Digest::of(answer.as_bytes());
        let bare = digest
            .as_str()
            .chars()
            .last()
            .and_then(|c| c.to_digit(16))
            .is_some_and(|nibble| nibble % 2 == 0);
        Some(if bare {
            identity(author)
        } else {
            persona_prompt(author)
        })
    }

    /// Which candidates to keep so that no family supplies more than its
    /// share of the tokens, each family thinned evenly.
    fn within_family_bound(&self, candidates: &[Candidate]) -> Vec<bool> {
        let mut by_family: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for (n, c) in candidates.iter().enumerate() {
            by_family.entry(&c.family).or_default().push(n);
        }
        let total: usize = candidates.iter().map(|c| c.tokens).sum();
        let share = self
            .max_family_share
            .max(1.0 / by_family.len().max(1) as f64);
        let allowed = (share * total as f64) as usize;
        let mut kept = vec![true; candidates.len()];
        for members in by_family.values() {
            let sizes: Vec<usize> = members.iter().map(|&n| candidates[n].tokens).collect();
            // A family keeps at least its largest chunk, whatever the bound.
            let room = allowed.max(sizes.iter().copied().max().unwrap_or(0));
            for (&n, keep) in members.iter().zip(spread(&sizes, room)) {
                kept[n] = keep;
            }
        }
        kept
    }
}

/// One record about to be projected: its family and size, for the bound.
struct Candidate {
    family: String,
    tokens: usize,
    body: RecordBody,
    system: Option<String>,
    source: Digest,
}

/// The line that says who the model is, with nothing about how to answer.
fn identity(author: &str) -> String {
    format!("You are {}.", author.trim())
}

impl View for Voice<'_, '_> {
    fn name(&self) -> &str {
        NAME
    }

    fn objective(&self) -> Objective {
        Objective::Sft
    }

    fn project(&self, corpus: &Corpus) -> Result<Projection, ViewError> {
        let mut projection = Projection::new(NAME, Objective::Sft, None, None);
        let mut seen: HashSet<Digest> = HashSet::new();
        let mut candidates: Vec<Candidate> = Vec::new();
        for id in corpus.sources() {
            let source = self.sources.get_source(id)?;
            let mut parts: Vec<_> = source
                .parts
                .iter()
                .filter(|p| p.media_type.starts_with("text/"))
                .collect();
            // Not in name order: a limited projection covers a spread of the
            // parts, not the first files.
            parts.sort_by_cached_key(|p| Digest::of(p.name.as_bytes()));
            for part in parts {
                let family = self.family_of.and_then(|f| f(&part.content));
                let other_print = family
                    .as_deref()
                    .is_some_and(|name| name != part.content.as_str());
                if other_print || !seen.insert(part.content.clone()) {
                    projection.exclude(Exclusion::Duplicate);
                    continue;
                }
                let text = match source_text(self.sources.read_blob(&part.content))? {
                    Ok(text) => text,
                    Err(reason) => {
                        projection.exclude(reason);
                        continue;
                    }
                };
                let family = family.unwrap_or_else(|| part.content.as_str().to_string());
                let text = text.trim();
                let (cut, heading) = chunks(
                    text,
                    &part.media_type,
                    &Self::limits(),
                    self.measure,
                    self.sectioner,
                );
                if cut.is_empty() {
                    projection.exclude(Exclusion::TooShort);
                    continue;
                }
                let heading = heading.map(|range| text[range].to_string());
                for chunk in cut {
                    let answer = &text[chunk.range.clone()];
                    let request = prompt::request(
                        self.author,
                        heading.as_deref(),
                        &text[chunk.body_start..chunk.range.end],
                    );
                    candidates.push(Candidate {
                        family: family.clone(),
                        tokens: (self.measure)(answer),
                        body: RecordBody::Chat {
                            messages: vec![
                                message("user", &request, false),
                                message("assistant", answer, true),
                            ],
                        },
                        system: self.system_for(answer),
                        source: part.content.clone(),
                    });
                }
            }
        }
        let kept = self.within_family_bound(&candidates);
        for (candidate, keep) in candidates.into_iter().zip(kept) {
            if !keep {
                projection.exclude(Exclusion::OverLimit);
                continue;
            }
            projection.push(
                candidate.body,
                Provenance {
                    sources: vec![candidate.source],
                    ..Provenance::default()
                },
            );
            // `push` opens every conversation with the default system turn;
            // a known writer's records open with the writer's own.
            if let (Some(system), Some(record)) = (candidate.system, projection.records.last_mut())
            {
                if let RecordBody::Chat { messages } = &mut record.body {
                    messages[0].content = system;
                }
            }
        }
        Ok(projection)
    }
}
