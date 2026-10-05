// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data built from a writer's own
// text without a model in the loop, for its clients. If your team needs
// expertise in training a model on a person's voice from what they wrote,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! The voice view: the writer's own text as the supervised answer, in the
//! shape the policy is trained and asked in - a system turn, a user turn,
//! and the writer's words word for word as the reply.
//!
//! A conversation written by a teacher costs a model's time for every
//! answer, and a run over a large body of writing reaches a few per cent of
//! it. The writer's own text costs nothing to turn into records and has the
//! diction and reasoning no paraphrase has. Every text part of every source
//! in the corpus is a candidate, in a stable order that does not follow the
//! parts' names, so a limited projection covers a spread of the corpus.
//!
//! A part is cut into stretches of whole sections - as the builder's
//! sectioner divides the text ([`Voice::sectioned_by`]; without one a part
//! is one section) - of at most
//! [`STRETCH_TOKENS`] tokens, measured as the policy's tokenizer counts them
//! when the builder has it ([`Voice::measured_by`]) and else as characters
//! at [`CHARS_PER_TOKEN`], so that no record outgrows the row the
//! conversations it is trained beside need (a chat fine-tune's row is sized
//! by its longest record, and a row twice as long costs twice the memory and
//! time at every step). Each stretch is one record: the one user turn
//! [`ASK`], and the stretch, whole, as the answer. Measured, opening the
//! user turn with the part's heading or with the stretch's own first
//! sentence changed nothing in what the policy learned for the dialogues
//! it is asked as, and left that much of the writer's text unsupervised; so
//! every word is the answer and no model writes any of it. Content
//! projected twice is a [`Exclusion::Duplicate`], content that is not text
//! [`Exclusion::NotText`], empty content [`Exclusion::Empty`]. There is no
//! student input to strip and no verdict to read.

use std::collections::HashSet;
use std::ops::Range;

use splinter_core::digest::Digest;
use splinter_store::sources::SourceStore;

use super::source_text;
use crate::render::message;
use crate::{Corpus, Exclusion, Objective, Projection, Provenance, RecordBody, View, ViewError};

/// The name every voice record carries.
pub const NAME: &str = "voice";

/// The most tokens of the writer's text one record answers with: with the
/// system and user turns before it, within the 1024-token row a persona
/// run's conversations need.
pub const STRETCH_TOKENS: usize = 768;

/// How many characters count as one token when no tokenizer measures the
/// text: fewer than prose takes, so the bound holds for text that tokenizes
/// badly.
pub const CHARS_PER_TOKEN: usize = 3;

/// The user turn every stretch is the answer to.
pub const ASK: &str = "Write one of your writings, as you wrote it.";

/// How a text's length is measured against [`STRETCH_TOKENS`].
pub type Measure<'m> = &'m (dyn Fn(&str) -> usize + 'm);

/// How a text of a media type is divided into sections, as byte ranges in
/// it: a stretch holds whole sections where it can.
pub type Sectioner<'x> = &'x (dyn Fn(&str, &str) -> Vec<Range<usize>> + 'x);

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
}

impl<'s, 'm> Voice<'s, 'm> {
    /// The view reading source content from `sources`, its stretches
    /// measured in characters, each part one section.
    #[must_use]
    pub fn new(sources: &'s SourceStore) -> Self {
        Self {
            sources,
            measure: &chars_as_tokens,
            sectioner: &whole_text,
        }
    }

    /// The same view with its stretches measured by `measure`: the tokens
    /// of the model to be trained.
    #[must_use]
    pub fn measured_by(self, measure: Measure<'m>) -> Self {
        Self { measure, ..self }
    }

    /// The same view with its parts divided into sections by `sectioner`.
    #[must_use]
    pub fn sectioned_by(self, sectioner: Sectioner<'m>) -> Self {
        Self { sectioner, ..self }
    }
}

/// `text` cut into stretches of whole sections of at most `limit` as
/// `measure` counts; a section longer than that is cut at whitespace. Each
/// piece is measured once, the blank line between pieces counted as one,
/// since a tokenizer's count over a paragraph boundary is the sum of its
/// parts to within a token.
fn stretches(
    text: &str,
    media_type: &str,
    limit: usize,
    measure: Measure<'_>,
    sectioner: Sectioner<'_>,
) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut open = String::new();
    let mut used = 0;
    for range in sectioner(text, media_type) {
        let Some(piece) = text.get(range).filter(|p| !p.trim().is_empty()) else {
            continue;
        };
        for piece in cut_to_fit(piece, limit, measure) {
            let size = measure(piece);
            if !open.is_empty() && used + 1 + size > limit {
                out.push(std::mem::take(&mut open));
                used = 0;
            }
            if !open.is_empty() {
                open.push_str("\n\n");
                used += 1;
            }
            open.push_str(piece);
            used += size;
        }
    }
    if !open.is_empty() {
        out.push(open);
    }
    out
}

/// `text` as pieces of at most `limit` as `measure` counts, each cut at the
/// last whitespace that fits (or, when no whitespace does, at the last
/// character that does). The measure grows with the text, so the cut is
/// found by bisection over the candidate boundaries.
fn cut_to_fit<'t>(text: &'t str, limit: usize, measure: Measure<'_>) -> Vec<&'t str> {
    let mut pieces = Vec::new();
    let mut rest = text;
    while measure(rest) > limit {
        let fits = |boundaries: Vec<usize>| -> Option<usize> {
            let fitting = boundaries.partition_point(|&at| measure(&rest[..at]) <= limit);
            (fitting > 0).then(|| boundaries[fitting - 1])
        };
        let at_whitespace: Vec<usize> = rest
            .char_indices()
            .filter(|(at, c)| *at > 0 && c.is_whitespace())
            .map(|(at, _)| at)
            .collect();
        let at_any = || rest.char_indices().map(|(at, _)| at).skip(1).collect();
        let Some(cut) = fits(at_whitespace).or_else(|| fits(at_any())) else {
            // Not even one character fits: the piece is what it is.
            pieces.push(rest);
            return pieces;
        };
        pieces.push(rest[..cut].trim_end());
        rest = rest[cut..].trim_start();
    }
    if !rest.is_empty() {
        pieces.push(rest);
    }
    pieces
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
                if !seen.insert(part.content.clone()) {
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
                for stretch in stretches(
                    text.trim(),
                    &part.media_type,
                    STRETCH_TOKENS,
                    self.measure,
                    self.sectioner,
                ) {
                    projection.push(
                        RecordBody::Chat {
                            messages: vec![
                                message("user", ASK, false),
                                message("assistant", &stretch, true),
                            ],
                        },
                        Provenance {
                            sources: vec![part.content.clone()],
                            ..Provenance::default()
                        },
                    );
                }
            }
        }
        Ok(projection)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A long text is cut into stretches of whole paragraphs within the
    /// limit, a paragraph longer than the limit at a word boundary, and
    /// nothing is lost or reordered.
    #[test]
    fn a_long_text_is_cut_into_stretches_of_whole_paragraphs() {
        let paragraph = |n: usize| format!("Paragraph {n} says {}.", "something ".repeat(8));
        let body: Vec<String> = (0..10).map(paragraph).collect();
        let body = body.join("\n\n");
        let chars = |text: &str| text.chars().count();
        // Paragraphs at blank lines, as a sectioner divides prose.
        let paragraphs = |text: &str, _: &str| -> Vec<Range<usize>> {
            let mut at = 0;
            text.split("\n\n")
                .map(|p| {
                    let range = at..at + p.len();
                    at += p.len() + 2;
                    range
                })
                .collect()
        };
        let cut = stretches(&body, "text/plain", 300, &chars, &paragraphs);
        assert!(cut.len() > 1);
        for stretch in &cut {
            assert!(stretch.chars().count() <= 300, "{}", stretch.len());
            assert!(
                stretch.starts_with("Paragraph"),
                "a stretch starts on a paragraph"
            );
        }
        assert_eq!(cut.join("\n\n"), body, "nothing is lost or reordered");
        let long = "word ".repeat(100);
        let pieces = cut_to_fit(long.trim(), 120, &chars);
        assert!(pieces.len() >= 4 && pieces.iter().all(|p| p.len() <= 120 && !p.ends_with(' ')));
        assert_eq!(pieces.join(" "), long.trim());
        // Measured in words, a stretch holds whole paragraphs of few words.
        let words = |text: &str| text.split_whitespace().count();
        let by_words = stretches(&body, "text/plain", 25, &words, &paragraphs);
        // Without a sectioner a text is one section, cut at whitespace alone.
        let one = stretches(&body, "text/plain", 300, &chars, &whole_text);
        assert!(one.iter().all(|s| s.chars().count() <= 300));
        assert_eq!(
            one.join(" ").split_whitespace().count(),
            body.split_whitespace().count()
        );
        assert!(by_words.iter().all(|s| words(s) <= 25), "{by_words:?}");
        assert_eq!(by_words.join("\n\n"), body);
        // Without a tokenizer, characters count conservatively as tokens.
        assert_eq!(chars_as_tokens("abcdefg"), 3);
    }
}
