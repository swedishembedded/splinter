// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data built from a writer's own
// text without a model in the loop, for its clients. If your team needs
// expertise in training a model on a person's voice from what they wrote,
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! A writer's text cut into chunks the size of a passage: whole sentences,
//! ended where a paragraph ends when one ends in reach, and never a heading
//! alone.
//!
//! A chunk is the unit a voice record answers with. Cut mid-sentence it
//! teaches the model to start and stop where no writer did; cut to a heading
//! it teaches a heading. So the text is divided into sentences (a
//! paragraph's, as the builder's sectioner divides the text), and a chunk
//! takes sentences while they fit [`Limits`], closing at the last paragraph
//! end that leaves it at least [`Limits::min_words`] long.

use std::ops::Range;

use super::{Measure, Sectioner};

/// How large a chunk may be, in the words it holds and in the tokens the
/// policy counts them as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// A chunk closes at a paragraph end only once it holds this many words.
    pub min_words: usize,
    /// The most words a chunk holds.
    pub max_words: usize,
    /// The most tokens a chunk holds, as the builder's measure counts them:
    /// what keeps a chunk within the row a run trains in however its words
    /// tokenize.
    pub max_tokens: usize,
    /// The fewest words, a heading not counted, a chunk must hold to be a
    /// record: under it is a heading, a signature or a stub.
    pub min_body_words: usize,
}

/// A section of at most this many words that does not end a sentence is a
/// heading (`To E. Randolph`, `Monticello, 1794`, `CHAPTER II`): it is never
/// the end of a chunk, and the chunk that opens a text starts with it.
const HEADING_WORDS: usize = 14;

/// One chunk of a text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunk {
    /// Where it is in the text, so its words are the text's own, spaces and
    /// line breaks included.
    pub range: Range<usize>,
    /// The words it holds, a heading not counted.
    pub body_words: usize,
    /// Whether it opens the text (and so carries the text's own heading).
    pub opens_text: bool,
    /// Where its first sentence of the body starts: after the heading the
    /// chunk opens with, else at its start.
    pub body_start: usize,
}

/// A sentence, or a heading, of the text.
struct Unit {
    range: Range<usize>,
    words: usize,
    tokens: usize,
    /// The unit ends its paragraph.
    ends_paragraph: bool,
    /// The unit is a heading.
    heading: bool,
}

/// `text` as chunks within `limits`, and the text's own heading when it has
/// one (the sections before the first sentence).
pub fn chunks(
    text: &str,
    media_type: &str,
    limits: &Limits,
    measure: Measure<'_>,
    sectioner: Sectioner<'_>,
) -> (Vec<Chunk>, Option<Range<usize>>) {
    let sections: Vec<Range<usize>> = sectioner(text, media_type)
        .into_iter()
        .filter(|r| text.get(r.clone()).is_some_and(|s| !s.trim().is_empty()))
        .collect();
    let mut units: Vec<Unit> = Vec::new();
    let mut heading = None;
    for (n, section) in sections.iter().enumerate() {
        let body = &text[section.clone()];
        let is_heading = n + 1 < sections.len() && looks_like_heading(body);
        if is_heading {
            if units.iter().all(|u| u.heading) {
                heading =
                    Some(heading.map_or(section.clone(), |h: Range<usize>| h.start..section.end));
            }
            units.push(unit(text, section.clone(), true, true, measure));
            continue;
        }
        let sentences = sentences_of(body);
        let last = sentences.len().saturating_sub(1);
        for (i, s) in sentences.into_iter().enumerate() {
            let range = section.start + s.start..section.start + s.end;
            units.push(unit(text, range, i == last, false, measure));
        }
    }
    (pack(&units, limits, text), heading)
}

fn unit(
    text: &str,
    range: Range<usize>,
    ends_paragraph: bool,
    heading: bool,
    measure: Measure<'_>,
) -> Unit {
    let piece = &text[range.clone()];
    Unit {
        words: piece.split_whitespace().count(),
        tokens: measure(piece),
        range,
        ends_paragraph,
        heading,
    }
}

/// A short section that does not end a sentence.
fn looks_like_heading(section: &str) -> bool {
    let words = section.split_whitespace().count();
    words <= HEADING_WORDS && !ends_sentence(section.trim_end())
}

fn ends_sentence(text: &str) -> bool {
    text.trim_end_matches(['"', '\'', ')'])
        .ends_with(['.', '?', '!'])
}

/// `section`'s sentences as byte ranges in it: a sentence ends at `.`, `?`
/// or `!` followed by space and a capital, a digit or a quote, unless the
/// word it ends is an abbreviation or an initial (a capitalised word of up
/// to three letters: `Mr.`, `Esq.`, `E.`).
fn sentences_of(section: &str) -> Vec<Range<usize>> {
    let bytes = section.as_bytes();
    let mut out = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < bytes.len() {
        if matches!(bytes[i], b'.' | b'?' | b'!') {
            let mut end = i + 1;
            while end < bytes.len() && matches!(bytes[end], b'"' | b'\'' | b')') {
                end += 1;
            }
            let rest = &section[end..];
            let spaced = rest.starts_with(char::is_whitespace);
            let next = rest.trim_start().chars().next();
            let opens = next.is_none_or(|c| c.is_uppercase() || c.is_ascii_digit() || c == '"');
            if spaced && opens && !abbreviation_before(&section[start..i]) {
                out.push(start..end);
                start = end + (rest.len() - rest.trim_start().len());
                i = start;
                continue;
            }
        }
        i += 1;
    }
    if start < section.len() {
        out.push(start..section.len());
    }
    out.retain(|r| !section[r.clone()].trim().is_empty());
    out
}

/// Whether the word that ends `before` is an abbreviation or an initial.
fn abbreviation_before(before: &str) -> bool {
    let word = before
        .rsplit(char::is_whitespace)
        .next()
        .unwrap_or("")
        .trim_start_matches(['(', '"', '\'']);
    let letters = word.chars().count();
    letters > 0
        && letters <= 3
        && word.chars().next().is_some_and(char::is_uppercase)
        && word.chars().all(char::is_alphabetic)
        && !matches!(word, "I" | "A")
        || matches!(word, "i.e" | "e.g" | "viz" | "etc" | "St" | "Mrs" | "No")
}

/// The units packed into chunks.
fn pack(units: &[Unit], limits: &Limits, text: &str) -> Vec<Chunk> {
    let mut chunks: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;
    while start < units.len() {
        let (mut words, mut tokens) = (0, 0);
        // The end of the last paragraph that leaves the chunk long enough.
        let mut paragraph_end: Option<usize> = None;
        let mut end = start;
        while end < units.len() {
            let u = &units[end];
            let fits = words + u.words <= limits.max_words && tokens + u.tokens < limits.max_tokens;
            if !fits && end > start {
                break;
            }
            words += u.words;
            tokens += u.tokens + 1;
            end += 1;
            if u.ends_paragraph && words >= limits.min_words {
                paragraph_end = Some(end);
            }
        }
        let mut cut = if end < units.len() {
            paragraph_end.unwrap_or(end)
        } else {
            end
        };
        // A chunk never ends on a heading that opens what follows.
        while cut > start + 1 && cut < units.len() && units[cut - 1].heading {
            cut -= 1;
        }
        chunks.push((start, cut));
        start = cut;
    }
    merge_tail(&mut chunks, units, limits);
    chunks
        .into_iter()
        .filter_map(|(a, b)| chunk_of(&units[a..b], a == 0, limits, text))
        .collect()
}

/// A tail too short to stand joins the chunk before it when that still fits.
fn merge_tail(chunks: &mut Vec<(usize, usize)>, units: &[Unit], limits: &Limits) {
    let [.., (a, _), (b, c)] = chunks[..] else {
        return;
    };
    let words = |from: usize, to: usize| units[from..to].iter().map(|u| u.words).sum::<usize>();
    let tokens =
        |from: usize, to: usize| units[from..to].iter().map(|u| u.tokens + 1).sum::<usize>();
    if words(b, c) < limits.min_words
        && words(a, c) <= limits.max_words
        && tokens(a, c) <= limits.max_tokens
    {
        chunks.truncate(chunks.len() - 2);
        chunks.push((a, c));
    }
}

/// The chunk of `units`, or `None` when it holds too few words of body.
fn chunk_of(units: &[Unit], opens_text: bool, limits: &Limits, _text: &str) -> Option<Chunk> {
    let first = units.first()?;
    let last = units.last()?;
    let body_words: usize = units.iter().filter(|u| !u.heading).map(|u| u.words).sum();
    if body_words < limits.min_body_words {
        return None;
    }
    let body_start = units
        .iter()
        .find(|u| !u.heading)
        .map_or(first.range.start, |u| u.range.start);
    Some(Chunk {
        range: first.range.start..last.range.end,
        body_words,
        opens_text: opens_text && first.heading,
        body_start,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paragraphs(text: &str, _: &str) -> Vec<Range<usize>> {
        let mut at = 0;
        text.split("\n\n")
            .map(|p| {
                let range = at..at + p.len();
                at += p.len() + 2;
                range
            })
            .collect()
    }

    fn words(text: &str) -> usize {
        text.split_whitespace().count()
    }

    fn limits() -> Limits {
        Limits {
            min_words: 40,
            max_words: 100,
            max_tokens: 150,
            min_body_words: 15,
        }
    }

    /// A paragraph of `n` sentences of ten words each.
    fn paragraph(label: &str, n: usize) -> String {
        (0..n)
            .map(|s| format!("Sentence {label}{s} says nothing of note but takes up ten words."))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn cut(text: &str) -> (Vec<String>, Option<String>) {
        let (chunks, heading) = chunks(text, "text/plain", &limits(), &words, &paragraphs);
        (
            chunks
                .iter()
                .map(|c| text[c.range.clone()].to_string())
                .collect(),
            heading.map(|h| text[h].to_string()),
        )
    }

    /// A chunk is whole sentences within the limits that close where a
    /// paragraph ends, and the words of the chunks are the text's, in order.
    #[test]
    fn a_text_is_cut_into_whole_sentences_closing_at_paragraph_ends() {
        let body = [
            paragraph("a", 4),
            paragraph("b", 4),
            paragraph("c", 8),
            paragraph("d", 3),
        ];
        let text = body.join("\n\n");
        let (cut, heading) = cut(&text);
        assert_eq!(heading, None);
        assert!(cut.len() >= 3, "{cut:?}");
        for chunk in &cut {
            assert!(words(chunk) <= 100, "{}", words(chunk));
            assert!(chunk.ends_with("ten words."), "ends on a sentence: {chunk}");
            assert!(
                chunk.starts_with("Sentence "),
                "starts on a sentence: {chunk}"
            );
        }
        // The first chunk takes both short paragraphs (80 words) and stops
        // at the paragraph end rather than 20 words into the third.
        assert_eq!(cut[0], format!("{}\n\n{}", body[0], body[1]));
        let rejoined: Vec<&str> = cut.iter().flat_map(|c| c.split_whitespace()).collect();
        let original: Vec<&str> = text.split_whitespace().collect();
        assert_eq!(rejoined, original, "nothing lost or reordered");
    }

    /// A letter's address lines go with the body, a chunk never ends on a
    /// heading, and a text with nothing but a heading and a stub is no record.
    #[test]
    fn a_heading_is_never_a_chunk_of_its_own() {
        let letter = format!(
            "To E. Randolph\n\nMonticello, 1794\n\n{}\n\n{}",
            paragraph("x", 9),
            paragraph("y", 5)
        );
        let (cut, heading) = cut(&letter);
        assert_eq!(
            heading.as_deref(),
            Some("To E. Randolph\n\nMonticello, 1794")
        );
        assert!(
            cut[0].starts_with("To E. Randolph\n\nMonticello, 1794\n\nSentence x0"),
            "{}",
            cut[0]
        );
        assert!(cut.iter().all(|c| words(c) > 20), "{cut:?}");

        let (cut, _) = cut_of("To Albert Gallatin, Esq\n\nMonticello, 1810");
        assert!(cut.is_empty(), "a heading alone is not a record: {cut:?}");
        let (cut, _) = cut_of("To a friend\n\nMonticello, 1800\n\nI am well.");
        assert!(cut.is_empty(), "a stub is not a record: {cut:?}");
    }

    fn cut_of(text: &str) -> (Vec<String>, Option<String>) {
        cut(text)
    }

    /// A paragraph longer than the limit is cut between sentences, and an
    /// abbreviation or an initial does not end one.
    #[test]
    fn a_long_paragraph_is_cut_between_sentences_not_at_abbreviations() {
        let text = format!(
            "Mr. E. Randolph, Esq. wrote to me; I replied at length. {}",
            paragraph("q", 20)
        );
        let (cut, _) = cut(&text);
        assert!(cut.len() >= 2);
        assert!(cut[0].starts_with("Mr. E. Randolph, Esq. wrote to me; I replied at length."));
        for chunk in &cut {
            assert!(chunk.ends_with('.'), "{chunk}");
            assert!(words(chunk) <= 100);
        }
    }

    /// The token limit binds as well as the word limit.
    #[test]
    fn a_chunk_stays_within_the_token_limit() {
        let text = [paragraph("a", 4), paragraph("b", 4), paragraph("c", 4)].join("\n\n");
        let doubled = |t: &str| 2 * words(t);
        let tight = Limits {
            max_tokens: 100,
            ..limits()
        };
        let (chunks, _) = chunks(&text, "text/plain", &tight, &doubled, &paragraphs);
        assert!(chunks.len() >= 3);
        assert!(chunks
            .iter()
            .all(|c| doubled(&text[c.range.clone()]) <= 100));
    }
}
