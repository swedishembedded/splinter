// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements delegated-task coding agents that grow
// their own training data. If your team needs expertise in data extraction
// from technical documents, you can procure our services by sending an
// email to info@swedishembedded.com.

//! A text split into addressable sections, and the subject every section
//! and question of a fact sheet is anchored on: the title.
//!
//! [`sections`] is the one sectioner: each [`Section`] is a byte range of
//! the text, so a section of a stored source part is a span of it.
//! Markdown splits at ATX headings (`#` to `######`, outside fenced code);
//! any other text splits into blank-line paragraphs. A section starts at
//! its first non-blank line and ends with its last non-blank line, line
//! terminator excluded; blank lines between sections belong to none.
//! [`split_sections`] is the fact-extraction view of the same split: capped
//! in size, and with the document's title re-attached to every chunk.

use std::ops::Range;

/// The media type that splits at headings; every other text splits into
/// paragraphs.
pub const MARKDOWN: &str = "text/markdown";

/// One section of a text: the bytes `range` covers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Section {
    /// Its byte range in the text; both ends are on line boundaries.
    pub range: Range<usize>,
}

impl Section {
    /// The section's text within `text`, the text it was split from.
    #[must_use]
    pub fn text<'a>(&self, text: &'a str) -> Option<&'a str> {
        text.get(self.range.clone())
    }
}

/// Splits `text` of media type `media_type` into sections: at headings for
/// [`MARKDOWN`], at blank lines otherwise.
#[must_use]
pub fn sections(text: &str, media_type: &str) -> Vec<Section> {
    if media_type == MARKDOWN {
        markdown_sections(text, None)
    } else {
        paragraphs(text)
    }
}

/// One line of a text: its byte range without the terminator, and what the
/// sectioners need to know about it.
struct Line<'a> {
    start: usize,
    end: usize,
    text: &'a str,
}

impl Line<'_> {
    fn is_blank(&self) -> bool {
        self.text.trim().is_empty()
    }
}

fn lines(text: &str) -> impl Iterator<Item = Line<'_>> {
    let mut offset = 0;
    text.split_inclusive('\n').map(move |raw| {
        let start = offset;
        offset += raw.len();
        let content = raw.strip_suffix('\n').unwrap_or(raw);
        let content = content.strip_suffix('\r').unwrap_or(content);
        Line {
            start,
            end: start + content.len(),
            text: content,
        }
    })
}

/// Whether `line` is an ATX heading: up to three spaces, one to six `#`,
/// then whitespace or the end of the line.
fn is_heading(line: &str) -> bool {
    let indent = line.len() - line.trim_start_matches(' ').len();
    let rest = &line[indent..];
    let hashes = rest.len() - rest.trim_start_matches('#').len();
    indent <= 3
        && (1..=6).contains(&hashes)
        && rest[hashes..]
            .chars()
            .next()
            .is_none_or(|c| c == ' ' || c == '\t')
}

/// The fence a line opens or closes (a run of three or more backticks or
/// tildes after up to three spaces), as its character and length.
fn fence(line: &str) -> Option<(char, usize)> {
    let indent = line.len() - line.trim_start_matches(' ').len();
    let rest = &line[indent..];
    let marker = rest.chars().next().filter(|c| *c == '`' || *c == '~')?;
    let run = rest.len() - rest.trim_start_matches(marker).len();
    (indent <= 3 && run >= 3).then_some((marker, run))
}

/// A section being built: the lines so far, and what closing it needs.
#[derive(Default)]
struct Open {
    /// Byte range of its non-blank lines so far.
    range: Option<Range<usize>>,
    /// Lines in it, blank ones included.
    lines: usize,
    /// Whether its last non-blank line is a heading.
    ends_in_heading: bool,
}

impl Open {
    fn push(&mut self, line: &Line<'_>, heading: bool) {
        self.lines += 1;
        if !line.is_blank() {
            let start = self.range.as_ref().map_or(line.start, |r| r.start);
            self.range = Some(start..line.end);
            self.ends_in_heading = heading;
        }
    }

    fn close(&mut self, out: &mut Vec<Section>) {
        if let Some(range) = std::mem::take(self).range {
            out.push(Section { range });
        }
    }
}

/// Splits Markdown `text` at its headings. A heading always opens a new
/// section. With `chunk_lines`, a section that reaches that many lines
/// (blank ones included) also closes at its next blank line, unless all it
/// holds so far is headings: a heading never ends up alone. Nothing inside
/// a fenced code block splits it.
#[must_use]
pub fn markdown_sections(text: &str, chunk_lines: Option<usize>) -> Vec<Section> {
    let mut out = Vec::new();
    let mut open = Open::default();
    let mut in_fence: Option<(char, usize)> = None;
    for line in lines(text) {
        let fenced = in_fence.is_some();
        match (in_fence, fence(line.text)) {
            (None, Some(opened)) => in_fence = Some(opened),
            (Some((c, n)), Some((c2, n2)))
                if c == c2 && n2 >= n && line.text.trim().len() == n2 =>
            {
                in_fence = None;
            }
            _ => {}
        }
        let heading = !fenced && is_heading(line.text);
        let overflowing = chunk_lines.is_some_and(|cap| open.lines >= cap);
        let boundary =
            heading || (!fenced && line.is_blank() && overflowing && !open.ends_in_heading);
        if boundary && open.range.is_some() {
            open.close(&mut out);
        }
        open.push(&line, heading);
    }
    open.close(&mut out);
    out
}

/// Splits `text` into paragraphs: runs of non-blank lines.
#[must_use]
pub fn paragraphs(text: &str) -> Vec<Section> {
    let mut out = Vec::new();
    let mut open = Open::default();
    for line in lines(text) {
        if line.is_blank() {
            open.close(&mut out);
        } else {
            open.push(&line, false);
        }
    }
    open.close(&mut out);
    out
}

/// The sections a fact extractor reads: [`markdown_sections`] capped at
/// `chunk_lines`, as text.
///
/// Every section carries the document's title line (the first `#`
/// heading). Chunking cuts later sections off from the document's subject,
/// and the extractor is told to use the identifiers its section shows - a
/// chunk that no longer names the device cannot produce an anchored
/// question about it. A section that holds nothing but the title gives the
/// extractor no facts to enumerate, and is dropped.
#[must_use]
pub fn split_sections(text: &str, chunk_lines: Option<usize>) -> Vec<String> {
    let title = document_title(text);
    markdown_sections(text, chunk_lines)
        .into_iter()
        .filter_map(|section| section.text(text))
        .filter(|section| {
            section
                .lines()
                .any(|l| !l.trim().is_empty() && Some(l) != title)
        })
        .map(|section| match title {
            Some(title) if !section.contains(title) => format!("{title}\n\n{section}\n"),
            _ => format!("{section}\n"),
        })
        .collect()
}

/// The document's title line - the first level-1 heading - which names
/// the subject every chunk must carry and every question must anchor on.
pub fn document_title(text: &str) -> Option<&str> {
    text.lines().find(|l| l.starts_with("# "))
}

/// The device identifiers a document title names: alphanumeric tokens of
/// four or more characters that mix letters and digits ("STM32F407",
/// "DS8626"). A pure-word title names no device, and the anchor gate
/// disables itself for such a document rather than rejecting everything.
pub fn title_identifiers(title: &str) -> Vec<String> {
    title
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| {
            t.len() >= 4
                && t.chars().any(|c| c.is_ascii_digit())
                && t.chars().any(|c| c.is_ascii_alphabetic())
        })
        .map(str::to_string)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Headings split; a table stays with its heading's section.
    #[test]
    fn sections_split_at_headings_and_tables_stay_whole() {
        let text = "# Title\n\nintro\n\n## Timers\n\nTIM1 is 16-bit.\n\n\
            | Block | Limit |\n|---|---|\n| SPI1 | 42 Mbit/s |\n\n## Clocks\n\nHSI is 16 MHz.\n";
        let sections = split_sections(text, None);
        assert_eq!(sections.len(), 3, "{sections:#?}");
        assert!(sections[1].contains("TIM1 is 16-bit."));
        // The whole table rides with the section that opened it.
        assert!(sections[1].contains("| SPI1 | 42 Mbit/s |"));
        assert!(sections[2].contains("HSI is 16 MHz."));
    }

    /// A section over the cap starts a new chunk at the next heading - and
    /// content in flight stays with the section that holds it, so the cap
    /// never tears a table apart.
    #[test]
    fn chunk_lines_caps_sections_at_the_next_heading() {
        let text = "## A\n\none\n\n## B\n\ntwo\n\n## C\n\nthree\n";
        let sections = split_sections(text, Some(4));
        assert_eq!(sections.len(), 3, "{sections:#?}");
        assert!(sections[0].starts_with("## A"));
        assert!(sections[1].starts_with("## B"));
        assert!(sections[2].starts_with("## C"));
    }

    /// A heading-less document must still chunk: many fact sheets use a
    /// single `#` plus bullets, and under the old heading-only rule the
    /// whole document was one section - too large for a small generator,
    /// which then enumerates a fraction of the claims. Blank lines are the
    /// paragraph boundaries every markdown document has.
    #[test]
    fn an_overflowing_headingless_document_chunks_at_paragraph_boundaries() {
        let text = "# Device\n\nalpha fact\n\nbeta fact\n\ngamma fact\n\ndelta fact\n";
        let sections = split_sections(text, Some(2));
        assert_eq!(sections.len(), 4, "{sections:#?}");
        assert!(sections[0].contains("# Device"));
        assert!(sections[0].contains("alpha fact"), "{}", sections[0]);
        assert!(sections[1].contains("beta fact"), "{}", sections[1]);
        assert!(sections[2].contains("gamma fact"), "{}", sections[2]);
        assert!(sections[3].contains("delta fact"), "{}", sections[3]);
    }

    /// The boundary rule must never park a heading alone: a heading with
    /// nothing under it gives the generator no facts to enumerate.
    #[test]
    fn a_chunk_boundary_never_orphans_a_heading() {
        let text = "# Device\n\n## Clocks\n\nHSI is 16 MHz.\n\nLSI is 32 kHz.\n";
        let sections = split_sections(text, Some(2));
        assert_eq!(sections.len(), 2, "{sections:#?}");
        assert!(sections[0].contains("## Clocks"));
        // The heading rides with its first paragraph, never alone.
        assert!(sections[0].contains("HSI is 16 MHz."), "{}", sections[0]);
        assert!(sections[1].contains("LSI is 32 kHz."), "{}", sections[1]);
    }

    /// Chunks past the document's own section are cut off from the
    /// document's subject: the generator is told to use the exact
    /// identifiers from the section it sees, so a chunk that no longer
    /// contains the chip name produces unanchored questions ("What is the
    /// maximum frequency of the SPI/I²S?") - the anchoring instruction
    /// cannot be followed from a section that never names the device.
    /// Every chunk therefore carries the document's title line.
    #[test]
    fn every_chunk_carries_the_document_subject() {
        let text = "# STM32F407 fact sheet\n\n## Clocks\n\nHSI is 16 MHz.\n\n## Timers\n\nTIM1 is 16-bit.\n";
        let sections = split_sections(text, None);
        assert_eq!(sections.len(), 2, "{sections:#?}");
        for s in &sections {
            assert!(
                s.contains("STM32F407 fact sheet"),
                "chunk lost the document subject: {s}"
            );
        }
    }
}
