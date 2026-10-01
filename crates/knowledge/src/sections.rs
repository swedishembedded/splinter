// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements source intake that turns technical
// documents into addressable training material. If your team needs
// expertise in data extraction from technical documents, you can procure
// our services by sending an email to info@swedishembedded.com.

//! A text split into addressable sections.
//!
//! [`sections`] is the one sectioner: each [`Section`] is a byte range of
//! the text, so a section of a stored source part is a span of it.
//! Markdown splits at ATX headings (`#` to `######`, outside fenced code);
//! any other text splits into blank-line paragraphs. A section starts at
//! its first non-blank line and ends with its last non-blank line, line
//! terminator excluded; blank lines between sections belong to none.

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
        markdown_sections(text)
    } else {
        paragraphs(text)
    }
}

/// A Markdown text's title: its first heading, without the `#` marks
/// around it; `None` for any other media type, or a text with no heading.
#[must_use]
pub fn title(text: &str, media_type: &str) -> Option<String> {
    if media_type != MARKDOWN {
        return None;
    }
    let heading = markdown_sections(text)
        .iter()
        .filter_map(|section| section.text(text)?.lines().next())
        .find(|line| is_heading(line))?;
    let words = heading.trim().trim_start_matches('#').trim();
    // A closing run of `#` counts only after a space: `C#` keeps its mark.
    let open = words.trim_end_matches('#');
    let words = if open.is_empty() || open.ends_with([' ', '\t']) {
        open.trim_end()
    } else {
        words
    };
    (!words.is_empty()).then(|| words.to_string())
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

/// A section being built: the byte range of its non-blank lines so far.
#[derive(Default)]
struct Open {
    range: Option<Range<usize>>,
}

impl Open {
    fn push(&mut self, line: &Line<'_>) {
        if !line.is_blank() {
            let start = self.range.as_ref().map_or(line.start, |r| r.start);
            self.range = Some(start..line.end);
        }
    }

    fn close(&mut self, out: &mut Vec<Section>) {
        if let Some(range) = std::mem::take(self).range {
            out.push(Section { range });
        }
    }
}

/// Splits Markdown `text` at its headings: a heading always opens a new
/// section, and nothing inside a fenced code block splits it.
fn markdown_sections(text: &str) -> Vec<Section> {
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
        if !fenced && is_heading(line.text) && open.range.is_some() {
            open.close(&mut out);
        }
        open.push(&line);
    }
    open.close(&mut out);
    out
}

/// Splits `text` into paragraphs: runs of non-blank lines.
fn paragraphs(text: &str) -> Vec<Section> {
    let mut out = Vec::new();
    let mut open = Open::default();
    for line in lines(text) {
        if line.is_blank() {
            open.close(&mut out);
        } else {
            open.push(&line);
        }
    }
    open.close(&mut out);
    out
}
