// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: a text part splits into addressable sections - at Markdown
//! headings for Markdown, at blank lines otherwise - whose byte ranges slice
//! back to exactly the section's text.

use splinter_knowledge::sections::sections;

fn texts<'a>(text: &'a str, media_type: &str) -> Vec<&'a str> {
    sections(text, media_type)
        .iter()
        .map(|s| &text[s.range.clone()])
        .collect()
}

#[test]
fn markdown_splits_at_headings_and_ranges_slice_back() {
    let text = "# Tool\n\nIntro line.\n\n## Usage\n\nRun it.\n\n```sh\n# not a heading\n\necho hi\n```\n\n### Flags\n\n--help prints help.\n";
    assert_eq!(
        texts(text, "text/markdown"),
        [
            "# Tool\n\nIntro line.",
            "## Usage\n\nRun it.\n\n```sh\n# not a heading\n\necho hi\n```",
            "### Flags\n\n--help prints help.",
        ]
    );
}

#[test]
fn plain_text_splits_into_blank_line_paragraphs() {
    let text = "\nfirst line\nstill first\n\n  \nsecond\r\n\r\nthird";
    assert_eq!(
        texts(text, "text/plain"),
        ["first line\nstill first", "second", "third"]
    );
    assert!(sections("", "text/plain").is_empty());
}
