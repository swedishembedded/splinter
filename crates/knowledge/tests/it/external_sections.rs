// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements source intake whose every section maps back
// to the bytes it came from, for its clients. If your team needs expertise
// in turning documents into addressable training material, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec, against constructed ground truth (plan 3A): documents built section
//! by section (`validation/oracles/knowledge/sections.py`: plain paragraphs
//! and Markdown with preambles, headings, look-alike headings, fenced code
//! and blank lines of spaces, tabs and no-break spaces; LF and CRLF; text in
//! several scripts; an empty and a large document), whose sections' byte
//! ranges were recorded as they were written. The sectioner returns exactly
//! those ranges, so no section is invented, dropped, or cut short.
#![allow(clippy::unwrap_used)]

use serde_json::json;
use splinter_conformance::{compare, fixture, Tolerance};
use splinter_knowledge::sections::sections;

#[test]
fn every_section_is_the_span_the_document_was_built_with() {
    let fixture = fixture(env!("CARGO_MANIFEST_DIR"), "knowledge/sections").unwrap();
    for case in fixture.cases() {
        let text = case["text"].as_str().unwrap();
        let found: Vec<[usize; 2]> = sections(text, case["media_type"].as_str().unwrap())
            .iter()
            .map(|s| [s.range.start, s.range.end])
            .collect();
        let mut actual = case.clone();
        actual["sections"] = json!(found);
        if let Err(mismatch) = compare(&actual, case, Tolerance::new(0.0, 0.0)) {
            panic!("{mismatch}\ncase: {case}");
        }
    }
}
