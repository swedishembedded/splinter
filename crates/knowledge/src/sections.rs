// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements delegated-task coding agents that grow
// their own training data. If your team needs expertise in data extraction
// from technical documents, you can procure our services by sending an
// email to info@swedishembedded.com.

//! A document split into the sections a model reads one at a time, and
//! the subject every section and question is anchored on: the title.

/// Splits `text` into sections at markdown headings (`##` / `###`).
///
/// Three invariants:
/// - A table row is never split from its section: a heading line or a
///   blank line opens a new chunk, and a markdown table contains neither,
///   so a table always stays whole.
/// - `chunk_lines` caps a section's size: when a section exceeds the cap,
///   the next heading starts a fresh chunk (content in flight is kept
///   with the section that holds it - splitting mid-table is what the
///   cap must never cause).
/// - A section that stays over the cap with no heading in sight (the
///   common fact-sheet shape: one `#`, then bullets) also chunks at the
///   next blank line. Without this, a heading-less document was ONE
///   section however large - too much for a small generator to enumerate,
///   which is how coverage collapses. The split point is a paragraph
///   boundary, and a heading is never parked alone: if the content in
///   flight is only a heading, it stays for the paragraph that follows.
/// - Every section carries the document's title line (the first `#`
///   heading). Chunking cuts later sections off from the document's
///   subject, and the generator is told to use the identifiers its
///   section shows - a chunk that no longer names the device cannot
///   produce an anchored question about it.
pub fn split_sections(text: &str, chunk_lines: Option<usize>) -> Vec<String> {
    let mut sections: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut overflow = false;
    let title = document_title(text);
    for line in text.lines() {
        let is_heading = line.starts_with("## ") || line.starts_with("### ");
        let mut close = is_heading && (!current.trim().is_empty() || overflow);
        if !close {
            if let Some(cap) = chunk_lines {
                let is_blank = line.trim().is_empty();
                let last_non_empty_is_heading = current
                    .lines()
                    .rev()
                    .find(|l| !l.trim().is_empty())
                    .is_some_and(|l| l.starts_with('#'));
                close = is_blank
                    && !current.trim().is_empty()
                    && !last_non_empty_is_heading
                    && current.lines().count() >= cap;
            }
        }
        if close {
            sections.push(std::mem::take(&mut current));
            overflow = false;
        }
        current.push_str(line);
        current.push('\n');
        if let Some(cap) = chunk_lines {
            if current.lines().count() >= cap {
                // Over the cap: the next heading or paragraph boundary
                // MUST open a new chunk.
                overflow = true;
            }
        }
    }
    if !current.trim().is_empty() {
        sections.push(current);
    }
    // Re-unite every section with the document's subject, and drop a
    // section that holds nothing but it - a title with no content gives
    // the generator no facts to enumerate.
    if let Some(title) = title {
        for section in &mut sections {
            if !section.contains(title) {
                let content = section.trim_start();
                *section = format!("{title}\n\n{content}");
            }
        }
        sections.retain(|section| section.lines().any(|l| !l.trim().is_empty() && l != title));
    }
    sections
}

/// Runs the whole exploration, tracing to its own run dir like `run` does:
/// manifest, one event per section, and an outcome with the counts.
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
