// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements source-grounded persona and expertise
// transfer for language models for its clients. If your team needs expertise
// in keeping a historical record's authorship honest before a model learns
// from it, you can procure our services by sending an email to
// info@swedishembedded.com.

//! How sure an edition's own words leave us that Adams wrote a document.
//!
//! The editor prints, in brackets under each heading, where the text comes
//! from. That note is the evidence: a draft in Adams's papers is nearer his
//! hand than a text a committee adopted, and a newspaper piece is nearer
//! still to an editor's guess than to his signature.

use crate::document::Authorship;

/// A verdict, with the reason it can be checked against.
#[derive(Clone, Debug, PartialEq)]
pub struct Attribution {
    pub authorship: Authorship,
    /// 0..=1, how far the evidence supports the class; ordered within a class.
    pub confidence: f32,
    pub basis: &'static str,
}

/// Newspapers the edition draws Adams's published pieces from.
const NEWSPAPERS: [&str; 8] = [
    "boston gazette",
    "massachusetts spy",
    "independent chronicle",
    "boston evening post",
    "continental journal",
    "boston chronicle",
    "essex gazette",
    "boston news-letter",
];

/// Words in a heading that name a body speaking for many, not one man.
const COLLECTIVE_HEADINGS: [&str; 6] = [
    "town of boston",
    "committee",
    "instructions",
    "resolves",
    "address of",
    "report to the town",
];

fn verdict(authorship: Authorship, confidence: f32, basis: &'static str) -> Attribution {
    Attribution {
        authorship,
        confidence,
        basis,
    }
}

/// Classify a document from its heading and the editor's bracketed source
/// note. Whitespace is normalised first: the OCR doubles and splits it.
pub fn attribute(heading: &str, note: &str) -> Attribution {
    let heading = normalise(heading);
    let note = normalise(note);
    // The Lenox Library held other collections too; only his own papers count.
    let in_his_papers = note.contains("samuel adams papers");

    if note.is_empty() {
        return verdict(
            Authorship::EditorAttributed,
            0.3,
            "no source note; the heading alone says it is his",
        );
    }
    if note.contains("draft") {
        return verdict(
            Authorship::DraftInHand,
            0.9,
            "the editor's note calls it a draft",
        );
    }
    if NEWSPAPERS.iter().any(|paper| note.contains(paper)) {
        return verdict(
            Authorship::PseudonymousAttributed,
            0.6,
            "printed in a newspaper; the attribution is the editor's",
        );
    }
    if note.contains("committee of correspondence")
        || COLLECTIVE_HEADINGS.iter().any(|h| heading.contains(h))
    {
        return verdict(
            Authorship::CommitteeCoauthored,
            0.5,
            "a text adopted by a town or committee he sat on",
        );
    }
    if in_his_papers && note.contains("copy") {
        return verdict(
            Authorship::SignedScribal,
            0.8,
            "a copy in his own papers, in another hand",
        );
    }
    if in_his_papers {
        return verdict(
            Authorship::DraftInHand,
            0.8,
            "a manuscript in his own papers",
        );
    }
    if note.starts_with("[ms") {
        return verdict(
            Authorship::SignedScribal,
            0.7,
            "a manuscript preserved outside his papers, usually by its recipient",
        );
    }
    verdict(
        Authorship::EditorAttributed,
        0.4,
        "known from another printing; the attribution rests on the editor",
    )
}

fn normalise(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn class(heading: &str, note: &str) -> Authorship {
        attribute(heading, note).authorship
    }

    #[test]
    fn a_text_in_his_own_papers_is_a_draft_in_his_hand() {
        let note = "[MS.,  Samuel  Adams  Papers,  Lenox  Library.]";
        assert_eq!(class("TO JOHN DICKINSON.", note), Authorship::DraftInHand);
    }

    #[test]
    fn a_copy_in_his_papers_is_his_text_in_another_hand() {
        let note = "[MS.,  copy  in  Samuel  Adams  Papers,  Lenox  Library.]";
        assert_eq!(class("TO JAMES OTIS.", note), Authorship::SignedScribal);
    }

    #[test]
    fn a_letter_kept_by_its_recipient_is_a_signed_letter() {
        let note = "[MS.,  Collections  of  the  Earl  of  Dartmouth.]";
        assert_eq!(class("TO REVEREND G W", note), Authorship::SignedScribal);
    }

    #[test]
    fn a_note_that_says_draft_beats_where_the_draft_is_kept() {
        let note = "[MS., draft in the Samuel Adams Papers, Lenox Library.]";
        let explicit = attribute("TO JOHN SMITH.", note);
        let plain = attribute(
            "TO JOHN SMITH.",
            "[MS., Samuel Adams Papers, Lenox Library.]",
        );
        assert_eq!(explicit.authorship, Authorship::DraftInHand);
        assert!(explicit.confidence > plain.confidence);
    }

    #[test]
    fn a_newspaper_piece_is_attributed_by_the_editor_not_signed() {
        let note = "[Boston  Gazette,  January  8,  1770.]";
        assert_eq!(
            class("ARTICLE SIGNED DETERMINATUS.", note),
            Authorship::PseudonymousAttributed
        );
    }

    #[test]
    fn what_a_town_or_committee_adopted_is_not_his_private_voice() {
        let town = "[MS., Office of the City Clerk of Boston.]";
        assert_eq!(
            class("THE TOWN OF BOSTON TO THE LIEUTENANT-GOVERNOR", town),
            Authorship::CommitteeCoauthored
        );
        let committee = "[MS., Committee of Correspondence Records, Boston Public Library.]";
        assert_eq!(
            class(
                "TO THE COMMITTEE OF CORRESPONDENCE OF LANCASTER.",
                committee
            ),
            Authorship::CommitteeCoauthored
        );
    }

    #[test]
    fn the_lenox_library_alone_does_not_make_a_text_his_own_papers() {
        let emmet = "[MS., Emmet Collection, Lenox Library.]";
        assert_eq!(
            class("TO RICHARD HENRY LEE.", emmet),
            Authorship::SignedScribal
        );
    }

    #[test]
    fn a_letter_with_no_source_note_rests_on_the_heading_alone() {
        let a = attribute("TO JOHN SMITH.", "");
        assert_eq!(a.authorship, Authorship::EditorAttributed);
        assert!(a.basis.contains("no source note"), "{}", a.basis);
    }

    #[test]
    fn a_text_known_only_from_another_printing_rests_on_the_editor() {
        let note = "[R. H. Lee, Life of Arthur Lee, vol. ii, p. 224.]";
        assert_eq!(class("TO ARTHUR LEE.", note), Authorship::EditorAttributed);
    }

    #[test]
    fn every_verdict_states_a_basis_and_a_confidence_in_range() {
        for (heading, note) in [
            ("TO A.", "[MS., Samuel Adams Papers, Lenox Library.]"),
            ("TO B.", "[Boston Gazette, May 1, 1770.]"),
            ("TO C.", "[something nobody anticipated]"),
        ] {
            let a = attribute(heading, note);
            assert!(
                a.confidence > 0.0 && a.confidence <= 1.0,
                "{note}: {}",
                a.confidence
            );
            assert!(!a.basis.is_empty());
        }
    }
}
