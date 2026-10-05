// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements provenance-tracked training corpora for
// language models for its clients. If your team needs expertise in turning a
// curated historical record into the directory a learning system is pointed
// at, with nothing in it that is not the person's own, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The directory Splinter learns from: his own documents of the training
//! split, one file each, with the editors' apparatus taken out.
//!
//! What Splinter is told to learn from must be his and only his. A text an
//! editor ascribes to him, or printed under a pseudonym, is kept out, however
//! likely the ascription: a model taught it as his voice learns a composite.
//! A text a town or committee he sat on adopted is his work but not his
//! private voice, so it goes under a directory of its own, which names it to
//! whoever reads the file. Every document of a held-out family (the exam and
//! the temporal test) is left out whole, so what Splinter learns from leaves
//! the independent check unseen. The stored text is not changed: the split
//! and its digests rest on it; only what is written here is cleaned.

use std::collections::BTreeMap;

use crate::apparatus::without_apparatus;
use crate::curate::Document;
use crate::document::Date;
use crate::split::{Assignment, Split};

/// The directory a document of his own hand, or signed by him, is written
/// under.
pub const OWN_DIR: &str = "own";
/// The directory a text a town or committee he sat on adopted is written
/// under.
pub const COMMITTEE_DIR: &str = "committee";

/// One file of the materials: its path under the output directory, and its
/// text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct File {
    /// `own/<id>.txt` or `committee/<id>.txt`.
    pub path: String,
    /// The heading, the date, a blank line and the text without apparatus.
    pub text: String,
}

/// The files Splinter is pointed at, in document order: every document of
/// the training split that may be taught as his voice.
#[must_use]
pub fn materials(documents: &[Document], assignments: &[Assignment]) -> Vec<File> {
    let split_of: BTreeMap<&str, Split> = assignments
        .iter()
        .map(|a| (a.doc_id.as_str(), a.split))
        .collect();
    documents
        .iter()
        .filter(|d| split_of.get(d.id.as_str()) == Some(&Split::Train) && d.authorship.is_voice())
        .map(|d| {
            let dir = if d.authorship.is_his_own_letter() {
                OWN_DIR
            } else {
                COMMITTEE_DIR
            };
            File {
                path: format!("{dir}/{}.txt", d.id),
                text: format!(
                    "{}\n{}\n\n{}\n",
                    heading_line(d),
                    date_line(d.date),
                    without_apparatus(&d.body)
                ),
            }
        })
        .collect()
}

/// `To James Warren` for a letter; the printed heading in title case for
/// anything else, its trailing stop and footnote marks dropped, and a date
/// the heading ends in left to the date line.
fn heading_line(document: &Document) -> String {
    match &document.recipient {
        Some(recipient) => format!("To {recipient}"),
        None => {
            let heading = document.heading.as_str();
            let dated = heading.rfind(". ").filter(|&at| {
                let tail = heading[at..].to_lowercase();
                tail.contains(|c: char| c.is_ascii_digit())
                    || MONTHS.iter().any(|m| tail.contains(&m.to_lowercase()))
            });
            title_case(
                dated
                    .map_or(heading, |at| &heading[..at])
                    .trim_end_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ' '),
            )
        }
    }
}

const MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

/// `May 12, 1776`, `May 1776` or `1776`: as much of the date as is known.
fn date_line(date: Date) -> String {
    let month = date
        .month
        .and_then(|m| MONTHS.get(usize::from(m).wrapping_sub(1)));
    match (month, date.day) {
        (Some(month), Some(day)) => format!("{month} {day}, {}", date.year),
        (Some(month), None) => format!("{month} {}", date.year),
        _ => date.year.to_string(),
    }
}

/// `INSTRUCTIONS OF THE TOWN` as `Instructions of the Town`: small joining
/// words stay lower case.
fn title_case(heading: &str) -> String {
    heading
        .split_whitespace()
        .enumerate()
        .map(|(n, word)| {
            let lower = word.to_lowercase();
            if n > 0 && matches!(lower.as_str(), "of" | "and" | "the" | "to" | "in" | "on") {
                lower
            } else {
                let mut chars = lower.chars();
                chars
                    .next()
                    .map(|c| c.to_uppercase().collect::<String>() + chars.as_str())
                    .unwrap_or_default()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Authorship, Period};

    fn document(id: &str, authorship: Authorship, recipient: Option<&str>, body: &str) -> Document {
        Document {
            schema: crate::curate::SCHEMA,
            id: id.into(),
            source_id: "cushing-1".into(),
            heading: match recipient {
                Some(r) => format!("TO {}.", r.to_uppercase()),
                None => "INSTRUCTIONS OF THE TOWN OF BOSTON TO ITS REPRESENTATIVES.1".into(),
            },
            recipient: recipient.map(str::to_string),
            note: "[MS.]".into(),
            date: Date {
                year: 1772,
                month: Some(11),
                day: Some(2),
            },
            period: Period::of(1772),
            authorship,
            authorship_confidence: 0.8,
            authorship_basis: "test".into(),
            temporal_holdout: false,
            body: body.into(),
        }
    }

    fn assigned(id: &str, split: Split) -> Assignment {
        Assignment {
            doc_id: id.into(),
            family: format!("family-{id}"),
            split,
        }
    }

    const SCANNED: &str = "My dear Sir,\n\nI have your favor of the tenth.\n\n[MS., Lenox Library.]\n\n1 A copy is in S. A. Wells, vol. i., pp. 363, 364.\n\ni7?0 SAMUEL ADAMS. 135\n\nYours, S. A.";

    /// Only his own documents of the training split are written: held-out
    /// families, ascribed texts and newspaper pieces are not, and a
    /// committee's text is named as one by its directory.
    #[test]
    fn the_materials_are_his_own_training_documents_and_nothing_else() {
        use Authorship::*;
        let documents = vec![
            document("d0", DraftInHand, Some("James Warren"), SCANNED),
            document("d1", SignedScribal, Some("Arthur Lee"), "A signed letter."),
            document(
                "d2",
                CommitteeCoauthored,
                None,
                "Gentlemen, you are chosen.",
            ),
            document("d3", PseudonymousAttributed, None, "A newspaper piece."),
            document(
                "d4",
                EditorAttributed,
                Some("Nobody"),
                "An ascribed letter.",
            ),
            document("d5", DraftInHand, Some("Held Out"), "An exam letter."),
            document("d6", DraftInHand, Some("Late"), "A late letter."),
        ];
        let assignments = vec![
            assigned("d0", Split::Train),
            assigned("d1", Split::Train),
            assigned("d2", Split::Train),
            assigned("d3", Split::Train),
            assigned("d4", Split::Train),
            assigned("d5", Split::Exam),
            assigned("d6", Split::Temporal),
        ];
        let files = materials(&documents, &assignments);
        let paths: Vec<&str> = files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(paths, ["own/d0.txt", "own/d1.txt", "committee/d2.txt"]);
        let all = files
            .iter()
            .map(|f| f.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        for absent in ["newspaper", "ascribed", "exam letter", "late letter"] {
            assert!(!all.contains(absent), "{absent} is in the materials");
        }
    }

    /// A file opens like a letter Splinter already learns from - who it was
    /// written to, when - and holds his words without the editors' footnotes,
    /// source notes and running heads.
    #[test]
    fn a_file_is_the_heading_the_date_and_his_words_without_apparatus() {
        let documents = vec![document(
            "d0",
            Authorship::DraftInHand,
            Some("James Warren"),
            SCANNED,
        )];
        let files = materials(&documents, &[assigned("d0", Split::Train)]);
        assert_eq!(
            files[0].text,
            "To James Warren\nNovember 2, 1772\n\nMy dear Sir,\n\nI have your favor of the tenth.\n\nYours, S. A.\n"
        );
    }

    #[test]
    fn a_document_that_is_not_a_letter_is_headed_by_its_printed_title() {
        let mut town = document(
            "d2",
            Authorship::CommitteeCoauthored,
            None,
            "Gentlemen, you are chosen.",
        );
        town.date = Date {
            year: 1764,
            month: Some(5),
            day: None,
        };
        let files = materials(&[town], &[assigned("d2", Split::Train)]);
        assert!(
            files[0].text.starts_with(
                "Instructions of the Town of Boston to Its Representatives\nMay 1764\n\n"
            ),
            "{}",
            files[0].text
        );
        assert_eq!(
            date_line(Date {
                year: 1770,
                month: None,
                day: None
            }),
            "1770"
        );
        let mut dated = document("d3", Authorship::CommitteeCoauthored, None, "Gentlemen.");
        dated.heading = "THE TOWN OF BOSTON TO THE TOWN OF PLYMOUTH. MARCH 24, 1766".into();
        dated.date = Date {
            year: 1766,
            month: Some(3),
            day: Some(24),
        };
        let files = materials(&[dated], &[assigned("d3", Split::Train)]);
        assert!(
            files[0]
                .text
                .starts_with("The Town of Boston to the Town of Plymouth\nMarch 24, 1766\n\n"),
            "{}",
            files[0].text
        );
        let mut wrapped = document("d4", Authorship::CommitteeCoauthored, None, "Voted.");
        wrapped.heading = "RESOLUTION OF THE TOWN OF BOSTON. SEPTEMBER".into();
        let files = materials(&[wrapped], &[assigned("d4", Split::Train)]);
        assert!(
            files[0]
                .text
                .starts_with("Resolution of the Town of Boston\n"),
            "{}",
            files[0].text
        );
    }
}
