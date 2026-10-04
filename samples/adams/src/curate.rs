// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements provenance-tracked training corpora for
// language models for its clients. If your team needs expertise in curating
// a historical record so that every document says how sure we are of its
// author, you can procure our services by sending an email to
// info@swedishembedded.com.

//! A parsed entry becomes a curated document: who wrote it and how sure we
//! are, when and in which period, and whether it is withheld from training.

use crate::attribution::attribute;
use crate::corpus::Entry;
use crate::document::{self, Authorship, Date, Identity, Period};

/// The corpus's own record of a document. `schema` changes when a field's
/// meaning does, so an old file is never read as a new one.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Document {
    pub schema: u32,
    pub id: String,
    /// The fetched text it was parsed from: `cushing-1`.
    pub source_id: String,
    pub heading: String,
    pub recipient: Option<String>,
    pub note: String,
    pub date: Date,
    pub period: Period,
    pub authorship: Authorship,
    pub authorship_confidence: f32,
    pub authorship_basis: String,
    /// Withheld from training: the late years the temporal test is made on.
    pub temporal_holdout: bool,
    pub body: String,
}

pub const SCHEMA: u32 = 1;

/// A document the namesake or date rules refused, and why.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct Excluded {
    pub id: String,
    pub heading: String,
    pub year: u16,
    pub reason: String,
}

#[derive(Debug, Default)]
pub struct Curated {
    pub documents: Vec<Document>,
    pub excluded: Vec<Excluded>,
    /// Sent to a person, not decided: a "junior" before the father's death.
    pub for_review: Vec<Excluded>,
}

/// Curate the entries of one source, whose author line is `author`.
pub fn curate(source_id: &str, author: &str, entries: &[Entry]) -> Curated {
    let mut curated = Curated::default();
    for entry in entries {
        let record = |reason: &str| Excluded {
            id: entry.id.clone(),
            heading: entry.heading.clone(),
            year: entry.date.year,
            reason: reason.to_string(),
        };
        match document::identity(author, entry.date) {
            Identity::Excluded(why) => curated.excluded.push(record(why)),
            Identity::NeedsReview(why) => curated.for_review.push(record(why)),
            Identity::Target => {
                let verdict = attribute(&entry.heading, &entry.note);
                curated.documents.push(Document {
                    schema: SCHEMA,
                    id: entry.id.clone(),
                    source_id: source_id.to_string(),
                    heading: entry.heading.clone(),
                    recipient: entry.recipient.clone(),
                    note: entry.note.clone(),
                    date: entry.date,
                    period: Period::of(entry.date.year),
                    authorship: verdict.authorship,
                    authorship_confidence: verdict.confidence,
                    authorship_basis: verdict.basis.to_string(),
                    temporal_holdout: document::in_temporal_holdout(entry.date.year),
                    body: entry.body.clone(),
                });
            }
        }
    }
    curated
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(heading: &str, note: &str, year: u16) -> Entry {
        Entry {
            id: "cushing-1-0".into(),
            edition: "cushing-1".into(),
            heading: heading.into(),
            recipient: heading
                .strip_prefix("TO ")
                .map(|n| n.trim_end_matches('.').to_string()),
            note: note.into(),
            date: Date {
                year,
                month: Some(4),
                day: Some(2),
            },
            body: "text".into(),
        }
    }

    #[test]
    fn a_document_carries_its_attribution_period_and_provenance() {
        let note = "[MS., Samuel Adams Papers, Lenox Library.]";
        let curated = curate(
            "cushing-1",
            "Samuel Adams",
            &[entry("TO JOHN DICKINSON.", note, 1768)],
        );
        let d = &curated.documents[0];
        assert_eq!(d.schema, SCHEMA);
        assert_eq!(d.source_id, "cushing-1");
        assert_eq!(d.authorship, Authorship::DraftInHand);
        assert!(d.authorship_confidence > 0.0 && !d.authorship_basis.is_empty());
        assert_eq!(d.period, Period::PreRevolution);
        assert!(!d.temporal_holdout);
        assert_eq!(d.note, note);
    }

    #[test]
    fn the_late_years_are_marked_withheld() {
        let curated = curate(
            "cushing-4",
            "Samuel Adams",
            &[entry("TO JOHN ADAMS.", "[MS., Samuel Adams Papers.]", 1795)],
        );
        assert!(curated.documents[0].temporal_holdout);
    }

    #[test]
    fn a_document_outside_his_working_years_is_excluded_with_its_reason() {
        let curated = curate(
            "cushing-1",
            "Samuel Adams",
            &[entry("TO NOBODY.", "[MS., Elsewhere.]", 1730)],
        );
        assert!(curated.documents.is_empty());
        assert_eq!(curated.excluded.len(), 1);
        assert!(
            curated.excluded[0].reason.contains("outside"),
            "{}",
            curated.excluded[0].reason
        );
    }

    #[test]
    fn a_namesake_author_line_excludes_every_document_of_the_source() {
        let curated = curate(
            "other-1",
            "Samuel Adams, physician",
            &[entry("TO A.", "[MS., X.]", 1775)],
        );
        assert!(curated.documents.is_empty() && curated.excluded.len() == 1);
    }

    #[test]
    fn a_document_the_rules_cannot_settle_goes_for_review_not_into_the_corpus() {
        let curated = curate(
            "cushing-1",
            "Samuel Adams, Jr.",
            &[entry("TO A.", "[MS., X.]", 1745)],
        );
        assert!(curated.documents.is_empty() && curated.excluded.is_empty());
        assert_eq!(curated.for_review.len(), 1);
    }

    #[test]
    fn identities_are_decided_per_document_so_no_document_is_lost_to_a_neighbour() {
        let entries = [
            entry("TO A.", "[MS., X.]", 1730),
            entry("TO B.", "[MS., X.]", 1770),
        ];
        let curated = curate("cushing-1", "Samuel Adams", &entries);
        assert_eq!((curated.documents.len(), curated.excluded.len()), (1, 1));
    }

    #[test]
    fn a_document_survives_a_round_trip_through_json() {
        let curated = curate(
            "cushing-1",
            "Samuel Adams",
            &[entry("TO A.", "[Boston Gazette, May 1, 1770.]", 1770)],
        );
        let line = serde_json::to_string(&curated.documents[0]).unwrap();
        let back: Document = serde_json::from_str(&line).unwrap();
        assert_eq!(back, curated.documents[0]);
        assert!(line.contains("PSEUDONYMOUS_ATTRIBUTED"), "{line}");
    }
}
