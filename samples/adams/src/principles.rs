// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements source-grounded persona and expertise
// transfer for language models for its clients. If your team needs expertise
// in turning a historical record into hypotheses that each trace to the words
// that support them, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Principles: how Adams worked, proposed by a helper model and accepted only
//! as far as the documents bear it out.
//!
//! A principle is a hypothesis, never a fact. It states when it applies, what
//! he did, and what limits it, and every claim to support it names a document
//! and the words, copied exactly. Code looks each quotation up: one that is
//! not in the cited document is dropped and the reason kept, a quotation from
//! a document held out of training is refused, and a principle none of whose
//! quotations can be found is rejected. The status is computed from what was
//! found, not asserted by the model that proposed it.

use std::collections::{BTreeSet, HashSet};

use splinter_sdk::agent::schemars::JsonSchema;
use splinter_sdk::measure::verifiers::quotation::words;

use crate::curate::Document;

/// Passages a quotation must run to count as one.
pub const MIN_QUOTE_WORDS: usize = 6;

/// How far the evidence has carried a principle. The code sets the first three;
/// the rest are earned by tests that come later.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Status {
    /// Proposed, and no quotation of it was found.
    Proposed,
    /// At least one quotation is in a document he is answerable for.
    SourceSupported,
    /// Found in several documents, across periods or audiences.
    Recurring,
}

/// A claim that a document shows a principle: the document and the words.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, JsonSchema)]
#[schemars(crate = "splinter_sdk::agent::schemars")]
pub struct Support {
    /// The id of the document, as it was given.
    pub doc_id: String,
    /// The words, copied exactly from that document.
    pub quote: String,
}

/// What a helper proposes.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, JsonSchema)]
#[schemars(crate = "splinter_sdk::agent::schemars")]
pub struct Proposal {
    /// A rule that can be tested: when this holds, he did that.
    pub description: String,
    /// What must be true before the principle applies.
    pub trigger_conditions: Vec<String>,
    /// What he did when it applied.
    pub expected_behavior: Vec<String>,
    /// Limits and exceptions: when it does not apply.
    pub qualifications: Vec<String>,
    /// Passages that show it.
    pub support: Vec<Support>,
}

/// A quotation found in its document, with where it sits in his career.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Found {
    pub doc_id: String,
    pub quote: String,
    pub period: crate::document::Period,
    pub recipient: Option<String>,
}

/// A principle as far as the documents bear it out.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Principle {
    pub id: String,
    pub description: String,
    pub trigger_conditions: Vec<String>,
    pub expected_behavior: Vec<String>,
    pub qualifications: Vec<String>,
    pub status: Status,
    pub support: Vec<Found>,
    /// Quotations that were dropped, each with why.
    pub dropped: Vec<String>,
}

/// Why a proposal was not accepted at all.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Rejection {
    pub reason: String,
    pub dropped: Vec<String>,
}

/// Documents, periods or audiences a principle must be found across to count
/// as recurring: one memorable sentence is not a habit.
const RECURRING_DOCUMENTS: usize = 3;
const RECURRING_SPREAD: usize = 2;

/// The first words of a quotation, for a reason a person can read.
fn gist(quote: &str) -> String {
    quote
        .split_whitespace()
        .take(8)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Whether `needle` occurs in `haystack` as consecutive words.
pub(crate) fn occurs(haystack: &[String], needle: &[String]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|w| w == needle)
}

/// Check a proposal against the documents it cites. `allowed` is the set of
/// document ids a principle may draw on: those the model may learn from.
pub fn verify(
    proposal: &Proposal,
    docs: &[Document],
    allowed: &HashSet<String>,
) -> Result<Principle, Rejection> {
    let reject = |reason: &str, dropped: Vec<String>| Rejection {
        reason: reason.to_string(),
        dropped,
    };
    if proposal.description.trim().is_empty() {
        return Err(reject("no description", Vec::new()));
    }
    if proposal.trigger_conditions.is_empty() {
        return Err(reject(
            "no trigger conditions, so nothing says when it applies",
            Vec::new(),
        ));
    }
    if proposal.expected_behavior.is_empty() {
        return Err(reject(
            "no expected behavior, so nothing says what it predicts",
            Vec::new(),
        ));
    }

    let mut support: Vec<Found> = Vec::new();
    let mut dropped = Vec::new();
    let mut counted: HashSet<(String, String)> = HashSet::new();
    for claim in &proposal.support {
        let quote = words(&claim.quote);
        let id = claim.doc_id.as_str();
        if quote.len() < MIN_QUOTE_WORDS {
            dropped.push(format!(
                "quotation from {id} is too short ({} words) to be a claim: {}",
                quote.len(),
                gist(&claim.quote)
            ));
            continue;
        }
        let Some(doc) = docs.iter().find(|d| d.id == id) else {
            dropped.push(format!("no document {id}"));
            continue;
        };
        if !allowed.contains(id) {
            dropped.push(format!("document {id} is held out of training"));
            continue;
        }
        if !doc.authorship.supports_principles() {
            dropped.push(format!("document {id} is not one he is answerable for"));
            continue;
        }
        if !occurs(&words(&doc.body), &quote) {
            dropped.push(format!(
                "quotation is not in document {id}: {}",
                gist(&claim.quote)
            ));
            continue;
        }
        if counted.insert((id.to_string(), quote.join(" "))) {
            support.push(Found {
                doc_id: id.to_string(),
                quote: claim.quote.trim().to_string(),
                period: doc.period,
                recipient: doc.recipient.clone(),
            });
        }
    }
    if support.is_empty() {
        return Err(reject(
            "no quotation could be found in an allowed document",
            dropped,
        ));
    }

    let documents: BTreeSet<&str> = support.iter().map(|f| f.doc_id.as_str()).collect();
    let periods: BTreeSet<_> = support.iter().map(|f| f.period).collect();
    let audiences: BTreeSet<_> = support
        .iter()
        .filter_map(|f| f.recipient.as_deref())
        .collect();
    let spread = periods.len().max(audiences.len());
    let status = if documents.len() >= RECURRING_DOCUMENTS && spread >= RECURRING_SPREAD {
        Status::Recurring
    } else {
        Status::SourceSupported
    };
    let id = format!(
        "P-{}",
        &blake3::hash(words(&proposal.description).join(" ").as_bytes()).to_hex()[..12]
    );
    Ok(Principle {
        id,
        description: proposal.description.trim().to_string(),
        trigger_conditions: proposal.trigger_conditions.clone(),
        expected_behavior: proposal.expected_behavior.clone(),
        qualifications: proposal.qualifications.clone(),
        status,
        support,
        dropped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Authorship, Date, Period};

    fn doc(id: &str, year: u16, recipient: &str, authorship: Authorship, body: &str) -> Document {
        Document {
            schema: crate::curate::SCHEMA,
            id: id.into(),
            source_id: "cushing-1".into(),
            heading: format!("TO {}.", recipient.to_uppercase()),
            recipient: Some(recipient.into()),
            note: "[MS.]".into(),
            date: Date {
                year,
                month: None,
                day: None,
            },
            period: Period::of(year),
            authorship,
            authorship_confidence: 0.8,
            authorship_basis: "test".into(),
            temporal_holdout: false,
            body: body.into(),
        }
    }

    const FIRST: &str = "We are not to wait for the Ministry to grant what is ours by right, but to state it plainly to every Town in the Province.";
    const SECOND: &str = "Let the Committee write to every Town, that the Sense of the People may be known before the Assembly meets again.";
    const THIRD: &str = "I wish you would collect the Opinion of the neighbouring Towns, and send me their Resolutions as soon as they are passed.";

    fn docs() -> Vec<Document> {
        vec![
            doc("d1", 1768, "James Otis", Authorship::DraftInHand, FIRST),
            doc("d2", 1773, "Arthur Lee", Authorship::DraftInHand, SECOND),
            doc("d3", 1778, "James Warren", Authorship::SignedScribal, THIRD),
            doc(
                "held",
                1770,
                "John Smith",
                Authorship::DraftInHand,
                "A held out document with words nobody trained on at all.",
            ),
        ]
    }

    fn allowed() -> HashSet<String> {
        ["d1", "d2", "d3"].iter().map(|s| s.to_string()).collect()
    }

    fn proposal(support: Vec<(&str, &str)>) -> Proposal {
        Proposal {
            description:
                "When the ministry will not act, he had the towns state the claim together.".into(),
            trigger_conditions: vec!["a grievance the ministry has not answered".into()],
            expected_behavior: vec!["write to the towns and collect their resolutions".into()],
            qualifications: vec!["not when the towns are divided".into()],
            support: support
                .into_iter()
                .map(|(d, q)| Support {
                    doc_id: d.into(),
                    quote: q.into(),
                })
                .collect(),
        }
    }

    #[test]
    fn a_quotation_found_in_its_document_supports_the_principle() {
        let p = verify(
            &proposal(vec![(
                "d1",
                "state it plainly to every Town in the Province",
            )]),
            &docs(),
            &allowed(),
        )
        .unwrap();
        assert_eq!(p.status, Status::SourceSupported);
        assert_eq!(p.support.len(), 1);
        assert_eq!(
            (p.support[0].doc_id.as_str(), p.support[0].period),
            ("d1", Period::PreRevolution)
        );
        assert!(p.dropped.is_empty());
    }

    #[test]
    fn a_quotation_is_matched_whatever_its_case_punctuation_or_line_breaks() {
        let q = "state it PLAINLY,\nto every Town in the Province.";
        assert!(verify(&proposal(vec![("d1", q)]), &docs(), &allowed()).is_ok());
    }

    #[test]
    fn a_quotation_that_is_not_in_the_document_is_dropped_with_its_reason_and_the_rest_stand() {
        let p = verify(
            &proposal(vec![
                ("d1", "state it plainly to every Town in the Province"),
                ("d2", "an invented line he never wrote at all"),
            ]),
            &docs(),
            &allowed(),
        )
        .unwrap();
        assert_eq!(p.support.len(), 1);
        assert_eq!(p.dropped.len(), 1);
        assert!(
            p.dropped[0].contains("not in") && p.dropped[0].contains("d2"),
            "{:?}",
            p.dropped
        );
    }

    #[test]
    fn a_single_altered_word_means_the_words_are_not_his() {
        assert!(verify(
            &proposal(vec![(
                "d1",
                "state it plainly to every City in the Province"
            )]),
            &docs(),
            &allowed()
        )
        .is_err());
    }

    #[test]
    fn a_principle_none_of_whose_quotations_can_be_found_is_rejected() {
        let r = verify(
            &proposal(vec![("d1", "nothing like this appears anywhere in it")]),
            &docs(),
            &allowed(),
        )
        .unwrap_err();
        assert!(r.reason.contains("no quotation"), "{}", r.reason);
        assert_eq!(r.dropped.len(), 1);
    }

    #[test]
    fn a_quotation_from_a_document_held_out_of_training_is_refused() {
        let r = verify(
            &proposal(vec![("held", "words nobody trained on at all")]),
            &docs(),
            &allowed(),
        )
        .unwrap_err();
        assert!(r.dropped[0].contains("held out"), "{:?}", r.dropped);
    }

    #[test]
    fn a_document_that_does_not_exist_is_a_dropped_claim_not_a_crash() {
        let r = verify(
            &proposal(vec![(
                "nope",
                "state it plainly to every Town in the Province",
            )]),
            &docs(),
            &allowed(),
        )
        .unwrap_err();
        assert!(r.dropped[0].contains("no document"), "{:?}", r.dropped);
    }

    #[test]
    fn a_quotation_too_short_to_be_a_claim_is_dropped() {
        let r = verify(&proposal(vec![("d1", "every Town")]), &docs(), &allowed()).unwrap_err();
        assert!(r.dropped[0].contains("short"), "{:?}", r.dropped);
    }

    #[test]
    fn found_in_several_documents_across_periods_it_is_recurring() {
        let support = vec![
            ("d1", "state it plainly to every Town in the Province"),
            ("d2", "Let the Committee write to every Town"),
            ("d3", "collect the Opinion of the neighbouring Towns"),
        ];
        let p = verify(&proposal(support), &docs(), &allowed()).unwrap();
        assert_eq!(p.status, Status::Recurring);
        let periods: BTreeSet<_> = p.support.iter().map(|f| f.period).collect();
        assert!(periods.len() >= 2);
    }

    #[test]
    fn the_same_quotation_cited_twice_counts_once() {
        let q = "state it plainly to every Town in the Province";
        let p = verify(
            &proposal(vec![("d1", q), ("d1", q), ("d1", q)]),
            &docs(),
            &allowed(),
        )
        .unwrap();
        assert_eq!(p.support.len(), 1);
        assert_eq!(
            p.status,
            Status::SourceSupported,
            "one document is not recurrence"
        );
    }

    #[test]
    fn a_proposal_with_nothing_to_test_is_rejected() {
        let mut p = proposal(vec![(
            "d1",
            "state it plainly to every Town in the Province",
        )]);
        p.trigger_conditions.clear();
        assert!(verify(&p, &docs(), &allowed())
            .unwrap_err()
            .reason
            .contains("trigger"));
        let mut p = proposal(vec![(
            "d1",
            "state it plainly to every Town in the Province",
        )]);
        p.expected_behavior.clear();
        assert!(verify(&p, &docs(), &allowed())
            .unwrap_err()
            .reason
            .contains("behavior"));
    }

    #[test]
    fn the_id_follows_the_description_so_a_restatement_is_the_same_principle() {
        let a = verify(
            &proposal(vec![(
                "d1",
                "state it plainly to every Town in the Province",
            )]),
            &docs(),
            &allowed(),
        )
        .unwrap();
        let b = verify(
            &proposal(vec![("d2", "Let the Committee write to every Town")]),
            &docs(),
            &allowed(),
        )
        .unwrap();
        assert_eq!(a.id, b.id);
        assert!(a.id.starts_with("P-"));
    }
}
