// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The tests of the object admission: the synthetic corpus, supersession, and agreement with the consumer's implementation.

use super::*;
use serde_json::json;

const FILES: [(&str, &str); 14] = [
    (
        "client-behaviour-sheet",
        include_str!("../../tests/fixtures/material/corpus/client-behaviour-sheet.md"),
    ),
    (
        "counseling-protocol-2027",
        include_str!("../../tests/fixtures/material/corpus/counseling-protocol-2027.md"),
    ),
    (
        "counseling-protocol",
        include_str!("../../tests/fixtures/material/corpus/counseling-protocol.md"),
    ),
    (
        "forum-fans",
        include_str!("../../tests/fixtures/material/corpus/forum-fans.md"),
    ),
    (
        "forum-note",
        include_str!("../../tests/fixtures/material/corpus/forum-note.md"),
    ),
    (
        "forum-pumps",
        include_str!("../../tests/fixtures/material/corpus/forum-pumps.md"),
    ),
    (
        "payment-act",
        include_str!("../../tests/fixtures/material/corpus/payment-act.md"),
    ),
    (
        "service-manual-2024",
        include_str!("../../tests/fixtures/material/corpus/service-manual-2024.md"),
    ),
    (
        "service-manual-b-2024",
        include_str!("../../tests/fixtures/material/corpus/service-manual-b-2024.md"),
    ),
    (
        "service-manual-b",
        include_str!("../../tests/fixtures/material/corpus/service-manual-b.md"),
    ),
    (
        "service-manual",
        include_str!("../../tests/fixtures/material/corpus/service-manual.md"),
    ),
    (
        "supply-act-defects",
        include_str!("../../tests/fixtures/material/corpus/supply-act-defects.md"),
    ),
    (
        "supply-act-delay-amendment",
        include_str!("../../tests/fixtures/material/corpus/supply-act-delay-amendment.md"),
    ),
    (
        "supply-act-delay",
        include_str!("../../tests/fixtures/material/corpus/supply-act-delay.md"),
    ),
];
const PROPOSALS: &str = include_str!("../../tests/fixtures/material/proposals.json");
const STORED_PACK: &str = include_str!("../../tests/fixtures/material/synthetic-pack.json");

fn documents() -> Vec<Document> {
    FILES
        .iter()
        .map(|(id, raw)| Document::parse(id, raw))
        .collect()
}

fn proposals() -> Vec<ProposedObject> {
    let file: Value = serde_json::from_str(PROPOSALS).unwrap();
    serde_json::from_value(file["proposals"].clone()).unwrap()
}

fn one(id: &str) -> ProposedObject {
    proposals().into_iter().find(|p| p.id == id).unwrap()
}

#[test]
fn the_synthetic_corpus_is_admitted_exactly_as_the_consumers_admission_reads_it() {
    let pack = admit("synthetic", &documents(), &proposals());
    let stored: Value = serde_json::from_str(STORED_PACK).unwrap();
    // The pack is one format with two implementations of its admission; they must agree object for object.
    for (ours, theirs) in [
        ("objects", &pack.objects as &dyn erased::Ser),
        ("resolutions", &pack.resolutions),
        ("flags", &pack.flags),
        ("rejected", &pack.rejected),
        ("documents", &pack.documents),
    ] {
        assert_eq!(theirs.value(), stored[ours], "{ours}");
    }
    assert_eq!(pack.schema, PACK_SCHEMA);
}

mod erased {
    pub trait Ser {
        fn value(&self) -> serde_json::Value;
    }
    impl<T: serde::Serialize> Ser for T {
        fn value(&self) -> serde_json::Value {
            serde_json::to_value(self).unwrap()
        }
    }
}

#[test]
fn a_number_the_quote_does_not_carry_is_refused_however_plausible() {
    let pack = admit("t", &documents(), &[one("invented-number")]);
    assert!(
        pack.objects.is_empty() && pack.rejected[0].reason.contains("not in the quote"),
        "{:?}",
        pack.rejected
    );
}

#[test]
fn a_quote_that_is_not_in_the_document_and_unlisted_terms_are_refused() {
    assert!(admit("t", &documents(), &[one("misquoted")]).rejected[0]
        .reason
        .contains("not in the document"));
    let mut trait_ = one("trait-sleep");
    trait_.data["terms"] = json!(["penalty"]);
    assert!(admit("t", &documents(), &[trait_]).rejected[0]
        .reason
        .contains("terms"));
    let mut invented = one("trait-sleep");
    invented.data["reveals"] = json!("I have a secret second job as a pilot.");
    assert!(admit("t", &documents(), &[invented]).rejected[0]
        .reason
        .contains("not supported"));
}

#[test]
fn a_proposal_of_the_wrong_shape_or_with_sentences_for_names_is_refused() {
    let mut no_outcome = one("terminate-late");
    no_outcome.data.as_object_mut().unwrap().remove("outcome");
    assert!(admit("t", &documents(), &[no_outcome]).rejected[0]
        .reason
        .contains("missing"));
    let mut bad_operator = one("terminate-late");
    bad_operator.data["op"] = json!("~");
    assert!(admit("t", &documents(), &[bad_operator]).rejected[0]
        .reason
        .contains("bad condition"));
    let mut unknown = one("terminate-late");
    unknown.kind = "opinion".into();
    assert!(admit("t", &documents(), &[unknown]).rejected[0]
        .reason
        .contains("unknown kind"));
    let mut sentence = one("terminate-late");
    sentence.data["outcome"] = json!("the buyer may terminate");
    assert!(admit("t", &documents(), &[sentence]).rejected[0]
        .reason
        .contains("not identifiers"));
}

#[test]
fn spans_are_character_offsets_that_resolve_to_the_quote() {
    let pack = admit("synthetic", &documents(), &proposals());
    for object in &pack.objects {
        let text = &pack
            .documents
            .iter()
            .find(|d| d.id == object.document)
            .unwrap()
            .text;
        let found: String = text
            .chars()
            .skip(object.span[0])
            .take(object.span[1] - object.span[0])
            .collect();
        assert_eq!(found, object.quote);
    }
}

#[test]
fn headers_decide_where_documents_disagree_and_the_decision_is_recorded() {
    let pack = admit("synthetic", &documents(), &proposals());
    let by_winner = |w: &str| pack.resolutions.iter().find(|r| r.winner == w).unwrap();
    assert_eq!(
        by_winner("terminate-late-amended").superseded,
        ["terminate-late"]
    );
    assert_eq!(by_winner("reduction-amount").basis, "authority"); // a newer forum post of lower authority loses
    assert_eq!(
        by_winner("service-manual-prop-service-interval-alder-pump-3").basis,
        "date"
    );
    assert!(superseded_ids(&pack).contains("forum-reduction"));
}

#[test]
fn what_no_header_separates_stays_flagged_and_both_are_kept() {
    let pack = admit("synthetic", &documents(), &proposals());
    assert_eq!(pack.flags.len(), 1);
    assert_eq!(
        pack.flags[0].objects,
        ["payment-due", "payment-due-consumer"]
    );
    assert!(
        pack.objects.iter().any(|o| o.id == "payment-due")
            && pack.objects.iter().any(|o| o.id == "payment-due-consumer")
    );
}

#[test]
fn an_amendment_from_an_earlier_document_or_of_another_kind_is_refused() {
    let mut backwards = one("terminate-late-amended");
    backwards.document = "supply-act-delay".into();
    backwards.quote = one("terminate-late").quote;
    backwards.data["value"] = json!(14);
    let result = admit("t", &documents(), &[one("terminate-late"), backwards]);
    assert!(
        result.rejected[0]
            .reason
            .contains("later or more authoritative"),
        "{:?}",
        result.rejected
    );
    let mut unknown = one("terminate-late-amended");
    unknown.data["supersedes"] = json!("no-such-object");
    assert!(admit("t", &documents(), &[unknown]).rejected[0]
        .reason
        .contains("not an admitted object"));
}

#[test]
fn a_fact_whose_key_is_a_placeholder_or_not_in_the_quote_is_refused() {
    let fact = proposals()
        .into_iter()
        .find(|p| p.id.starts_with("service-manual-prop-service-interval"))
        .unwrap();
    for (key, reason) in [
        ("attribute", "attribute | entity"),
        ("attribute | entity", "not grounded"),
        ("rated flow | alder pump 3", "not grounded"),
    ] {
        let mut bad = fact.clone();
        bad.data["key"] = json!(key);
        assert!(
            admit("t", &documents(), &[bad]).rejected[0]
                .reason
                .contains(reason),
            "{key}"
        );
    }
}

#[test]
fn chapters_numbered_differently_never_compete_but_an_unnumbered_forum_post_competes_with_each() {
    let doc = |id: &str, title: &str, authority: u8| Document {
        id: id.into(),
        title: title.into(),
        text: format!(
            "More than {} days late ends it.",
            if id == "a" {
                14
            } else if id == "b" {
                21
            } else {
                30
            }
        ),
        provenance: Provenance {
            authority,
            version: 1,
            date: Some("2026-01-01".into()),
        },
    };
    let limit = |d: &str, days: u32| ProposedObject {
        id: format!("t-{d}"),
        kind: "threshold".into(),
        document: d.into(),
        quote: format!("More than {days} days late ends it."),
        data: json!({"var": "days_late", "op": ">", "value": days, "outcome": "ended", "vars": {"days_late": "int"}}),
    };
    let chapters = [
        doc("a", "Act, chapter 11", 3),
        doc("b", "Act, chapter 12", 3),
    ];
    let pack = admit("t", &chapters, &[limit("a", 14), limit("b", 21)]);
    assert!(pack.resolutions.is_empty() && pack.flags.is_empty());
    let all = [
        chapters[0].clone(),
        chapters[1].clone(),
        doc("c", "Forum note", 1),
    ];
    let pack = admit("t", &all, &[limit("a", 14), limit("b", 21), limit("c", 30)]);
    assert_eq!(
        pack.resolutions
            .iter()
            .map(|r| r.superseded[0].as_str())
            .collect::<Vec<_>>(),
        ["t-c", "t-c"]
    );
}
