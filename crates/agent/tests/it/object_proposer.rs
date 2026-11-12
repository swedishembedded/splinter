// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The object proposer on a scripted model: a reply with a refused proposal is sent back with the reasons, the vocabulary step puts free
//! text into identifiers, and a pack comes out of the documents.

use serde_json::{json, Value};
use splinter_agent::object_proposer::SvenObjectProposer;
use splinter_agent::solve::Model;
use splinter_knowledge::objects::propose::{propose_pack, Strategy, Vocabulary};
use splinter_knowledge::objects::Document;

use crate::common::Scripted;

const DELAY: &str = "---\nauthority: 3\nversion: 1\ndate: 2026-03-01\n---\n# Supply Act\n\nSection 3.1. If the seller delivers more than 14 days after the agreed date, the buyer may terminate the contract by written notice.\n";
const AMENDMENT: &str = "---\nauthority: 3\nversion: 2\ndate: 2027-01-01\n---\n# Supply Act amendment\n\nSection 3.1 (amended). If the seller delivers more than 21 days after the agreed date, the buyer may terminate the contract by written notice.\n";
const QUOTE_14: &str = "If the seller delivers more than 14 days after the agreed date, the buyer may terminate the contract by written notice.";
const QUOTE_21: &str = "If the seller delivers more than 21 days after the agreed date, the buyer may terminate the contract by written notice.";

fn threshold(id: &str, quote: &str, value: Value, outcome: &str, extra: Value) -> Value {
    let mut object = json!({"id": id, "kind": "threshold", "document": "ignored", "quote": quote,
        "data": {"var": "Days Late", "op": ">", "value": value, "outcome": outcome, "label": "days of delay", "vars": {"Days Late": "int"}}});
    if let Some(map) = extra.as_object() {
        map.iter().for_each(|(k, v)| object["data"][k] = v.clone());
    }
    object
}

fn reply(objects: Vec<Value>) -> String {
    json!({"objects": objects}).to_string()
}

fn proposer(script: Vec<String>) -> (SvenObjectProposer, std::sync::Arc<Scripted>) {
    let model = Scripted::new(script);
    (
        SvenObjectProposer::new(Model::new(model.clone(), "scripted/extractor-1")).with_repairs(2),
        model,
    )
}

#[tokio::test]
async fn a_reply_with_an_invented_number_is_sent_back_with_the_reason_and_the_corrected_one_is_admitted(
) {
    let documents = vec![Document::parse("supply-act-delay", DELAY)];
    let wrong = reply(vec![threshold(
        "late",
        QUOTE_14,
        json!(21),
        "May Terminate",
        json!({}),
    )]);
    let right = reply(vec![threshold(
        "late",
        QUOTE_14,
        json!(14),
        "May Terminate",
        json!({}),
    )]);
    let (proposer, model) = proposer(vec![wrong, right]);
    let pack = propose_pack(
        "t",
        &documents,
        &proposer,
        &Vocabulary::default(),
        Strategy::default(),
    )
    .await
    .unwrap()
    .pack;
    assert_eq!(model.seen.lock().unwrap().len(), 2, "one correction round");
    let correction = format!("{:?}", model.seen.lock().unwrap()[1].messages);
    assert!(
        correction.contains("not in the quote"),
        "the model was told why: {correction}"
    );
    assert!(pack.rejected.is_empty() && pack.objects.len() == 1);
    let data = &pack.objects[0].data;
    assert_eq!(
        (
            data["var"].as_str(),
            data["outcome"].as_str(),
            data["vars"]["days_late"].as_str()
        ),
        (Some("days_late"), Some("may_terminate"), Some("int"))
    );
    assert_eq!(
        pack.objects[0].id, "supply-act-delay-late",
        "ids are namespaced by document"
    );
}

#[tokio::test]
async fn a_declared_vocabulary_refuses_other_outcomes_until_the_model_uses_it() {
    let documents = vec![Document::parse("supply-act-delay", DELAY)];
    let off = reply(vec![threshold(
        "late",
        QUOTE_14,
        json!(14),
        "buyer may cancel",
        json!({}),
    )]);
    let on = reply(vec![threshold(
        "late",
        QUOTE_14,
        json!(14),
        "may_terminate",
        json!({}),
    )]);
    let (proposer, model) = proposer(vec![off, on]);
    let vocabulary = Vocabulary {
        outcomes: vec!["may_terminate".into(), "no_right".into()],
    };
    let pack = propose_pack("t", &documents, &proposer, &vocabulary, Strategy::default())
        .await
        .unwrap()
        .pack;
    assert!(format!("{:?}", model.seen.lock().unwrap()[1].messages).contains("declared vocabulary"));
    assert_eq!(pack.objects[0].data["outcome"], "may_terminate");
}

#[tokio::test]
async fn what_is_still_refused_after_the_last_repair_is_dropped_and_the_good_proposals_stay() {
    let documents = vec![Document::parse("supply-act-delay", DELAY)];
    let mixed = reply(vec![
        threshold("good", QUOTE_14, json!(14), "may_terminate", json!({})),
        threshold("bad", QUOTE_14, json!(99), "may_terminate", json!({})),
    ]);
    let (proposer, model) = proposer(vec![mixed.clone(), mixed.clone(), mixed]);
    let pack = propose_pack(
        "t",
        &documents,
        &proposer,
        &Vocabulary::default(),
        Strategy::default(),
    )
    .await
    .unwrap()
    .pack;
    assert_eq!(
        model.seen.lock().unwrap().len(),
        3,
        "the original and two repairs, no more"
    );
    assert_eq!(pack.objects.len(), 1);
    assert_eq!(pack.rejected[0].id, "supply-act-delay-bad");
}

#[tokio::test]
async fn an_amendment_names_the_earlier_object_it_replaces_and_documents_are_taken_in_date_order() {
    let documents = vec![
        Document::parse("amendment", AMENDMENT),
        Document::parse("supply-act-delay", DELAY),
    ];
    let first = reply(vec![threshold(
        "late",
        QUOTE_14,
        json!(14),
        "may_terminate",
        json!({}),
    )]);
    let second = reply(vec![threshold(
        "late",
        QUOTE_21,
        json!(21),
        "may_terminate",
        json!({"supersedes": "supply-act-delay-late"}),
    )]);
    let (proposer, model) = proposer(vec![first, second]);
    let pack = propose_pack(
        "t",
        &documents,
        &proposer,
        &Vocabulary::default(),
        Strategy::default(),
    )
    .await
    .unwrap()
    .pack;
    assert!(
        format!("{:?}", model.seen.lock().unwrap()[1].messages).contains("supply-act-delay-late"),
        "the second document is shown the first one's objects"
    );
    assert_eq!(pack.resolutions.len(), 1);
    assert_eq!(
        (
            pack.resolutions[0].winner.as_str(),
            pack.resolutions[0].superseded.as_slice()
        ),
        (
            "amendment-late",
            ["supply-act-delay-late".to_string()].as_slice()
        )
    );
}

#[tokio::test]
async fn one_pass_per_family_of_kinds_puts_a_brief_about_those_kinds_and_drops_the_rest() {
    let documents = vec![Document::parse("supply-act-delay", DELAY)];
    let strategy = Strategy {
        per_kind: true,
        examples: true,
        chunk_words: None,
    };
    let on_topic = reply(vec![threshold(
        "late",
        QUOTE_14,
        json!(14),
        "may_terminate",
        json!({}),
    )]);
    let nothing = reply(vec![]);
    // Five passes: only the first (conditions) answers with a threshold; the second answers with a threshold too, which is not its kind.
    let off_topic = on_topic.clone();
    let (proposer, model) = proposer(vec![
        on_topic,
        off_topic,
        nothing.clone(),
        nothing.clone(),
        nothing,
    ]);
    let pack = propose_pack("t", &documents, &proposer, &Vocabulary::default(), strategy)
        .await
        .unwrap()
        .pack;
    let seen = model.seen.lock().unwrap();
    assert_eq!(seen.len(), 5);
    let system = |i: usize| format!("{:?}", seen[i].messages);
    assert!(system(0).contains("limit, states a condition") && system(0).contains("Example text"));
    assert!(system(1).contains("amount, percentage or formula"));
    assert_eq!(
        pack.objects.len(),
        1,
        "a threshold answered in the formula pass is not that pass's kind"
    );
}

#[tokio::test]
async fn a_document_is_put_in_runs_of_paragraphs_when_asked() {
    let text = "---\nauthority: 3\n---\n# T\n\nFirst paragraph with several words in it.\n\nSecond paragraph with several more words.\n";
    let documents = vec![Document::parse("d", text)];
    let strategy = Strategy {
        per_kind: false,
        examples: false,
        chunk_words: Some(8),
    };
    let (proposer, model) = proposer(vec![reply(vec![]), reply(vec![]), reply(vec![])]);
    let proposed = propose_pack("t", &documents, &proposer, &Vocabulary::default(), strategy)
        .await
        .unwrap();
    assert!(proposed.declined.is_empty());
    assert_eq!(
        model.seen.lock().unwrap().len(),
        3,
        "a title and two paragraphs, none of them fits with another in eight words"
    );
}

#[tokio::test]
async fn a_pass_that_never_gets_a_usable_reply_is_reported_and_does_not_stop_the_run() {
    let documents = vec![Document::parse("supply-act-delay", DELAY)];
    let prose: Vec<String> = (0..4)
        .map(|n| format!("no objects here, attempt {n}"))
        .collect();
    let (proposer, _) = proposer(prose);
    let proposed = propose_pack(
        "t",
        &documents,
        &proposer,
        &Vocabulary::default(),
        Strategy::default(),
    )
    .await
    .unwrap();
    assert_eq!(proposed.declined.len(), 1, "{:?}", proposed.declined);
    assert!(
        proposed.declined[0].contains("supply-act-delay/all")
            && proposed.declined[0].contains("no usable reply")
    );
    assert!(proposed.pack.objects.is_empty());
}
