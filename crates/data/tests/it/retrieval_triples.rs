// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-data pipelines held to a second,
// independent implementation, for its clients. If your team needs expertise
// in validating contrastive data for retrieval models, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec, against an independent reference (plan 6B): for each frozen case,
//! the retrieval view writes exactly the (query, positive, negatives) records
//! the reference (`validation/oracles/data/retrieval_triples.py`, written from
//! the view's documentation) wrote, in its order, and leaves out exactly what
//! it left out, for the reason it gave: spans over content the store lacks,
//! ranges that cut a character, failed, undecided and too weak experiences,
//! and passes with no evidence.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use crate::common;

use common::{answered, task_with_evidence, verdict, Scratch};
use serde_json::{json, Map, Value};
use splinter_conformance::{compare, fixture, Tolerance};
use splinter_core::annotation::{Outcome, Strength};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Digest, Span};
use splinter_core::source::{CapturedSource, Origin, PartContent, PartRef};
use splinter_data::{Corpus, Projection, RecordBody, Retrieval, View};
use splinter_store::sources::SourceStore;
use splinter_store::StateRoot;

fn strength(name: &str) -> Strength {
    if name == "judged" {
        Strength::Judged
    } else {
        Strength::Executable
    }
}

/// The case's spans, each over its content: through the captured part when
/// the store holds it, by the content's digest alone when it does not.
fn spans(sources: &SourceStore, case: &Value) -> Vec<Box<dyn Fn(u64, u64) -> Span>> {
    case["contents"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(n, content)| {
            let text = content["text"].as_str().unwrap().to_string();
            let digest = Digest::of(text.as_bytes());
            let part = content["stored"].as_bool().unwrap().then(|| {
                let captured = CapturedSource::new(
                    Origin::Document {
                        path: format!("/doc-{n}.txt"),
                    },
                    vec![PartContent {
                        name: "doc.txt".into(),
                        media_type: "text/plain".into(),
                        bytes: text.into_bytes(),
                    }],
                    &FixedClock::new("2026-09-30T00:00:00.000Z"),
                )
                .unwrap();
                PartRef {
                    source: sources.put_source(&captured).unwrap(),
                    name: "doc.txt".into(),
                }
            });
            Box::new(move |start, end| match &part {
                Some(part) => Span::in_part(part.clone(), digest.clone(), start, end).unwrap(),
                None => Span::new(digest.clone(), start, end).unwrap(),
            }) as Box<dyn Fn(u64, u64) -> Span>
        })
        .collect()
}

fn corpus(sources: &SourceStore, case: &Value) -> Corpus {
    let over = spans(sources, case);
    let mut corpus = Corpus::new();
    for (n, e) in case["experiences"].as_array().unwrap().iter().enumerate() {
        let evidence = e["evidence"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| {
                let at = |key: &str| s[key].as_u64().unwrap();
                over[at("content") as usize](at("start"), at("end"))
            })
            .collect();
        let task = task_with_evidence(
            "recall",
            e["instruction"].as_str().unwrap(),
            Vec::new(),
            evidence,
        );
        let exp = answered(&task, "an answer", &format!("2026-09-30T01:00:{n:02}.000Z"));
        let strength = strength(e["strength"].as_str().unwrap());
        let notes = match e["verdict"].as_str().unwrap() {
            "pass" => vec![verdict(&exp, Outcome::Pass, strength)],
            "fail" => vec![verdict(&exp, Outcome::Fail, strength)],
            _ => Vec::new(),
        };
        corpus.insert(exp, notes).unwrap();
    }
    corpus
}

fn excluded(projection: &Projection) -> Value {
    let out: Map<String, Value> = projection
        .excluded
        .iter()
        .map(|(reason, n)| (format!("{reason:?}"), json!(n)))
        .collect();
    Value::Object(out)
}

fn observed(case: &Value, index: usize) -> Value {
    let scratch = Scratch::new(&format!("retrieval-triples-{index}"));
    let sources = SourceStore::new(&splinter_store::workspace::Workspace::at(&StateRoot::new(
        &scratch.0,
    )));
    let corpus = corpus(&sources, case);
    let minimum = strength(case["minimum"].as_str().unwrap());
    let projection = Retrieval::new(minimum, &sources).project(&corpus).unwrap();
    let records: Vec<Value> = projection
        .records
        .iter()
        .map(|r| match &r.body {
            RecordBody::Contrastive {
                query,
                positive,
                negatives,
            } => json!({"query": query, "positive": positive, "negatives": negatives}),
            other => panic!("a contrastive record, got {other:?}"),
        })
        .collect();
    let mut out = case.clone();
    out["view"] = json!({"records": records, "excluded": excluded(&projection)});
    out
}

#[test]
fn the_view_writes_and_leaves_out_what_the_reference_does() {
    let fixture = fixture(env!("CARGO_MANIFEST_DIR"), "data/retrieval-triples").unwrap();
    let mut wrong = Vec::new();
    for (index, case) in fixture.cases().iter().enumerate() {
        if let Err(mismatch) = compare(&observed(case, index), case, Tolerance::new(0.0, 0.0)) {
            wrong.push(format!("{mismatch}\n  case: {case}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} cases disagree with the reference; the first:\n{}",
        wrong.len(),
        fixture.cases().len(),
        wrong[..wrong.len().min(3)].join("\n")
    );
}
