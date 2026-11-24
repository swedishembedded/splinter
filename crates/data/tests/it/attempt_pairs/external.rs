// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-data pipelines held to a second,
// independent implementation, for its clients. If your team needs expertise
// in validating preference and feedback data, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec, against an independent reference (plan 6B): for each frozen case,
//! the preference view pairs exactly the attempts the reference
//! (`validation/oracles/data/attempt_pairs.py`, written from the views'
//! documentation) paired, and the feedback view keeps exactly the attempts it
//! kept, each desirable as it said, with its class weights, both leaving out
//! exactly what it left out, for the reason it gave.

use serde_json::{json, Map, Value};
use splinter_conformance::{compare, fixture, Tolerance};
use splinter_core::annotation::Strength;
use splinter_data::{Corpus, Feedback, Preference, Projection, RecordBody, View};

use super::{add, Attempt};

fn strength(name: &str) -> Strength {
    if name == "judged" {
        Strength::Judged
    } else {
        Strength::Executable
    }
}

fn corpus(case: &Value) -> Corpus {
    let mut corpus = Corpus::new();
    for a in case["attempts"].as_array().unwrap() {
        let text = |key: &str| a[key].as_str().unwrap();
        let attempt = match text("verdict") {
            "pass" => Attempt::pass(text("trial")),
            "fail" => Attempt::fail(text("trial")),
            _ => Attempt::infrastructure(text("trial")),
        }
        .model(text("model"));
        let attempt = Attempt {
            strength: strength(text("strength")),
            ..attempt
        };
        add(&mut corpus, &attempt);
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

fn observed(case: &Value) -> Value {
    let minimum = strength(case["minimum"].as_str().unwrap());
    let corpus = corpus(case);
    let pairing = Preference::new(minimum).project(&corpus).unwrap();
    let mut pairs: Vec<Value> = pairing
        .records
        .iter()
        .map(|r| match &r.body {
            RecordBody::Preference { identity, .. } => {
                json!(identity.as_ref().unwrap().rollouts)
            }
            other => panic!("a preference record, got {other:?}"),
        })
        .collect();
    pairs.sort_by_key(ToString::to_string);
    let judging = Feedback::new(minimum).project(&corpus).unwrap();
    let records: Vec<Value> = judging
        .records
        .iter()
        .map(|r| match &r.body {
            RecordBody::Feedback {
                desirable,
                identity,
                ..
            } => json!({"trial": identity.rollouts[0], "desirable": desirable}),
            other => panic!("a feedback record, got {other:?}"),
        })
        .collect();
    let balance = judging.feedback.unwrap();
    let mut out = case.clone();
    out["preference"] = json!({"pairs": pairs, "excluded": excluded(&pairing)});
    out["feedback"] = json!({
        "records": records,
        "excluded": excluded(&judging),
        "balance": {
            "desirable": balance.desirable,
            "undesirable": balance.undesirable,
            "desirable_weight": balance.desirable_weight,
            "undesirable_weight": balance.undesirable_weight,
        },
    });
    out
}

#[test]
fn the_views_pair_and_keep_what_the_reference_does() {
    let fixture = fixture(env!("CARGO_MANIFEST_DIR"), "data/attempt-pairs").unwrap();
    let mut wrong = Vec::new();
    for case in fixture.cases() {
        if let Err(mismatch) = compare(&observed(case), case, Tolerance::new(1e-12, 0.0)) {
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
