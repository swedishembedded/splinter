// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-data pipelines held to a second,
// independent implementation, for its clients. If your team needs expertise
// in validating reinforcement-learning data, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec, against an independent reference (plan 6B): for each frozen case,
//! the policy-gradient view writes exactly the records the reference
//! (`validation/oracles/data/policy_gradient_groups.py`, written from the
//! view's documentation) wrote, each as its trial, its reward and its group,
//! groups numbered in the order they first appear, and leaves out exactly the
//! experiences it left out, for the reason it gave.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};
use splinter_conformance::{compare, fixture, Tolerance};
use splinter_core::annotation::Outcome;
use splinter_data::{Corpus, Projection, RecordBody, View};

use super::{add, view, Trial};

/// The corpus of `case`: its conversations in order, each a trial's
/// conversation in its role.
fn corpus(case: &Value) -> Corpus {
    let mut corpus = Corpus::new();
    for c in case["conversations"].as_array().unwrap() {
        let text = |key: &str| c[key].as_str().unwrap();
        let trial = Trial {
            verdict: if text("verdict") == "pass" {
                Outcome::Pass
            } else {
                Outcome::Fail
            },
            metrics: &[],
            ..Trial::pass(text("trial"))
        }
        .version(text("version"))
        .model(text("model"))
        .role(text("role"));
        add(&mut corpus, &trial, None);
    }
    corpus
}

fn observed(case: &Value, projection: &Projection) -> Value {
    let mut groups: BTreeMap<String, usize> = BTreeMap::new();
    let records: Vec<Value> = projection
        .records
        .iter()
        .map(|r| {
            let RecordBody::Chat {
                outcome: Some(outcome),
                ..
            } = &r.body
            else {
                panic!("a chat record with an outcome, got {:?}", r.body);
            };
            let next = groups.len();
            let group = *groups.entry(outcome.group.clone()).or_insert(next);
            json!({"trial": outcome.rollout, "reward": outcome.reward, "group": group})
        })
        .collect();
    let excluded: Map<String, Value> = projection
        .excluded
        .iter()
        .map(|(reason, n)| (format!("{reason:?}"), json!(n)))
        .collect();
    let mut out = case.clone();
    out["records"] = Value::Array(records);
    out["excluded"] = Value::Object(excluded);
    out
}

#[test]
fn the_view_groups_and_writes_what_the_reference_does() {
    let fixture = fixture(env!("CARGO_MANIFEST_DIR"), "data/policy-gradient-groups").unwrap();
    let mut wrong = Vec::new();
    for case in fixture.cases() {
        let projection = view().project(&corpus(case)).unwrap();
        if let Err(mismatch) = compare(&observed(case, &projection), case, Tolerance::new(0.0, 0.0))
        {
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
