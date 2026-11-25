// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-data pipelines held to a second,
// independent implementation, for its clients. If your team needs expertise
// in preference data from corrected agent runs, you can procure our services
// by sending an email to info@swedishembedded.com.

//! Spec, against an independent reference (plan 6B): for each frozen case,
//! the preference view pairs each teacher's repair with the failures of its
//! task exactly as the reference (`validation/oracles/data/repair_pairs.py`,
//! written from the view's documentation) did: from where the two
//! conversations part, the shared prompt and each side's first message, and
//! leaves out exactly what it left out, for the reason it gave.

use serde_json::{json, Map, Value};
use splinter_conformance::{compare, fixture, Tolerance};
use splinter_core::annotation::Outcome;
use splinter_data::{Corpus, View};

use super::{add, pairs, texts, view, Attempt};

/// A command of the fixture's alphabet, as the helper keeps it.
fn command(text: &str) -> &'static str {
    ["cat ledger.csv", "ls", "book accrual", "book wrong"]
        .into_iter()
        .find(|c| *c == text)
        .unwrap()
}

fn corpus(case: &Value) -> Corpus {
    let mut corpus = Corpus::new();
    for a in case["attempts"].as_array().unwrap() {
        let trial = a["trial"].as_str().unwrap();
        let base = match a["verdict"].as_str().unwrap() {
            "pass" => Attempt::pass(trial),
            "fail" => Attempt::fail(trial),
            _ => Attempt::infrastructure(trial),
        };
        let last = if base.verdict == Outcome::Pass {
            "Booked."
        } else {
            "Done."
        };
        let commands = a["commands"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| command(c.as_str().unwrap()))
            .collect();
        let attempt = base
            .commands(commands, last)
            .model(if a["teacher"].as_bool().unwrap() {
                "teacher"
            } else {
                "student"
            })
            .copied(a["copied"].as_u64().unwrap() as usize);
        add(&mut corpus, &attempt);
    }
    corpus
}

fn observed(case: &Value) -> Value {
    let projection = view().project(&corpus(case)).unwrap();
    let records: Vec<Value> = pairs(&projection)
        .iter()
        .map(|p| {
            json!({
                "chosen": p.identity.rollouts[0],
                "rejected": p.identity.rollouts[1],
                "prompt": texts(&p.prompt),
                "chosen_first": texts(&p.chosen)[0],
                "rejected_first": texts(&p.rejected)[0],
            })
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
fn repairs_are_paired_and_left_out_as_the_reference_does() {
    let fixture = fixture(env!("CARGO_MANIFEST_DIR"), "data/repair-pairs").unwrap();
    let mut wrong = Vec::new();
    for case in fixture.cases() {
        if let Err(mismatch) = compare(&observed(case), case, Tolerance::new(0.0, 0.0)) {
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
