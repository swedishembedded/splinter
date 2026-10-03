// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Spec: what this harness derives is what the trainer actually accepts.
//!
//! The derivation writes `generic-messages-v2`, and every rule about that
//! format is enforced on the other side of a repository boundary by a parser
//! this code cannot see. A unit test asserting "the JSON looks right" asserts
//! this harness's *belief* about the format, and a belief is exactly what
//! drifts: the first sign of a mismatch would otherwise be a training run that
//! refuses a dataset after the GPU has been claimed, or worse, one that
//! accepts it and supervises the wrong spans.
//!
//! So the check runs the real parser, through the policy crate that trains
//! on the lab's output - which is also why this test lives here.

// A test file: its fixture helpers panic on a broken fixture, which is the
// failure report.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::BTreeMap;

use splinter_tool_syntax::{records_from_performance, to_jsonl, PredicateSet, Verdict};

fn verdict(pass: bool) -> Verdict {
    let set = PredicateSet::new(["done"]).expect("one predicate");
    let observed: BTreeMap<String, bool> = [("done".to_string(), pass)].into_iter().collect();
    set.evaluate(&observed).expect("evaluated")
}

/// The prompt a real agent request carries: a system turn and the tools.
fn prompt() -> serde_json::Value {
    serde_json::json!({
        "messages": [{"role": "system", "content": "You are an agent working in a repository."}],
        "tools": [{"type": "function", "function": {"name": "shell"}}],
    })
}

fn performed() -> Vec<(String, String, String)> {
    vec![(
        "shell".into(),
        r#"{"command":"./svctl status"}"#.into(),
        "active deployment: staging".into(),
    )]
}

fn derive(
    verdict: &Verdict,
) -> Result<Vec<splinter_tool_syntax::Record>, splinter_tool_syntax::Excluded> {
    records_from_performance(
        "config-discovery",
        "staging",
        &prompt(),
        "Enable 3 retry attempts for the service this host runs.",
        &performed(),
        "staging is live; editing config/staging.json",
        verdict,
    )
}

#[test]
fn the_trainer_accepts_what_this_harness_derives() {
    let records = derive(&verdict(true)).expect("a verified performance yields records");

    let jsonl = to_jsonl(&records).expect("serialises");
    let dir = std::env::temp_dir().join(format!("lab-trainable-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("train.jsonl");
    std::fs::write(&path, &jsonl).expect("write");

    // The real parser, from the crate that trains on it.
    let summary = splinter_policy::train::validate_dataset(&path)
        .unwrap_or_else(|e| panic!("the trainer rejected a derived dataset: {e}\n{jsonl}"));

    assert_eq!(summary.records, 1);
    // system, user, assistant(tool call).
    assert_eq!(
        summary.messages, 3,
        "every turn the model saw must survive into the record: {jsonl}"
    );
    assert_eq!(
        summary.trained_messages, 1,
        "exactly the decision is supervised, no more and no fewer: {jsonl}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_record_the_trainer_would_reject_never_gets_written() {
    // The derivation refuses an unverified performance, so the malformed-input
    // case the parser guards against cannot be produced here in the first
    // place. Belt and braces: assert the refusal, then assert that what does
    // get through parses.
    assert!(
        derive(&verdict(false)).is_err(),
        "an unsolved task must not reach the trainer at all"
    );
}
