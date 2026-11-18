// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements causal diagnosis of why an agent failed, from
// the evidence its run left, for its clients. If your team needs expertise in
// telling model failures from infrastructure failures before you train on
// them, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: the deterministic diagnosis classifies real failed practice attempts
//! as a person reading them does, with precision and recall per class.
//!
//! `tests/fixtures/diagnosis.jsonl` holds failed attempts from real practice
//! exports (`bench export` of train-set practice runs of the student Qwen3-4B,
//! base and trained), each with the evidence the diagnosis reads (score
//! status and cause, the verifier's reason, whether a worktree change was
//! exported, the trajectory with long texts cut) and the class a person gave
//! it after reading the whole trajectory. Each record names its source export,
//! trial and manifest digest. No sealed or eval-set material is in it: every
//! record is of the train set.
//!
//! The person's labels follow the class definitions: `tool` when the agent's
//! last calls failed and it ended there without the artifact; `no_answer`
//! when its calls worked and it ended without writing the artifact (an
//! answer in its reply is not one); `malformed_artifact` when it wrote one the
//! verifier could not read; `wrong_content` when the verifier read it and
//! found it wrong; `resource` when a limit ended a run that was still working.

// A fixture that does not parse is a broken fixture file: the panic is the report.
#![allow(clippy::expect_used)]

use std::collections::BTreeMap;

use serde::Deserialize;
use splinter_core::outcome::FailureClass;
use splinter_eval::diagnosis::{diagnose, AttemptEvidence};

const FIXTURES: &str = include_str!("../fixtures/diagnosis.jsonl");

/// The least precision and recall each class must reach on the fixtures.
const FLOOR: f64 = 0.8;

#[derive(Deserialize)]
struct Evidence {
    status: Option<String>,
    cause_code: Option<String>,
    cause_message: Option<String>,
    verification_reason: Option<String>,
    workspace_changed: Option<bool>,
}

#[derive(Deserialize)]
struct Fixture {
    label: String,
    evidence: Evidence,
    trajectory: Option<atif::Trajectory>,
    source: serde_json::Value,
}

fn fixtures() -> Vec<Fixture> {
    FIXTURES
        .lines()
        .map(|line| serde_json::from_str(line).expect("a fixture parses"))
        .collect()
}

#[test]
fn every_fixture_is_train_set_practice_material() {
    for fixture in fixtures() {
        let set = fixture
            .source
            .get("set")
            .and_then(serde_json::Value::as_str);
        let skipped = fixture.source.get("skipped").is_some();
        assert!(set == Some("train") || skipped, "{}", fixture.source);
    }
}

#[test]
fn the_diagnosis_matches_the_labels_with_precision_and_recall_per_class() {
    // (true positives, predicted, labelled) per class.
    let mut counts: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
    let mut misses = Vec::new();
    for fixture in fixtures() {
        let e = &fixture.evidence;
        let result = diagnose(&AttemptEvidence {
            status: e.status.as_deref(),
            cause_code: e.cause_code.as_deref(),
            cause_message: e.cause_message.as_deref(),
            verification_reason: e.verification_reason.as_deref(),
            workspace_changed: e.workspace_changed,
            gateway_statuses: &[],
            trajectory: fixture.trajectory.as_ref(),
        });
        let label = FailureClass::parse(&fixture.label).expect("a label is a class");
        if result.class.is_model() {
            assert!(
                result.decisive_step.is_some() || result.class == FailureClass::Resource,
                "a model failure places its decisive step: {}",
                fixture.source
            );
        }
        let predicted = result.class.as_str().to_string();
        counts.entry(predicted.clone()).or_default().1 += 1;
        let entry = counts.entry(fixture.label.clone()).or_default();
        entry.2 += 1;
        if result.class == label {
            entry.0 += 1;
        } else {
            misses.push(format!(
                "{} -> {predicted} ({})",
                fixture.label, fixture.source
            ));
        }
    }
    let mut report = Vec::new();
    for (class, (hit, predicted, labelled)) in &counts {
        let precision = (*predicted > 0).then(|| *hit as f64 / *predicted as f64);
        let recall = (*labelled > 0).then(|| *hit as f64 / *labelled as f64);
        report.push(format!(
            "{class}: precision {precision:?} recall {recall:?} (n={labelled})"
        ));
        for value in [precision, recall].into_iter().flatten() {
            assert!(
                value >= FLOOR,
                "{class} below {FLOOR}:\n{}\n{}",
                report.join("\n"),
                misses.join("\n")
            );
        }
    }
    println!("{}", report.join("\n"));
}
