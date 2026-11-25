// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-data pipelines held to a second,
// independent implementation, for its clients. If your team needs expertise
// in on-policy self-distillation data, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Spec, against an independent reference (plan 6B): for each frozen case,
//! the policy-gradient view under each distill setting writes exactly the
//! records the reference (`validation/oracles/data/self_distill.py`, written
//! from the view's documentation) wrote: the distilled attempts with a teacher
//! context that says whether the attempt passed and quotes the verifier, the
//! others with their reward and group, and leaves out exactly what it left
//! out, for the reason it gave.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};
use splinter_conformance::{compare, fixture, Tolerance};
use splinter_core::annotation::Outcome;
use splinter_core::experience::ExperienceId;
use splinter_data::{Corpus, Distill, Projection, RecordBody, View};

use super::{add, view, Trial};

const REPORTED: &str = "The verifier reported: ";

/// The corpus of `case`, and the trial of each experience in it.
fn corpus(case: &Value) -> (Corpus, BTreeMap<ExperienceId, String>) {
    let mut corpus = Corpus::new();
    let mut trials = BTreeMap::new();
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
        let mut evidence = json!({
            "trial": trial.name,
            "task": {"id": "cad/bracket", "version": trial.version},
            "model": trial.model,
            "metrics": {},
        });
        if let Some(reason) = c["reason"].as_str() {
            evidence["verification"] = json!({ "reason": reason });
        }
        add(&mut corpus, &trial, Some(evidence));
        let id = corpus.entries().last().unwrap().id.clone();
        trials.insert(id, trial.name.clone());
    }
    (corpus, trials)
}

fn observed(
    case: &Value,
    projection: &Projection,
    trials: &BTreeMap<ExperienceId, String>,
) -> Value {
    let mut groups: BTreeMap<String, usize> = BTreeMap::new();
    let records: Vec<Value> = projection
        .records
        .iter()
        .map(|r| {
            let RecordBody::Chat {
                outcome,
                teacher_context,
                ..
            } = &r.body
            else {
                panic!("a chat record, got {:?}", r.body);
            };
            let trial = &trials[&r.metadata.experiences[0]];
            match (outcome, &teacher_context[..]) {
                (Some(outcome), []) => {
                    let next = groups.len();
                    let group = *groups.entry(outcome.group.clone()).or_insert(next);
                    json!({"trial": trial, "reward": outcome.reward, "group": group})
                }
                (None, [context]) => {
                    let text = &context.content;
                    let reason = text
                        .split_once(REPORTED)
                        .map(|(_, rest)| rest.split("\n\n").next().unwrap());
                    json!({
                        "trial": trial,
                        "passed": text.contains("This attempt passed every hidden check."),
                        "reason": reason,
                    })
                }
                other => panic!("a reward or a teacher context, got {other:?}"),
            }
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

fn distill(name: &str) -> Distill {
    match name {
        "cliff" => Distill::Cliff,
        "every" => Distill::Every,
        _ => Distill::Off,
    }
}

#[test]
fn the_view_distils_and_rewards_what_the_reference_does() {
    let fixture = fixture(env!("CARGO_MANIFEST_DIR"), "data/self-distill").unwrap();
    let mut wrong = Vec::new();
    for case in fixture.cases() {
        let (corpus, trials) = corpus(case);
        let projection = view()
            .distilling(distill(case["distill"].as_str().unwrap()))
            .project(&corpus)
            .unwrap();
        if let Err(mismatch) = compare(
            &observed(case, &projection, &trials),
            case,
            Tolerance::new(0.0, 0.0),
        ) {
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
