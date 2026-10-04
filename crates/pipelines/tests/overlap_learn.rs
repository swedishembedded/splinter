// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning agents whose held-out measurement
// cannot leak, for its clients. If your team needs expertise in evaluating a
// model on documents it has not seen when its sources overlap, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: the dataset a `learn` writes names, for each record, the group of
//! overlapping source text it came from, so the records of two prints of one
//! letter share a group and a held-out split keeps them together; a record
//! from an unrelated letter is in a group of its own.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::{Arc, Mutex};

use common::gate::{gate_context, Brain, FakeTrainer, ANCHOR};
use common::Scripted;
use serde_json::json;
use splinter_agent::solve::{Model, MATERIAL_HEADING};
use splinter_agent::CancelToken;
use splinter_core::model_ref::ModelRef;
use splinter_core::role::Role;
use splinter_model::train::{Trained, TrainedPreference};
use splinter_orchestrator::{Context, OrchestratorError};
use splinter_pipelines::dialogue::STUDENT_ROLE;
use splinter_pipelines::learn::{learn, LearnRequest, Learned};
use splinter_pipelines::release::arm;
use splinter_pipelines::train::{TrainPlan, Trainer};

/// `i` spelled in letters, so a word carries no digit and the text is prose.
fn letters(mut i: usize) -> String {
    let mut out = String::new();
    loop {
        out.push(char::from(b'a' + u8::try_from(i % 26).unwrap()));
        i /= 26;
        if i == 0 {
            return out;
        }
    }
}

/// `n` lower-case words that no other seed shares.
fn words(seed: &str, n: usize) -> String {
    (0..n)
        .map(|i| format!("{seed}x{}", letters(i)))
        .collect::<Vec<_>>()
        .join(" ")
}

struct Student;

impl Trainer for Student {
    fn train(
        &self,
        ctx: &Context,
        plan: &TrainPlan,
        cancel: &CancelToken,
    ) -> Result<Trained, OrchestratorError> {
        let trained = FakeTrainer::knowing(&[ANCHOR]).train(ctx, plan, cancel)?;
        ctx.add_model(
            arm(ctx.config(), Some(&trained.adapter)),
            Model::new(
                Arc::new(Scripted::new(|_| "I do not know.".into())),
                "scripted/learned",
            ),
        );
        Ok(trained)
    }

    fn train_preference(
        &self,
        _ctx: &Context,
        _plan: &TrainPlan,
        _cancel: &CancelToken,
    ) -> Result<TrainedPreference, OrchestratorError> {
        panic!("learn trains chat datasets only")
    }
}

#[test]
fn records_of_two_prints_of_one_letter_share_a_group_and_another_letter_has_its_own() {
    let body = format!("I believe that {}", words("a", 130));
    let other = format!("I believe that {}", words("c", 130));
    let letters = [
        ("one.md", format!("# One\n\n## Body\n\n{body}\n")),
        (
            "one-reprinted.md",
            format!("# One, reprinted\n\n## Body\n\nDear sir {body} yours\n"),
        ),
        ("two.md", format!("# Two\n\n## Body\n\n{other}\n")),
    ];
    let seen = Arc::new(Mutex::new(0usize));
    let counter = seen.clone();
    let policy = Scripted::new(move |prompt| {
        if prompt.contains("You write training tasks") {
            let (seed, text) = if prompt.contains(&words("c", 5)) {
                ("c", other_text())
            } else {
                ("a", body_text())
            };
            let n = {
                let mut n = counter.lock().unwrap();
                *n += 1;
                *n
            };
            let passage: String = text.split(' ').take(40).collect::<Vec<_>>().join(" ");
            json!({ "tasks": [{
                "instruction": format!("question {n} about {seed}xa matters"),
                "reference": passage,
                "evidence": [{ "section": 0, "quote": passage }]
            }]})
            .to_string()
        } else if prompt.contains(STUDENT_ROLE) {
            json!({ "message": "and then?" }).to_string()
        } else if prompt.contains(MATERIAL_HEADING) {
            if prompt.contains(&words("c", 5)) {
                "cxa cxb cxc cxd".to_string()
            } else {
                "axa axb axc axd".to_string()
            }
        } else {
            "I do not know.".to_string()
        }
    });
    fn body_text() -> String {
        format!("I believe that {}", words("a", 130))
    }
    fn other_text() -> String {
        format!("I believe that {}", words("c", 130))
    }

    let (scratch, ctx) = gate_context("overlap-learn", Brain::Missing);
    ctx.add_model(
        ModelRef::policy_default(),
        Model::new(Arc::new(policy), common::POLICY),
    );
    // A judge that passes an answer in the words of the reference's letter.
    let judge_ref: ModelRef = "local:test/judge".parse().unwrap();
    let judge = Scripted::new(|prompt| {
        let reference = prompt
            .split("REFERENCE:\\n")
            .nth(1)
            .and_then(|t| t.split("\\n\\nANSWER:").next())
            .unwrap_or_default();
        let answer = prompt.split("ANSWER:\\n").nth(1).unwrap_or_default();
        if ["axa", "cxa"]
            .iter()
            .any(|key| reference.contains(key) && answer.contains(key))
        {
            "PASS\nit does".into()
        } else {
            "FAIL\nit does not".into()
        }
    });
    ctx.add_model(
        judge_ref.clone(),
        Model::new(Arc::new(judge), "scripted/judge"),
    );
    let dir = scratch.0.join("letters");
    std::fs::create_dir_all(&dir).unwrap();
    for (name, text) in &letters {
        std::fs::write(dir.join(name), text).unwrap();
    }
    let Learned::Ran(ran) = learn(
        &ctx,
        &LearnRequest {
            sources: vec![dir.display().to_string()],
            kinds: vec!["converse".into()],
            roles: [(Role::Judge, judge_ref)].into(),
            no_release: true,
            distill: true,
            ..LearnRequest::default()
        },
        &Student,
    )
    .unwrap() else {
        panic!("a learn that is not a dry run runs");
    };

    let dataset = ran.report.dataset.as_ref().unwrap();
    let text = std::fs::read_to_string(&dataset.path).unwrap();
    let groups: Vec<(String, Option<String>)> = text
        .lines()
        .map(|line| {
            let record: serde_json::Value = serde_json::from_str(line).unwrap();
            (
                record["messages"][1]["content"]
                    .as_str()
                    .unwrap()
                    .to_string(),
                record["metadata"]["group"].as_str().map(str::to_string),
            )
        })
        .collect();
    assert_eq!(groups.len(), 3, "{groups:?}");
    let group_of = |marker: &str| -> Vec<&Option<String>> {
        groups
            .iter()
            .filter(|(q, _)| q.contains(marker))
            .map(|(_, g)| g)
            .collect()
    };
    let a = group_of("axa");
    let c = group_of("cxa");
    assert_eq!(a.len(), 2);
    assert!(a[0].is_some() && a[0] == a[1], "{groups:?}");
    assert!(c[0].is_some() && c[0] != a[0], "{groups:?}");
}
