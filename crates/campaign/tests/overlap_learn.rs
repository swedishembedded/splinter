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
use splinter_campaign::dialogue::STUDENT_ROLE;
use splinter_campaign::learn::{learn, LearnRequest, Learned};
use splinter_campaign::release::arm;
use splinter_campaign::train::{TrainPlan, Trainer};
use splinter_campaign::{CampaignError, Context};
use splinter_core::model_ref::ModelRef;
use splinter_model::train::{Trained, TrainedPreference};

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
    ) -> Result<Trained, CampaignError> {
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
    ) -> Result<TrainedPreference, CampaignError> {
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
            "axa axb axc axd".to_string()
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
