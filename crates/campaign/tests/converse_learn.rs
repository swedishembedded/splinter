// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning agents that learn what a person
// advised from what they wrote, for its clients. If your team needs
// expertise in grounding a model's advice in a body of writing, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: `learn` over a person's letters with the `converse` kind teaches the
//! policy to talk as the person, from a dialogue the teacher holds with a
//! simulated other speaker.
//!
//! The teacher is shown the letter; the other speaker never is. The training
//! record is the whole conversation, the writer's every reply supervised and
//! the letter in none of it. A dialogue whose replies state a year or name
//! the letter does not hold teaches nothing: the grounding check fails it
//! and no record is made. The other speaker is sometimes told to ask, on its
//! last turn, for a detail the exchange has not given, so that the student
//! also sees what declining to invent looks like.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::{Arc, Mutex};

use common::gate::{gate_context, Brain, FakeTrainer, ANCHOR};
use common::Scripted;
use serde_json::json;
use splinter_agent::solve::{Model, MATERIAL_HEADING};
use splinter_campaign::dialogue::{probes_beyond_the_source, STUDENT_ROLE};
use splinter_campaign::learn::{learn, LearnRequest, Learned};
use splinter_campaign::model_ref::ModelRef;
use splinter_campaign::release::arm;
use splinter_campaign::train::{TrainPlan, Trainer};
use splinter_campaign::{CampaignError, Context};
use splinter_policy::train::{Trained, TrainedPreference};
use sven_sdk::CancelToken;

const LETTER: &str = "# To a young man

## Study

I advise you to fix a habit of study every morning before you do anything else. Never let a day pass without reading something of history or ethics, and always write down what you have read, for the memory fails what the pen has not fixed.
";

const PASSAGE: &str = "I advise you to fix a habit of study every morning before you do anything else. Never let a day pass without reading something of history or ethics, and always write down what you have read, for the memory fails what the pen has not fixed.";

const OPENING: &str = "My mornings vanish in idleness; how do you keep a habit of study?";

/// A policy that writes one opening from the letter, plays the other speaker
/// with fixed follow-ups, and, as the teacher shown the letter, replies with
/// `third` as its last reply.
fn policy(third: &'static str) -> Scripted {
    Scripted::new(move |prompt| {
        if prompt.contains("You write training tasks") {
            json!({ "tasks": [{
                "instruction": OPENING,
                "reference": PASSAGE,
                "evidence": [{ "section": 1, "quote": PASSAGE }]
            }]})
            .to_string()
        } else if prompt.contains(STUDENT_ROLE) {
            let message = if prompt.contains("And then?") {
                "Why does the pen matter so much?"
            } else {
                "And then?"
            };
            json!({ "message": message }).to_string()
        } else if prompt.contains(MATERIAL_HEADING) {
            if prompt.contains("Why does the pen matter so much?") {
                third.to_string()
            } else if prompt.contains("And then?") {
                "Read some history or ethics each day, and write down what you read.".to_string()
            } else {
                "Fix a habit of study every morning, before anything else.".to_string()
            }
        } else {
            "I do not know.".to_string()
        }
    })
}

/// A trainer that records what it was asked to train on.
struct Student {
    plans: Mutex<Vec<TrainPlan>>,
}

impl Trainer for Student {
    fn train(
        &self,
        ctx: &Context,
        plan: &TrainPlan,
        cancel: &CancelToken,
    ) -> Result<Trained, CampaignError> {
        self.plans.lock().unwrap().push(plan.clone());
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

fn run(
    test: &str,
    third: &'static str,
) -> (splinter_campaign::learn::LearnReport, common::Scratch) {
    let (scratch, ctx) = gate_context(test, Brain::Missing);
    ctx.add_model(
        ModelRef::policy_default(),
        Model::new(Arc::new(policy(third)), common::POLICY),
    );
    let letter = scratch.0.join("letter.md");
    std::fs::write(&letter, LETTER).unwrap();
    let student = Student {
        plans: Mutex::new(Vec::new()),
    };
    let Learned::Ran(ran) = learn(
        &ctx,
        &LearnRequest {
            sources: vec![letter.display().to_string()],
            kinds: vec!["converse".into()],
            no_release: true,
            distill: true,
            ..LearnRequest::default()
        },
        &student,
    )
    .unwrap() else {
        panic!("a learn that is not a dry run runs");
    };
    (ran.report, scratch)
}

#[test]
fn a_grounded_dialogue_becomes_one_record_of_the_whole_conversation() {
    let (report, _scratch) = run(
        "converse-learn-grounded",
        "The pen fixes what the memory lets slip, so write down what you have read.",
    );
    let taught = report.teach.as_ref().unwrap();
    assert_eq!(
        (taught.verify.passed, taught.verify.failed),
        (1, 0),
        "{taught:#?}"
    );
    let dataset = report.dataset.as_ref().unwrap();
    assert_eq!(dataset.records, 1, "{dataset:#?}");
    let text = std::fs::read_to_string(&dataset.path).unwrap();
    let record: serde_json::Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
    let turns: Vec<(&str, bool)> = record["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| (m["role"].as_str().unwrap(), m["train"].as_bool().unwrap()))
        .collect();
    assert_eq!(
        turns,
        [
            ("system", false),
            ("user", false),
            ("assistant", true),
            ("user", false),
            ("assistant", true),
            ("user", false),
            ("assistant", true),
        ]
    );
    assert_eq!(record["messages"][1]["content"], OPENING);
    assert_eq!(record["messages"][3]["content"], "And then?");
    assert!(
        !text.contains(MATERIAL_HEADING),
        "the letter is teacher-only"
    );
    assert!(
        !text.contains(STUDENT_ROLE),
        "the other speaker's instructions are not training data"
    );
}

#[test]
fn a_dialogue_that_states_a_year_the_letter_does_not_hold_teaches_nothing() {
    let (report, _scratch) = run(
        "converse-learn-invents",
        "I took up this habit in 1762 and kept it all my life.",
    );
    let taught = report.teach.as_ref().unwrap();
    assert_eq!(
        (taught.verify.passed, taught.verify.failed),
        (0, 1),
        "{taught:#?}"
    );
    assert!(report.dataset.is_none(), "{report:#?}");
}

#[test]
fn one_dialogue_in_four_ends_by_asking_beyond_the_letter_and_the_choice_is_stable() {
    let digests: Vec<_> = (0..400u32)
        .map(|n| splinter_record::digest::Digest::of(&n.to_le_bytes()))
        .collect();
    let probing = digests
        .iter()
        .filter(|d| probes_beyond_the_source(d))
        .count();
    assert!((60..=140).contains(&probing), "{probing} of 400");
    assert!(digests
        .iter()
        .all(|d| probes_beyond_the_source(d) == probes_beyond_the_source(d)));
}
