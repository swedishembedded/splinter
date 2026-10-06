// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuned models that answer under the
// prompt they were trained under, for its clients. If your team needs
// expertise in training and serving persona models, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a model that answers under a system prompt of its own is asked
//! under it, wherever it is asked; a model with none is asked under the
//! default, so the teacher and the judge never speak as the person the policy
//! was trained to be.

// Helpers outside a test function unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::Arc;

use common::gate::{
    anchor_file, candidate_on, dataset_under, decide, gate_context, reply, request_log, Brain,
    ANCHOR, FACTS,
};
use common::{scratch_context, Scripted};
use splinter_agent::solve::{Model, SYSTEM_PROMPT};
use splinter_core::model_ref::ModelRef;
use splinter_pipelines::ask::ask;
use splinter_pipelines::release::{anchor, arm};

const PERSONA: &str = "You are a surveyor of the old school. Answer as one.";

#[test]
fn a_model_is_asked_under_its_own_system_prompt_and_another_under_the_default() {
    let (_scratch, ctx) = scratch_context("system-prompt", Scripted::new(|_| String::new()), false);
    let policy = Scripted::new(|_| "Measure twice.".into());
    let other = Scripted::new(|_| "Measure once.".into());
    ctx.add_model(
        ModelRef::policy_default(),
        Model::new(Arc::new(policy.clone()), "scripted/policy").with_system(PERSONA),
    );
    let assistant: ModelRef = "local:test/assistant".parse().unwrap();
    ctx.add_model(
        assistant.clone(),
        Model::new(Arc::new(other.clone()), "scripted/assistant"),
    );
    ask(
        &ctx,
        "How do you measure?",
        None,
        &ModelRef::policy_default(),
    )
    .unwrap();
    ask(&ctx, "How do you measure?", None, &assistant).unwrap();
    assert_eq!(policy.systems.lock().unwrap()[0], [PERSONA]);
    assert_eq!(other.systems.lock().unwrap()[0], [SYSTEM_PROMPT]);
}

#[test]
fn a_release_answers_under_the_prompt_its_datasets_were_trained_under() {
    let (scratch, ctx) = gate_context("system-prompt-release", Brain::Honest);
    anchor::freeze(&ctx, &[anchor_file(&scratch.0, 4)]).unwrap();
    let facts: Vec<usize> = (0..FACTS).collect();
    let data = dataset_under(&ctx, "alpha", &facts, Some(PERSONA));
    let (candidate, _) = candidate_on(&ctx, data, &[ANCHOR, "alpha"]);
    // What a candidate was trained under is what its datasets say.
    assert_eq!(
        ctx.system_prompt_of(&candidate.datasets)
            .unwrap()
            .as_deref(),
        Some(PERSONA)
    );
    // Datasets of two people are refused.
    let other = dataset_under(&ctx, "beta", &facts, Some("You are a mason."));
    let both = [candidate.datasets[0].clone(), other];
    assert!(ctx.system_prompt_of(&both).is_err());

    let decided = decide(&ctx, &candidate);
    assert!(decided.release.is_some(), "{:#?}", decided.gate);
    // A later command resolves the alias afresh and finds the persona.
    ctx.add_model(
        ModelRef::policy_default(),
        Model::new(
            Arc::new(Scripted::new(|_| "ok".into())),
            "scripted/released",
        ),
    );
    let asking = splinter_orchestrator::Context::on(std::sync::Arc::clone(ctx.runtime()));
    let model = asking.model(&ModelRef::policy_default()).unwrap();
    assert_eq!(model.system.as_deref(), Some(PERSONA));
}

#[test]
fn the_served_candidate_is_asked_under_the_prompt_it_was_trained_under() {
    let (scratch, ctx) = gate_context("system-prompt-serve", Brain::Honest);
    anchor::freeze(&ctx, &[anchor_file(&scratch.0, 4)]).unwrap();
    let facts: Vec<usize> = (0..FACTS).collect();
    let data = dataset_under(&ctx, "alpha", &facts, Some(PERSONA));
    let (candidate, _) = candidate_on(&ctx, data, &[ANCHOR, "alpha"]);
    let decided = decide(&ctx, &candidate);
    assert!(decided.release.is_some(), "{:#?}", decided.gate);
    let log = std::fs::read_to_string(request_log(&scratch)).unwrap();
    let asked: Vec<Vec<String>> = log
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert!(!asked.is_empty());
    assert!(
        asked
            .iter()
            .all(|systems| systems == &[PERSONA.to_string()]),
        "{asked:?}"
    );
}

#[test]
fn the_anchor_is_asked_under_the_default_prompt_by_both_arms_and_the_report_says_which() {
    let (scratch, ctx) = gate_context("system-prompt-anchor", Brain::Honest);
    anchor::freeze(&ctx, &[anchor_file(&scratch.0, 4)]).unwrap();
    let facts: Vec<usize> = (0..FACTS).collect();
    let data = dataset_under(&ctx, "alpha", &facts, Some(PERSONA));
    let (candidate, _) = candidate_on(&ctx, data, &[ANCHOR, "alpha"]);
    // Both arms record the system turns they are asked under.
    let topics = |known: &[&str]| -> Scripted {
        let known: Vec<String> = known.iter().map(|t| t.to_string()).collect();
        Scripted::new(move |prompt| reply(&known, prompt))
    };
    let base = topics(&[ANCHOR]);
    let tuned = topics(&[ANCHOR, "alpha"]);
    ctx.add_model(
        arm(ctx.config(), None),
        Model::new(Arc::new(base.clone()), "scripted/base"),
    );
    ctx.add_model(
        arm(ctx.config(), Some(&candidate.adapter)),
        Model::new(Arc::new(tuned.clone()), "scripted/tuned"),
    );
    let decided = decide(&ctx, &candidate);
    assert!(decided.release.is_some(), "{:#?}", decided.gate);

    let systems_for = |model: &Scripted, topic: &str| -> Vec<Vec<String>> {
        let prompts = model.prompts.lock().unwrap();
        let systems = model.systems.lock().unwrap();
        prompts
            .iter()
            .zip(systems.iter())
            .filter(|(prompt, _)| prompt.contains(&format!("the {topic} code")))
            .map(|(_, systems)| systems.clone())
            .collect()
    };
    let only = |asked: Vec<Vec<String>>, system: &str| {
        assert!(!asked.is_empty());
        assert!(
            asked.iter().all(|s| s == &[system.to_string()]),
            "{asked:?}"
        );
    };
    only(systems_for(&tuned, ANCHOR), SYSTEM_PROMPT);
    only(systems_for(&base, ANCHOR), SYSTEM_PROMPT);
    only(systems_for(&tuned, "alpha"), PERSONA);
    only(systems_for(&base, "alpha"), SYSTEM_PROMPT);

    let prompts = &decided.gate.prompts;
    let of = |name: &str| prompts.iter().find(|p| p.suite.starts_with(name)).unwrap();
    assert!(of("held-out")
        .candidate
        .starts_with("persona: You are a surveyor"));
    assert_eq!(of("held-out").champion, "default");
    assert_eq!(of("anchor").candidate, "default");
    assert_eq!(of("anchor").champion, "default");
}
