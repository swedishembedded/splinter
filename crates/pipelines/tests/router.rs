// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: the front door. A sentence is classified by the policy model,
//! through sven's typed method call, into one intent with its arguments;
//! code - not the model - decides what happens next. A confident, single
//! reading runs as the command it maps to; an ambiguous or unconfident one
//! becomes a question back, never an action; a reading that needs the
//! network without the opt-in is refused, and one that is destructive or
//! opts into the network is never run on a guess.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::{config, scratch_context, Scratch, Scripted, POLICY};
use serde_json::json;
use splinter_pipelines::router::{interpret, interpret_with, Intent, Routed};

/// The classification the scripted policy gives each sentence.
fn classifier() -> Scripted {
    Scripted::new(|prompt| {
        let reply = if prompt.contains("learn ./docs") {
            json!({ "candidates": [{
                "intent": { "verb": "learn", "sources": ["./docs"] },
                "confidence": 0.94
            }]})
        } else if prompt.contains("what does X do?") {
            json!({ "candidates": [{
                "intent": { "verb": "ask", "question": "what does X do?" },
                "confidence": 0.9
            }]})
        } else if prompt.contains("the docs thing") {
            // Both confident, too close to tell apart.
            json!({ "candidates": [
                { "intent": { "verb": "learn", "sources": ["./docs"] }, "confidence": 0.85 },
                { "intent": { "verb": "add_source", "target": "./docs" }, "confidence": 0.8 }
            ]})
        } else if prompt.contains("maybe the docs") {
            // One reading, not confident enough.
            json!({ "candidates": [
                { "intent": { "verb": "learn", "sources": ["./docs"] }, "confidence": 0.5 }
            ]})
        } else if prompt.contains("ask the big remote model") {
            json!({ "candidates": [{
                "intent": {
                    "verb": "ask",
                    "question": "what is X?",
                    "policy": "remote:openrouter/z-ai/glm-5.3-flash"
                },
                "confidence": 0.97
            }]})
        } else if prompt.contains("explain the flags") {
            // A verb no command has, however often it is corrected.
            json!({ "candidates": [{
                "intent": { "verb": "explain", "question": "the flags" },
                "confidence": 0.9
            }]})
        } else if prompt.contains("materials she wrote")
            || prompt.contains("from her letters in ./nowhere")
        {
            // The model garbles the path it was given.
            json!({ "candidates": [{
                "intent": {
                    "verb": "learn",
                    "sources": ["./typo/ada/letters"],
                    "goal": "think like Ada Lovelace"
                },
                "confidence": 0.93
            }]})
        } else if prompt.contains("stop that run") {
            json!({ "candidates": [{
                "intent": { "verb": "cancel_run", "run": "run-1" },
                "confidence": 0.99
            }]})
        } else {
            json!({ "candidates": [] })
        };
        reply.to_string()
    })
}

#[test]
fn a_confident_reading_runs_as_its_command() {
    let (_scratch, ctx) = scratch_context("front-door-run", classifier(), false);
    let Routed::Execute(intent) = interpret(&ctx, "learn ./docs").unwrap() else {
        panic!("a confident learn runs");
    };
    assert_eq!(
        intent,
        Intent::Learn {
            sources: vec!["./docs".into()],
            goal: None
        }
    );
    assert_eq!(intent.command_line(), ["learn", "./docs"]);

    let Routed::Execute(intent) = interpret(&ctx, "what does X do?").unwrap() else {
        panic!("a confident question is asked");
    };
    assert_eq!(intent.command_line(), ["ask", "what does X do?"]);
}

#[test]
fn an_ambiguous_reading_is_a_question_not_an_action() {
    let (_scratch, ctx) = scratch_context("front-door-ambiguous", classifier(), false);
    let Routed::Clarify(question) = interpret(&ctx, "the docs thing").unwrap() else {
        panic!("two close readings are a question");
    };
    assert_eq!(question.candidates.len(), 2);
    assert_eq!(question.candidates[0].command, ["learn", "./docs"]);
    assert_eq!(question.candidates[1].command, ["source", "add", "./docs"]);

    let Routed::Clarify(question) = interpret(&ctx, "maybe the docs").unwrap() else {
        panic!("an unconfident reading is a question");
    };
    assert_eq!(question.candidates.len(), 1);

    let Routed::Clarify(question) = interpret(&ctx, "gibberish").unwrap() else {
        panic!("no reading is a question");
    };
    assert!(question.candidates.is_empty());
}

#[test]
fn a_reading_that_needs_the_network_is_refused_without_the_opt_in() {
    let (_scratch, ctx) = scratch_context("front-door-remote", classifier(), false);
    let Routed::Refuse(why) = interpret(&ctx, "ask the big remote model").unwrap() else {
        panic!("a remote model without the opt-in is refused");
    };
    assert!(why.contains("--allow-remote"), "{why}");

    // With the opt-in it is still not run on a guess: it is confirmed.
    let (_scratch, opted_in) = scratch_context("front-door-remote-ok", classifier(), true);
    assert!(matches!(
        interpret(&opted_in, "ask the big remote model").unwrap(),
        Routed::Clarify(_)
    ));
}

#[test]
fn a_destructive_reading_is_confirmed_not_run() {
    let (_scratch, ctx) = scratch_context("front-door-cancel", classifier(), false);
    let Routed::Clarify(question) = interpret(&ctx, "stop that run").unwrap() else {
        panic!("a cancel is never run on a guess");
    };
    assert_eq!(question.candidates[0].command, ["runs", "cancel", "run-1"]);
}

/// A local policy has no constrained decoding: it learns the shape of a
/// classification, and how a sentence maps onto one, only from its prompt.
#[test]
fn the_policy_is_shown_the_classification_schema() {
    let policy = classifier();
    let (_scratch, ctx) = scratch_context("front-door-schema", policy.clone(), false);
    interpret(&ctx, "learn ./docs").unwrap();
    let prompts = policy.prompts.lock().unwrap();
    // Names only the schema carries: the verbs, as the reply must spell them.
    for name in [
        "add_source",
        "list_sources",
        "cancel_run",
        "cmd:<tool> --help",
    ] {
        assert!(
            prompts[0].contains(name),
            "{name} is not shown: {}",
            prompts[0]
        );
    }
}

/// A reply the policy cannot shape into a classification, even after
/// correction, is no reading: it is asked back, as a sentence that matches
/// no command is, never an error.
#[test]
fn an_unreadable_classification_is_a_question_not_a_failure() {
    let (_scratch, ctx) = scratch_context("front-door-unreadable", classifier(), false);
    let Routed::Clarify(question) = interpret(&ctx, "explain the flags").unwrap() else {
        panic!("an unreadable reply is a question");
    };
    assert!(question.candidates.is_empty());
    assert!(question.reason.contains("explain"), "{}", question.reason);
}

/// A path the model got wrong is corrected from the sentence, by code: when
/// the reading's source is no file or directory but the sentence names exactly
/// one that exists, that is what the person meant.
#[test]
fn a_path_the_model_garbled_is_corrected_from_the_sentence() {
    let (scratch, ctx) = scratch_context("front-door-path", classifier(), false);
    let letters = scratch.0.join("ada-letters");
    std::fs::create_dir_all(&letters).unwrap();
    let sentence = format!(
        "Learn to think like Ada based on the materials she wrote in directory {}.",
        letters.display()
    );
    let Routed::Execute(Intent::Learn { sources, goal }) = interpret(&ctx, &sentence).unwrap()
    else {
        panic!("a confident learn runs");
    };
    assert_eq!(sources, [letters.display().to_string()]);
    assert_eq!(goal.as_deref(), Some("think like Ada Lovelace"));
}

#[test]
fn a_source_that_exists_or_is_not_a_path_is_left_as_the_model_wrote_it() {
    let (scratch, ctx) = scratch_context("front-door-path-kept", classifier(), false);
    // The model's path is wrong and the sentence names no path that exists:
    // nothing is invented, so the run is refused later where the path is read.
    let Routed::Execute(Intent::Learn { sources, .. }) = interpret(
        &ctx,
        "Learn to think like Ada from her letters in ./nowhere/at/all",
    )
    .unwrap() else {
        panic!("a confident learn runs");
    };
    assert_eq!(sources, ["./typo/ada/letters"]);
    drop(scratch);
}

#[test]
fn the_policy_is_shown_how_a_persona_sentence_is_read() {
    let policy = classifier();
    let (_scratch, ctx) = scratch_context("front-door-persona", policy.clone(), false);
    interpret(&ctx, "learn ./docs").unwrap();
    let prompts = policy.prompts.lock().unwrap();
    assert!(
        prompts[0].contains("think like"),
        "a sentence naming a person to think like is shown as a learn: {}",
        prompts[0]
    );
}

/// The front door can be given a stronger model than the policy to read a
/// sentence with: the configuration names it, and the policy is not asked.
#[test]
fn a_model_named_in_the_configuration_reads_the_sentence_instead_of_the_policy() {
    let scratch = Scratch::new("front-door-model");
    let mut settings = config(&scratch);
    settings.front_door_model = Some("local:./reader".into());
    let reader = classifier();
    let policy = Scripted::new(|_| json!({ "candidates": [] }).to_string());
    let ctx = splinter_orchestrator::Context::new(settings, false)
        .unwrap()
        .with_model(
            splinter_core::model_ref::ModelRef::policy_default(),
            splinter_agent::solve::Model::new(std::sync::Arc::new(policy.clone()), POLICY),
        )
        .with_model(
            "local:./reader".parse().unwrap(),
            splinter_agent::solve::Model::new(
                std::sync::Arc::new(reader.clone()),
                "scripted/reader",
            ),
        );
    let Routed::Execute(_) = interpret(&ctx, "learn ./docs").unwrap() else {
        panic!("the reader's confident reading runs");
    };
    assert!(
        policy.prompts.lock().unwrap().is_empty(),
        "the policy was not asked"
    );
    assert_eq!(reader.prompts.lock().unwrap().len(), 1);
}

#[test]
fn a_front_door_model_that_is_no_model_reference_is_refused_by_name() {
    let scratch = Scratch::new("front-door-model-bad");
    let mut settings = config(&scratch);
    settings.front_door_model = Some("not a reference".into());
    let ctx = splinter_orchestrator::Context::new(settings, false).unwrap();
    let error = interpret(&ctx, "learn ./docs").err().unwrap();
    assert!(error.is_refusal(), "{error}");
    assert!(error.to_string().contains("not a reference"), "{error}");
}

/// A classifier that returns what it was given: routing without a model.
struct Canned(Vec<(Intent, f64)>);

impl splinter_pipelines::router::IntentClassifier for Canned {
    fn classify(
        &self,
        _: &splinter_orchestrator::Context,
        _: &str,
    ) -> Result<splinter_pipelines::router::Classification, splinter_orchestrator::OrchestratorError>
    {
        Ok(splinter_pipelines::router::Classification {
            candidates: self
                .0
                .iter()
                .map(
                    |(intent, confidence)| splinter_pipelines::router::Candidate {
                        intent: intent.clone(),
                        confidence: *confidence,
                    },
                )
                .collect(),
        })
    }
}

#[test]
fn routing_is_decided_by_code_whatever_classifier_read_the_sentence() {
    let (_scratch, ctx) = scratch_context("router-canned", Scripted::new(|_| String::new()), false);
    let routed = interpret_with(
        &Canned(vec![(Intent::Status, 0.95)]),
        &ctx,
        "how are things?",
    )
    .unwrap();
    assert!(
        matches!(routed, Routed::Execute(Intent::Status)),
        "{routed:?}"
    );

    // The classifier's own confidence does not make a destructive command
    // run on a guess.
    let routed = interpret_with(
        &Canned(vec![(
            Intent::CancelRun {
                run: "run-1".into(),
            },
            0.99,
        )]),
        &ctx,
        "stop it",
    )
    .unwrap();
    assert!(matches!(routed, Routed::Clarify(_)), "{routed:?}");
}
