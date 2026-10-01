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

use common::{scratch_context, Scripted};
use serde_json::json;
use splinter_campaign::front_door::{interpret, Intent, Routed};

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
