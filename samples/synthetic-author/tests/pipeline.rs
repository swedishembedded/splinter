// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements controlled validation of learning systems
// against ground truth that cannot have been seen in pretraining, for its
// clients. If your team needs expertise in proving that a trained model has
// learned what a corpus implies, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Spec: a model is taught an author's decisions from the author's articles
//! and asked to act on situations it has not met, through tools it has not
//! seen. What was decided is read out of an article and kept only if code can
//! check it; what a model does is the argument of a tool call, compared as a
//! string with what the beliefs determine.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use serde_json::json;
use synthetic_author::cache::StoredArticle;
use synthetic_author::gates::{Chose, Reading};
use synthetic_author::mine::{verify, Mined, MinedCase};
use synthetic_author::prompts::{render, SYSTEM};
use synthetic_author::records::chat_record;
use synthetic_author::scenario::Archetype;
use synthetic_author::tools::{HELD_OUT, TRAINING};

const OPTIONS: [Archetype; 3] = [
    Archetype::ActAlone,
    Archetype::DeferToOutsider,
    Archetype::HoldAndWait,
];

#[test]
fn a_tool_offers_exactly_the_options_of_the_situation_under_its_own_labels() {
    for schema in TRAINING.iter().chain(HELD_OUT.iter()) {
        let def = schema.definition(&OPTIONS);
        assert_eq!(def["function"]["name"], schema.name);
        let labels = def["function"]["parameters"]["properties"][schema.choice]["enum"]
            .as_array()
            .unwrap();
        assert_eq!(labels.len(), 3);
        let required = def["function"]["parameters"]["required"]
            .as_array()
            .unwrap();
        assert!(required.contains(&json!(schema.choice)));
        // Every label reads back as the option it names.
        for (position, &option) in OPTIONS.iter().enumerate() {
            let label = schema.label(option, position);
            let args = json!({ schema.choice: label, schema.reason: "because" });
            assert_eq!(schema.read(schema.name, &args, &OPTIONS), Some(option));
        }
    }
}

#[test]
fn a_call_to_another_tool_or_with_a_label_not_offered_reads_as_no_answer() {
    let schema = &TRAINING[0];
    let args = json!({ schema.choice: schema.label(OPTIONS[0], 0) });
    assert_eq!(schema.read("some_other_tool", &args, &OPTIONS), None);
    assert_eq!(
        schema.read(schema.name, &json!({ schema.choice: "nonsense" }), &OPTIONS),
        None
    );
    assert_eq!(schema.read(schema.name, &json!({}), &OPTIONS), None);
}

#[test]
fn the_tools_a_model_is_examined_with_are_not_the_ones_it_was_trained_with() {
    for held in HELD_OUT.iter() {
        for trained in TRAINING.iter() {
            assert_ne!(held.name, trained.name);
            assert_ne!(held.choice, trained.choice);
            assert_ne!(held.reason, trained.reason);
        }
    }
}

#[test]
fn a_prompt_gives_the_briefing_and_each_option_under_the_label_the_tool_uses() {
    let schema = &HELD_OUT[1];
    let text = render("A dispute over a ferry.", &OPTIONS, schema);
    assert!(text.contains("A dispute over a ferry."));
    for (position, &option) in OPTIONS.iter().enumerate() {
        assert!(
            text.contains(&format!("{}:", schema.label(option, position))),
            "{text}"
        );
    }
    assert!(
        !SYSTEM.to_lowercase().contains("convict"),
        "the system turn names no belief"
    );
}

#[test]
fn a_training_record_supervises_only_the_tool_call() {
    let schema = &TRAINING[1];
    let record = chat_record(
        "g1",
        "A dispute over a ferry.",
        &OPTIONS,
        schema,
        Archetype::DeferToOutsider,
        "We took the help because the water was rising.",
    );
    let messages = record["messages"].as_array().unwrap();
    let roles: Vec<_> = messages
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, ["system", "user", "assistant"]);
    let trained: Vec<_> = messages
        .iter()
        .map(|m| m["train"].as_bool().unwrap())
        .collect();
    assert_eq!(trained, [false, false, true]);
    let call = &messages[2]["tool_calls"][0]["function"];
    assert_eq!(call["name"], schema.name);
    let args: serde_json::Value =
        serde_json::from_str(call["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(
        schema.read(schema.name, &args, &OPTIONS),
        Some(Archetype::DeferToOutsider)
    );
    assert_eq!(
        args[schema.reason],
        "We took the help because the water was rising."
    );
    assert_eq!(record["tools"].as_array().unwrap().len(), 1);
    assert_eq!(record["metadata"]["group"], "g1");
}

fn article() -> StoredArticle {
    StoredArticle {
        index: 0,
        topic: 0,
        title: "t".into(),
        body: "Of the Dunmere granary I will say this. The winter came early and the stores ran low. We took the outside help because the water was rising fast.\n\nOf the Osmere ferry I will say little. Nothing moved that season.".into(),
        cases: vec!["the Dunmere granary".into(), "the Osmere ferry".into()],
        attempts: 1,
    }
}

fn mined(case: &str, briefing: &str, chose: Chose, reasons: &[&str]) -> MinedCase {
    MinedCase {
        case: case.into(),
        briefing: briefing.into(),
        offer_made: true,
        chose,
        reasons: reasons.iter().map(|s| (*s).to_string()).collect(),
    }
}

#[test]
fn a_mined_case_keeps_only_reasons_found_word_for_word_in_the_article() {
    let briefing = "A winter shortage faces the Dunmere granary, and an outside house has offered help, while the water is rising near the stores and nobody yet knows how long it will last this year.";
    let m = Mined {
        cases: vec![
            mined(
                "the Dunmere granary",
                briefing,
                Chose::DeferToOutsider,
                &[
                    "We took the outside help because the water was rising fast.",
                    "an invented reason",
                ],
            ),
            mined("the Osmere ferry", "short", Chose::HoldAndWait, &[]),
        ],
    };
    let out = verify(&article(), &m);
    let ok = out[0].as_ref().unwrap();
    assert_eq!(ok.chose, Archetype::DeferToOutsider);
    assert_eq!(
        ok.reasons,
        ["We took the outside help because the water was rising fast."]
    );
    assert_eq!(ok.dropped, 1);
    // A briefing too short to be one is refused.
    assert!(out[1].is_err());
}

#[test]
fn a_briefing_that_gives_the_decision_away_or_a_case_left_unclear_is_refused() {
    let leaky = "The winter came early and the stores ran low, so that in the end we took the outside help because the water was rising fast, and so it was.";
    let m = Mined {
        cases: vec![
            mined(
                "the Dunmere granary",
                leaky,
                Chose::DeferToOutsider,
                &["We took the outside help because the water was rising fast."],
            ),
            mined(
                "the Osmere ferry",
                "A ferry operator faces a season in which little can be moved across the river and the community must decide what to do about the crossing and its fees.",
                Chose::Unclear,
                &[],
            ),
        ],
    };
    let out = verify(&article(), &m);
    assert!(out[0].as_ref().unwrap_err().contains("gives away"));
    assert!(out[1].as_ref().unwrap_err().contains("unclear"));
}

#[test]
fn a_reading_of_an_article_is_a_list_of_cases() {
    // The reader's shape is the one the writer's gate already uses.
    let r = Reading { cases: vec![] };
    assert!(r.cases.is_empty());
}
