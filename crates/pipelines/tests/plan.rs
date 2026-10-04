// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that survey a body of writing and
// plan what to learn from it, for its clients. If your team needs expertise
// in turning an unfamiliar corpus into a learning plan, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Spec: a planner model reads a survey of the sources and the learner's goal
//! and chooses how to learn from them - which kinds of task, whether to
//! distil from a teacher, who the learner is becoming - and code holds the
//! choice to what the sources hold. The model may only choose from the menu;
//! it may not choose to teach advice the survey did not find; a plan that
//! breaks a rule is sent back once with the rule it broke, and one still
//! broken is refused, never replaced by a default.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::{Arc, Mutex};

use common::{scratch_context, Scripted};
use splinter_agent::CancelToken;
use splinter_core::model_ref::ModelRef;
use splinter_knowledge::survey::Survey;
use splinter_pipelines::plan::{plan, MIN_ADVICE_SECTIONS, MIN_JUDGMENT_SECTIONS};

fn survey(advice_sections: usize, judgment_sections: usize) -> Survey {
    Survey {
        parts: 300,
        text_bytes: 4_000_000,
        sections: 9000,
        advice_sections,
        judgment_sections,
        names: vec!["to-carr.txt".into(), "to-jay.txt".into()],
        excerpts: vec!["I advise you to fix a habit of study every morning.".into()],
    }
}

const GOOD: &str = r#"{"persona": "Thomas Jefferson", "kinds": ["advise", "recall"], "distill": true, "rationale": "the letters hold advice a small model cannot answer closed-book"}"#;

/// A planner that replies with `replies` in order and remembers its prompts.
fn planner(replies: Vec<&'static str>) -> (Scripted, Arc<Mutex<usize>>) {
    let asked = Arc::new(Mutex::new(0));
    let counter = asked.clone();
    let replies = Arc::new(replies);
    (
        Scripted::new(move |_prompt| {
            let mut n = counter.lock().unwrap();
            let reply = replies[(*n).min(replies.len() - 1)];
            *n += 1;
            reply.to_string()
        }),
        asked,
    )
}

fn run(
    test: &str,
    replies: Vec<&'static str>,
    advice: usize,
) -> (
    Result<splinter_pipelines::plan::Plan, splinter_orchestrator::OrchestratorError>,
    Scripted,
    usize,
) {
    run_surveyed(test, replies, advice, advice)
}

fn run_surveyed(
    test: &str,
    replies: Vec<&'static str>,
    advice: usize,
    judgment: usize,
) -> (
    Result<splinter_pipelines::plan::Plan, splinter_orchestrator::OrchestratorError>,
    Scripted,
    usize,
) {
    let (model, asked) = planner(replies);
    let (_scratch, ctx) = scratch_context(test, model.clone(), false);
    let result = plan(
        &ctx,
        &survey(advice, judgment),
        Some("think like Thomas Jefferson"),
        &ModelRef::policy_default(),
        &CancelToken::new(),
    );
    let count = *asked.lock().unwrap();
    (result, model, count)
}

#[test]
fn the_planner_chooses_from_the_menu_and_is_shown_the_survey_and_the_goal() {
    let (result, model, asked) = run("plan-good", vec![GOOD], MIN_ADVICE_SECTIONS * 100);
    let chosen = result.unwrap();
    assert_eq!(chosen.kinds, ["advise", "recall"]);
    assert!(chosen.distill);
    assert_eq!(chosen.persona.as_deref(), Some("Thomas Jefferson"));
    assert_eq!(asked, 1, "a valid plan needs no correction");
    let prompt = model.prompts.lock().unwrap()[0].clone();
    assert!(
        prompt.contains("think like Thomas Jefferson"),
        "the goal reaches the planner"
    );
    assert!(
        prompt.contains("9000") && prompt.contains("habit of study"),
        "the survey reaches it"
    );
}

#[test]
fn a_kind_outside_the_menu_is_corrected_once_and_then_accepted() {
    let bad = r#"{"persona": null, "kinds": ["telepathy"], "distill": false, "rationale": "x"}"#;
    let (result, _, asked) = run("plan-menu", vec![bad, GOOD], 100);
    assert_eq!(result.unwrap().kinds, ["advise", "recall"]);
    assert_eq!(asked, 2, "the first plan was sent back");
}

#[test]
fn advice_is_refused_when_the_survey_found_none_and_nothing_replaces_the_plan() {
    let (result, _, asked) = run("plan-no-advice", vec![GOOD], 0);
    assert!(
        result.is_err(),
        "a plan that teaches advice the sources lack is refused"
    );
    assert!(asked >= 2, "it was sent back before being refused");
}

#[test]
fn a_plan_with_no_kinds_or_the_same_kind_twice_is_not_a_plan() {
    let none = r#"{"persona": null, "kinds": [], "distill": false, "rationale": "x"}"#;
    assert!(run("plan-none", vec![none], 100).0.is_err());
    let twice =
        r#"{"persona": null, "kinds": ["recall", "recall"], "distill": false, "rationale": "x"}"#;
    assert!(run("plan-twice", vec![twice], 100).0.is_err());
}

#[test]
fn a_conversation_needs_sections_where_the_writer_judges_and_not_advice_alone() {
    let converse = r#"{"persona": "Thomas Jefferson", "kinds": ["converse", "recall"], "distill": true, "rationale": "the letters show how the writer talks and reasons"}"#;
    // Positions stated without counsel are enough: no advice in the sources.
    let chosen = run_surveyed("plan-converse", vec![converse], 0, MIN_JUDGMENT_SECTIONS)
        .0
        .unwrap();
    assert_eq!(chosen.kinds, ["converse", "recall"]);
    // Sources that are all business and news give a conversation nothing to
    // teach, and the plan is refused.
    let (result, _, asked) = run_surveyed("plan-converse-none", vec![converse], 0, 0);
    assert!(result.is_err());
    assert!(asked >= 2, "it was sent back before being refused");
}

#[test]
fn a_goal_to_think_like_a_person_needs_a_kind_that_teaches_their_judgment() {
    // Recall and explanation alone teach the facts of the writing, not how
    // the writer reasons: with advice and judgment in the sources, a plan that
    // names a person and chooses neither is sent back.
    let facts_only = r#"{"persona": "Thomas Jefferson", "kinds": ["recall", "explain"], "distill": true, "rationale": "the letters hold facts"}"#;
    let (result, _, asked) = run_surveyed(
        "plan-persona-facts-only",
        vec![facts_only, GOOD],
        MIN_ADVICE_SECTIONS,
        MIN_JUDGMENT_SECTIONS,
    );
    assert_eq!(result.unwrap().kinds, ["advise", "recall"]);
    assert_eq!(asked, 2, "the facts-only plan was sent back");

    // Where the sources hold no advice and no judgment there is nothing to
    // teach: the facts-only plan stands.
    let (result, _, asked) = run_surveyed("plan-persona-no-judgment", vec![facts_only], 0, 0);
    assert_eq!(result.unwrap().kinds, ["recall", "explain"]);
    assert_eq!(asked, 1);

    // No person, no such rule.
    let nobody =
        r#"{"persona": null, "kinds": ["recall"], "distill": false, "rationale": "a datasheet"}"#;
    let (result, _, asked) = run_surveyed(
        "plan-no-persona",
        vec![nobody],
        MIN_ADVICE_SECTIONS,
        MIN_JUDGMENT_SECTIONS,
    );
    assert!(result.is_ok());
    assert_eq!(asked, 1);
}
