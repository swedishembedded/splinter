// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: supervisor correction turns. When an attempt ends with a wrong tree a
//! supervisor, who knows the wanted change, sends the agent up to a few
//! follow-up USER messages. They speak as a user would (intent, behaviour,
//! place, in words) and carry no oracle: a message that quotes the change,
//! gives a line number, refers to the change or names what only the change
//! has is refused and counted. In the record, what the agent did before the
//! last correction is context and only what it did after is trained on; the
//! final tree must match exactly. First-try and corrected records are kept
//! apart.

#![allow(clippy::unwrap_used)]

use std::collections::HashSet;

use splinter_repo_replay::audit::{
    default_config, history_diffs, open_context, run_blocking, scripted_models,
};
use splinter_repo_replay::dataset::{attempts_set_filtered, build_dataset, DatasetSettings};
use splinter_repo_replay::fixture;
use splinter_repo_replay::justify::Support;
use splinter_repo_replay::ledger::{self, Entry, Outcome};
use splinter_repo_replay::pipeline::Models;
use splinter_repo_replay::plan_prompt::PlanStyle;
use splinter_repo_replay::scripted::{
    Behaviour, ScriptedSupervisor, Specificity, SupervisorBehaviour,
};
use splinter_repo_replay::serving_prompt::SERVING_PROMPT;
use splinter_repo_replay::supervise::{check_message, MessageContext};

const DIFF: &str = "diff --git a/src/calc.rs b/src/calc.rs\n--- a/src/calc.rs\n+++ b/src/calc.rs\n@@ -3,3 +3,4 @@\n fn total(price: u32) -> u32 {\n-    price\n+    apply_surcharge(price, 4096)\n }\n";

fn refusal(message: &str) -> Option<&'static str> {
    let parent: HashSet<String> = ["total", "price", "calc"].map(String::from).into();
    let mut support = Support::of("Please make total() in src/calc.rs charge a surcharge.");
    support.add("L3:fn total(price: u32) -> u32 {");
    let ctx = MessageContext::new(
        "Please make total() in src/calc.rs charge a surcharge.",
        DIFF,
        &parent,
        &support,
    );
    check_message(message, &ctx).err().map(|r| r.class())
}

#[test]
fn a_correction_speaks_as_a_user_and_carries_no_oracle() {
    assert_eq!(
        refusal("The surcharge is still missing: total() should add it to the price it returns."),
        None
    );
    assert_eq!(
        refusal("Make it return apply_surcharge(price, 4096) there."),
        Some("diff_line"),
        "a changed line quoted"
    );
    assert_eq!(
        refusal("Look at line 4 of src/calc.rs, the return value is wrong."),
        Some("location_offset")
    );
    assert_eq!(
        refusal("Your result does not match the diff I wanted."),
        Some("diff_reference")
    );
    assert_eq!(
        refusal("total() should call a function named apply_surcharge."),
        Some("leaked_target"),
        "a name only the change has"
    );
    assert_eq!(
        refusal("The surcharge should be 4096 in total()."),
        Some("leaked_target"),
        "a number only the change has"
    );
    assert_eq!(
        refusal("The surcharge should be 8192 in total()."),
        Some("unjustified_content"),
        "a number nobody gave the agent"
    );
    assert_eq!(
        refusal("Add the helper called `compute_fee_for_order` next to total()."),
        Some("unjustified_content"),
        "a code-shaped name nobody gave the agent"
    );
}

struct Run {
    work: tempfile::TempDir,
    entries: Vec<Entry>,
    ctx: splinter_sdk::Context,
}

/// Replays the three-file commit with an agent that skips its last file, a
/// supervisor of `supervisor`'s kind and up to `corrections` messages.
fn run_multi(supervisor: SupervisorBehaviour, corrections: usize, behaviour: Behaviour) -> Run {
    let work = tempfile::tempdir().unwrap();
    let source = work.path().join("source");
    std::fs::create_dir(&source).unwrap();
    let ids = fixture::build(&source).unwrap();
    let truths = history_diffs(&source).unwrap();
    let state = work.path().join("state");
    let ctx = open_context(&state).unwrap();
    let mut config = default_config(&source, &state);
    config.from = Some(ids[4].clone());
    config.to = Some(ids[5].clone());
    config.styles = vec![PlanStyle::Terse];
    config.corrections = corrections;
    let models = Models {
        supervisor: splinter_sdk::agent::solve::Model::new(
            ScriptedSupervisor::provider(supervisor),
            "scripted/supervisor",
        ),
        ..scripted_models(behaviour, Specificity::NamesFiles, &truths)
    };
    run_blocking(&ctx, &models, &config).unwrap();
    let entries = ledger::read(&state)
        .unwrap()
        .into_iter()
        .filter(|e| e.outcome != Outcome::Skipped)
        .collect();
    Run { work, entries, ctx }
}

#[test]
fn a_corrected_attempt_passes_with_the_correction_counted_and_only_the_fix_trained_on() {
    let run = run_multi(
        SupervisorBehaviour::Honest,
        3,
        Behaviour::FixesAfterCorrection,
    );
    let [entry] = &run.entries[..] else {
        panic!("{:?}", run.entries)
    };
    assert_eq!(entry.outcome, Outcome::Passed, "{entry:?}");
    assert_eq!(entry.corrections, 1);
    assert!(entry.supervisor_refusals.is_empty());

    let state = run.work.path().join("state");
    let set =
        attempts_set_filtered(&run.ctx, &state, "corrected", &|e| e.corrections >= 1).unwrap();
    let built = build_dataset(
        &run.ctx,
        &[set],
        &DatasetSettings {
            serving_prompt: SERVING_PROMPT.to_string(),
            max_block: None,
        },
    )
    .unwrap();
    let text = std::fs::read_to_string(&built.path).unwrap();
    let record: serde_json::Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
    let messages = record["messages"].as_array().unwrap();
    let users: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m["role"] == "user")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(users.len(), 2, "the plan and one correction");
    let correction = users[1];
    assert!(
        messages[..correction]
            .iter()
            .all(|m| m["train"] != serde_json::json!(true)),
        "what came before the correction is context only"
    );
    assert!(
        messages[correction..]
            .iter()
            .any(|m| m["role"] == "assistant" && m["train"] == serde_json::json!(true)),
        "the fix after it is trained on"
    );
    let said = messages[correction]["content"].as_str().unwrap();
    assert!(!said.contains("@@") && !said.contains("diff"), "{said}");
}

#[test]
fn a_first_try_pass_has_no_corrections_and_the_two_record_types_filter_apart() {
    let run = run_multi(SupervisorBehaviour::Honest, 3, Behaviour::Honest);
    let [entry] = &run.entries[..] else {
        panic!("{:?}", run.entries)
    };
    assert_eq!((entry.outcome, entry.corrections), (Outcome::Passed, 0));
    let state = run.work.path().join("state");
    assert!(attempts_set_filtered(&run.ctx, &state, "corrected", &|e| e.corrections >= 1).is_err());
    assert!(attempts_set_filtered(&run.ctx, &state, "first-try", &|e| e.corrections == 0).is_ok());
    let bias = ledger::bias(&run.entries);
    assert_eq!((bias.first_try_passed, bias.corrected_passed), (1, 0));
}

#[test]
fn a_leaking_supervisor_message_is_refused_counted_and_never_sent() {
    for (kind, class) in [
        (SupervisorBehaviour::QuotesDiffLine, "diff_line"),
        (SupervisorBehaviour::GivesLineNumber, "location_offset"),
        (SupervisorBehaviour::ReferencesTheDiff, "diff_reference"),
    ] {
        let run = run_multi(kind, 3, Behaviour::FixesAfterCorrection);
        let [entry] = &run.entries[..] else {
            panic!("{:?}", run.entries)
        };
        assert_eq!(entry.outcome, Outcome::Failed, "{kind:?}: {entry:?}");
        assert_eq!(entry.corrections, 0, "{kind:?}");
        assert!(
            entry.supervisor_refusals.iter().any(|c| c == class),
            "{kind:?}: {:?}",
            entry.supervisor_refusals
        );
    }
}

#[test]
fn without_corrections_the_same_attempt_just_fails() {
    let run = run_multi(
        SupervisorBehaviour::Honest,
        0,
        Behaviour::FixesAfterCorrection,
    );
    let [entry] = &run.entries[..] else {
        panic!("{:?}", run.entries)
    };
    assert_eq!((entry.outcome, entry.corrections), (Outcome::Failed, 0));
}
