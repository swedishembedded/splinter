// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: what the prompts in the harness documents say. The prompts are data
//! (`harness/*.yaml`), so these specs read the documents, not Rust.

#![allow(clippy::unwrap_used)]

mod support;

use support::{agent_system, grounded, plan_only, plan_system, task};

/// The rules the grounded prompt gives after the diff.
fn rules() -> String {
    agent_system(&grounded())
        .split("How to work:")
        .nth(1)
        .unwrap_or_default()
        .to_string()
}

#[test]
fn the_review_step_compares_the_agents_own_diff_with_the_task() {
    let rules = rules();
    let review = rules
        .split("When every change is done")
        .nth(1)
        .unwrap_or_else(|| panic!("a review step is missing: {rules}"));
    assert!(
        review.contains("git diff") && review.contains("task"),
        "the review reads the agent's git diff against the task: {review}"
    );
    for hidden in ["change above", "the diff above", "masked", "xxx"] {
        assert!(
            !review.contains(hidden),
            "the review must not send the agent back to the hidden change ({hidden}): {review}"
        );
    }
}

#[test]
fn the_todo_list_asks_for_a_review_against_the_task() {
    let rules = rules();
    assert!(
        rules.contains("review my work against the task"),
        "the last todo item names the task: {rules}"
    );
}

#[test]
fn both_prompts_say_what_cannot_work_here_so_the_agent_does_not_try_to_work_around_it() {
    for prompt in [agent_system(&grounded()), agent_system(&plan_only())] {
        for part in ["no network", "git stash", "/tmp"] {
            assert!(prompt.contains(part), "{part} missing from {prompt}");
        }
    }
}

#[test]
fn the_prompts_say_where_the_repository_is_and_that_new_paths_must_be_given() {
    assert!(agent_system(&grounded()).contains("working directory is the repository root"));
    assert!(agent_system(&plan_only()).contains("working directory is the repository root"));
    assert!(plan_system("terse").contains("the exact new path"));
}

#[test]
fn the_replay_prompt_carries_the_per_file_procedure_and_the_budget() {
    let prompt = agent_system(&grounded());
    for part in [
        "calling todo with action set",
        "Locate.",
        "Read the region.",
        "Edit, minimally.",
        "Read the edited region again",
        "git diff --stat",
        "budget of {{ inputs.calls }} tool calls",
        "{{ inputs.masked_diff }}",
    ] {
        assert!(prompt.contains(part), "{part} missing");
    }
    assert!(plan_system("detailed").contains("Examples of good requests"));
}

#[test]
fn the_plan_only_prompt_shows_no_change() {
    let prompt = agent_system(&plan_only());
    assert!(!prompt.contains("masked_diff") && !prompt.contains("The change:"));
}

#[test]
fn the_supervisor_speaks_as_the_user_and_is_shown_the_change_for_its_eyes_only() {
    let doc = grounded();
    let author = &task(&doc, "work")["corrections"]["author"];
    let system = author["system"].as_str().unwrap();
    assert!(system.contains("You play the user who gave an engineer a task"));
    let request = author["task"].as_str().unwrap();
    assert!(request.contains("(for your eyes only)") && request.contains("{{ inputs.diff }}"));
}
