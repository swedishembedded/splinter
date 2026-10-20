// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements pipelines that turn a repository's history
// into verified training data for coding agents, for its clients. If your
// team needs expertise in learning from your own codebase, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: the agent's self-review checks its own `git diff` against the TASK it
//! was given (which the training record contains), never against the change
//! shown in the grounding prompt (which the record does not contain).

#![allow(clippy::unwrap_used)]

use splinter_repo_replay::grounding_prompt::grounding_prompt;
use splinter_repo_replay::mask::MaskedDiff;

/// The rules the grounding prompt gives after the diff.
fn rules() -> String {
    let prompt = grounding_prompt(&MaskedDiff::default(), 30);
    prompt
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
