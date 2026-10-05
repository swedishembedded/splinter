// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements task generation that learns from what was
// refused, for its clients. If your team needs expertise in synthetic data
// pipelines, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: task generation tells a kind that keeps being refused for one
//! reason how to correct it. Once a kind's proposals were refused for the
//! same reason often enough, every later request for it carries the
//! correction; before that, and for a kind with no refusals, none does.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::{scratch_context, Scripted};
use serde_json::json;
use splinter_agent::CancelToken;
use splinter_core::model_ref::ModelRef;
use splinter_pipelines::sources::{self, SourceTarget};
use splinter_pipelines::tasks::{generate, Generation, MIN_REFUSALS_BEFORE_CORRECTING};

/// A part of `sections` sections, each stating something of its own.
fn part(sections: usize) -> String {
    let mut text = String::from("# Notes\n\n");
    for n in 0..sections {
        text.push_str(&format!(
            "## S{n}\n\nThe machine number {n} is housed in the north shed and is oiled each Tuesday morning by the keeper.\n\n"
        ));
    }
    text
}

/// A generator that proposes `count` tasks, each citing a quote the section
/// does not hold word for word.
fn misquoting(count: usize) -> Scripted {
    Scripted::new(move |prompt| {
        if !prompt.contains("You write training tasks") {
            return String::new();
        }
        let tasks: Vec<_> = (0..count)
            .map(|n| {
                json!({
                    "instruction": format!("Where is machine {n} housed and when is it oiled?"),
                    "reference": "the north shed",
                    "evidence": [{ "section": 1, "quote": "a quotation nobody ever wrote" }],
                })
            })
            .collect();
        json!({ "tasks": tasks }).to_string()
    })
}

fn requests(policy: &Scripted) -> Vec<String> {
    policy
        .prompts
        .lock()
        .unwrap()
        .iter()
        .filter(|p| p.contains("You write training tasks"))
        .cloned()
        .collect()
}

#[test]
fn a_kind_refused_again_and_again_for_one_reason_is_told_how_to_correct_it() {
    let policy = misquoting(MIN_REFUSALS_BEFORE_CORRECTING);
    let (scratch, ctx) = scratch_context("guided-tasks", policy.clone(), false);
    let dir = scratch.0.join("parts");
    std::fs::create_dir_all(&dir).unwrap();
    // Nine sections: three windows, a request each.
    std::fs::write(dir.join("notes.md"), part(9)).unwrap();
    let id = sources::add(
        &ctx,
        &SourceTarget::from_learn_arg(&dir.display().to_string()).unwrap(),
    )
    .unwrap()
    .source
    .id;
    generate(
        &ctx,
        &Generation {
            sources: &[id],
            sections: &[],
            kinds: &["recall".to_string()],
            generator: &ModelRef::policy_default(),
            goal: None,
            author: None,
            deadline: None,
            cancel: CancelToken::new(),
        },
    )
    .unwrap();
    let seen = requests(&policy);
    assert!(seen.len() >= 2, "{} requests", seen.len());
    let correction = "word for word from the section it cites";
    assert!(
        !seen[0].contains(correction),
        "nothing refused yet, so nothing to correct"
    );
    assert!(
        seen[1].contains(correction),
        "the first window's refusals are corrected in the next request"
    );
}
