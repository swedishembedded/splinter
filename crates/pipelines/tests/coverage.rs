// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that cover a large body of writing
// within a time budget, for its clients. If your team needs expertise in
// turning a corpus too large to read into a bounded learning run, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: task generation over more text than a budget can cover is bounded
//! and spread. A part is shown to the generator through at most
//! `MAX_WINDOWS_PER_PART` windows of sections, evenly spaced over the part so
//! its end is represented as well as its start; the budget stops generation
//! inside a part, not only between parts; and parts are visited in a stable
//! order that does not follow their names, so a run that stops early has
//! covered a spread of the corpus and not its first files.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::time::{Duration, Instant};

use common::{scratch_context, Scratch, Scripted};
use serde_json::json;
use splinter_agent::CancelToken;
use splinter_core::model_ref::ModelRef;
use splinter_pipelines::sources::{self, SourceTarget};
use splinter_pipelines::tasks::{generate, Generation, MAX_WINDOWS_PER_PART};

/// A part of `sections` sections, each carrying its own marker.
fn part(tag: &str, sections: usize) -> String {
    let mut text = format!("# Part {tag}\n\n");
    for n in 0..sections {
        text.push_str(&format!("## S{n}\n\nMARK-{tag}-{n} is stated here.\n\n"));
    }
    text
}

fn run(
    scratch: &Scratch,
    ctx: &splinter_orchestrator::Context,
    files: &[(&str, String)],
    deadline: Option<Instant>,
) -> splinter_pipelines::tasks::TasksGenerated {
    let dir = scratch.0.join("parts");
    std::fs::create_dir_all(&dir).unwrap();
    for (name, text) in files {
        std::fs::write(dir.join(name), text).unwrap();
    }
    let target = SourceTarget::from_learn_arg(&dir.display().to_string()).unwrap();
    let id = sources::add(ctx, &target).unwrap().source.id;
    generate(
        ctx,
        &Generation {
            sources: &[id],
            sections: &[],
            kinds: &["recall".to_string()],
            generator: &ModelRef::policy_default(),
            goal: None,
            deadline,
            cancel: CancelToken::new(),
        },
    )
    .unwrap()
}

fn empty_reply(delay: Duration) -> Scripted {
    Scripted::new(move |prompt| {
        if prompt.contains("You write training tasks") {
            std::thread::sleep(delay);
        }
        json!({ "tasks": [] }).to_string()
    })
}

/// The generation requests the policy was sent.
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
fn a_long_part_is_shown_through_a_few_windows_spread_over_its_whole_length() {
    let policy = empty_reply(Duration::ZERO);
    let (scratch, ctx) = scratch_context("coverage-spread", policy.clone(), false);
    run(&scratch, &ctx, &[("long.md", part("L", 60))], None);
    let seen = requests(&policy);
    assert!(
        (2..=MAX_WINDOWS_PER_PART).contains(&seen.len()),
        "{} requests",
        seen.len()
    );
    assert!(seen.iter().any(|p| p.contains("MARK-L-0 ")), "its start");
    assert!(
        seen.iter().any(|p| p.contains("MARK-L-59 ")),
        "its end, not only its start"
    );
}

#[test]
fn a_short_part_is_covered_in_full() {
    let policy = empty_reply(Duration::ZERO);
    let (scratch, ctx) = scratch_context("coverage-short", policy.clone(), false);
    run(&scratch, &ctx, &[("short.md", part("S", 5))], None);
    let seen = requests(&policy).join("\n");
    for n in 0..5 {
        assert!(seen.contains(&format!("MARK-S-{n} ")), "section {n}");
    }
}

#[test]
fn the_budget_stops_generation_inside_a_part() {
    let policy = empty_reply(Duration::from_millis(300));
    let (scratch, ctx) = scratch_context("coverage-budget", policy.clone(), false);
    let generated = run(
        &scratch,
        &ctx,
        &[("long.md", part("L", 60))],
        Some(Instant::now() + Duration::from_millis(450)),
    );
    assert!(generated.stopped.is_some(), "{generated:#?}");
    assert!(
        requests(&policy).len() < MAX_WINDOWS_PER_PART,
        "{} requests",
        requests(&policy).len()
    );
}

#[test]
fn parts_are_visited_in_a_stable_order_that_does_not_follow_their_names() {
    let files: Vec<(String, String)> = (0..8)
        .map(|n| (format!("p{n}.md"), part(&format!("P{n}"), 3)))
        .collect();
    let named: Vec<(&str, String)> = files.iter().map(|(n, t)| (n.as_str(), t.clone())).collect();
    let order = |test: &str| -> Vec<usize> {
        let policy = empty_reply(Duration::ZERO);
        let (scratch, ctx) = scratch_context(test, policy.clone(), false);
        run(&scratch, &ctx, &named, None);
        let mut visited: Vec<usize> = requests(&policy)
            .iter()
            .map(|p| (0..8).find(|n| p.contains(&format!("MARK-P{n}-"))).unwrap())
            .collect();
        visited.dedup();
        visited
    };
    let first = order("coverage-order-a");
    assert_eq!(first, order("coverage-order-b"), "stable");
    assert_ne!(first, (0..8).collect::<Vec<_>>(), "not in name order");
    let mut sorted = first.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..8).collect::<Vec<_>>(), "every part is visited");
}

/// Three proposals a request, none of which names a subject, so each is
/// refused: the way a question about a letter fails a kind that asks for one.
fn unadmittable() -> Scripted {
    Scripted::new(|prompt| {
        if !prompt.contains("You write training tasks") {
            return String::new();
        }
        let task = |n: u32| {
            json!({
                "instruction": format!("What is stated in the passage, question {n}?"),
                "reference": "it is stated",
                "evidence": [{ "section": 0, "quote": "is stated here" }]
            })
        };
        json!({ "tasks": [task(1), task(2), task(3)] }).to_string()
    })
}

#[test]
fn a_kind_the_sources_cannot_satisfy_is_given_up_on_instead_of_asked_for_on_every_window() {
    let policy = unadmittable();
    let (scratch, ctx) = scratch_context("coverage-give-up", policy.clone(), false);
    let files: Vec<(String, String)> = (0..12)
        .map(|n| (format!("p{n}.md"), part(&format!("P{n}"), 20)))
        .collect();
    let named: Vec<(&str, String)> = files.iter().map(|(n, t)| (n.as_str(), t.clone())).collect();
    let generated = run(&scratch, &ctx, &named, None);

    let asked = requests(&policy).len();
    assert!(
        asked * 3 < 12 * MAX_WINDOWS_PER_PART * 3 / 2,
        "{asked} requests: a kind that was refused every time kept being asked for"
    );
    assert!(
        asked >= 8,
        "it was asked for enough times to be sure: {asked}"
    );
    let why = generated.dropped.get("recall").expect("it says it gave up");
    assert!(why.contains("admitted 0"), "{why}");
    assert_eq!(generated.tasks, 0);
}
