// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning agents whose held-out measurement
// cannot leak, for its clients. If your team needs expertise in evaluating a
// model on documents it has not seen when its sources overlap, you can
// procure our services by sending an email to info@swedishembedded.com.

//! Spec: the tasks of a held-out suite cluster by the family of source text
//! they are grounded in: two tasks about two prints of one letter share a
//! cluster, a task about an unrelated letter has its own, and a task
//! grounded in nothing has none, so a significance test counts a family once.

// Helpers outside a test function unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::{scratch_context, Scripted};
use splinter_core::experience::{Environment, Span, Task};
use splinter_pipelines::grouping::task_clusters;
use splinter_pipelines::sources::{self, SourceTarget};

/// `i` spelled in letters, so a word carries no digit and the text is prose.
fn letters(mut i: usize) -> String {
    let mut out = String::new();
    loop {
        out.push(char::from(b'a' + u8::try_from(i % 26).unwrap()));
        i /= 26;
        if i == 0 {
            return out;
        }
    }
}

/// `n` lower-case words that no other seed shares.
fn words(seed: &str, n: usize) -> String {
    (0..n)
        .map(|i| format!("{seed}x{}", letters(i)))
        .collect::<Vec<_>>()
        .join(" ")
}

fn span_of(
    ctx: &splinter_orchestrator::Context,
    text: &str,
    dir: &std::path::Path,
    name: &str,
) -> Span {
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    let id = sources::add(
        ctx,
        &SourceTarget::from_learn_arg(&path.display().to_string()).unwrap(),
    )
    .unwrap()
    .source
    .id;
    let content = ctx.sources().get_source(&id).unwrap().parts[0]
        .content
        .clone();
    Span::new(content, 0, text.len() as u64).unwrap()
}

fn task(span: Option<&Span>, question: &str) -> Task {
    Task::new(
        "recall",
        span.cloned().into_iter().collect(),
        Environment::closed_book(),
        question,
        vec![],
    )
    .unwrap()
}

#[test]
fn tasks_cluster_by_the_family_of_text_they_are_grounded_in() {
    let (scratch, ctx) = scratch_context("clusters", Scripted::new(|_| String::new()), false);
    let letter = words("alpha", 120);
    // A second print of the letter: the same words with a few changed.
    let reprint = format!("{} {}", letter, words("coda", 6));
    let other = words("omega", 120);
    let first = span_of(&ctx, &letter, &scratch.0, "first.txt");
    let second = span_of(&ctx, &reprint, &scratch.0, "second.txt");
    let third = span_of(&ctx, &other, &scratch.0, "third.txt");
    let tasks = [
        task(Some(&first), "What was said in the first print?"),
        task(Some(&second), "What was said in the second print?"),
        task(Some(&third), "What was said in the other letter?"),
        task(None, "What is a question about nothing?"),
    ];
    let clusters = task_clusters(&ctx, &tasks).unwrap();
    assert!(clusters[0].is_some());
    assert_eq!(clusters[0], clusters[1], "two prints of one letter");
    assert!(clusters[2].is_some() && clusters[2] != clusters[0]);
    assert_eq!(clusters[3], None, "grounded in nothing");
}
