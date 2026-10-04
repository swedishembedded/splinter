// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data in a writer's own words, for
// its clients. If your team needs expertise in training models on a person's
// voice, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: for the tasks whose reference is the writer's own passage, the
//! training answer is that passage word for word, and the task is the message
//! written for it (instruction backtranslation). The answer is the writer's,
//! not a model's, so it is recorded as the source's; a calibrated judge of fit
//! keeps only the pairs where the passage is a natural reply to its message.
//! Tasks of any other kind are left alone.

// Helpers outside a test function unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::sync::Arc;

use common::{scratch_context, Scripted};
use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_core::experience::{Environment, Privileged, PrivilegedKind, Span, Task};
use splinter_core::model_ref::ModelRef;
use splinter_pipelines::author::{author, AuthorRequest};
use splinter_pipelines::sources::{self, SourceTarget};
use splinter_store::tasks::{TaskEntry, TaskSet};

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

fn words(seed: &str, n: usize) -> String {
    (0..n)
        .map(|i| format!("{seed}x{}", letters(i)))
        .collect::<Vec<_>>()
        .join(" ")
}

fn span_of(
    ctx: &splinter_orchestrator::Context,
    dir: &std::path::Path,
    name: &str,
    text: &str,
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

fn task(kind: &str, span: &Span, instruction: &str, passage: &str) -> Task {
    Task::new(
        kind,
        vec![span.clone()],
        Environment::closed_book(),
        instruction,
        vec![Privileged {
            kind: PrivilegedKind::Reference,
            content: passage.into(),
            span: Some(span.clone()),
        }],
    )
    .unwrap()
}

#[test]
fn the_writers_own_passage_is_the_answer_to_the_message_written_for_it() {
    // A judge of fit: the message and the passage must share a key word.
    let fit = Scripted::new(|prompt| {
        let message = prompt
            .split("MESSAGE:\\n")
            .nth(1)
            .and_then(|t| t.split("\\n\\nPASSAGE:").next())
            .unwrap_or_default();
        let passage = prompt.split("PASSAGE:\\n").nth(1).unwrap_or_default();
        let fits = ["study", "expense"]
            .iter()
            .any(|key| message.contains(key) && passage.contains(key));
        if fits {
            "PASS\nit answers the message".into()
        } else {
            "FAIL\nit does not".into()
        }
    });
    let (scratch, ctx) = scratch_context("author", Scripted::new(|_| String::new()), false);
    let judge_ref: ModelRef = "local:test/fit".parse().unwrap();
    ctx.add_model(judge_ref.clone(), Model::new(Arc::new(fit), "scripted/fit"));
    let a = span_of(&ctx, &scratch.0, "a.txt", &words("alpha", 100));
    let b = span_of(&ctx, &scratch.0, "b.txt", &words("omega", 100));
    let passage_a = |i: usize| format!("Keep up your study {i} every morning of your life.");
    let passage_b = |i: usize| format!("Mind each expense {i} and the account will mind you.");
    let mut tasks = Vec::new();
    for i in 0..9 {
        tasks.push(task(
            "advise",
            &a,
            &format!("How do I keep a habit of study (case {i})?"),
            &passage_a(i),
        ));
        tasks.push(task(
            "advise",
            &b,
            &format!("How do I keep my expense in hand (case {i})?"),
            &passage_b(i),
        ));
    }
    // A message the passage written for another does not answer.
    tasks.push(task(
        "advise",
        &a,
        "How do I keep an account of every expense (case X)?",
        &passage_a(99),
    ));
    // A kind whose reference is not the writer's own words is left alone.
    tasks.push(task("recall", &a, "What is the study of case Y?", "short"));
    let store = ctx.tasks();
    let set = store
        .put_set(&TaskSet {
            name: "authored".into(),
            members: tasks
                .iter()
                .map(|t| {
                    store.put(t).unwrap();
                    TaskEntry {
                        task: t.task.id.clone(),
                        generator: None,
                        prompt: None,
                        variant_of: None,
                        subject: None,
                    }
                })
                .collect(),
        })
        .unwrap();

    let authored = author(
        &ctx,
        &AuthorRequest {
            task_set: &set,
            judge: &judge_ref,
        },
        &CancelToken::new(),
    )
    .unwrap();
    assert_eq!(authored.tasks, 19, "the recall task is not one of them");
    assert_eq!((authored.kept, authored.refused), (18, 1), "{authored:#?}");

    let experiences = ctx.experiences();
    let members = experiences
        .get_set(authored.experience_set.as_ref().unwrap())
        .unwrap()
        .members;
    assert_eq!(members.len(), 19);
    let decisions = experiences.decisions(&members).unwrap();
    let kept: Vec<_> = members
        .iter()
        .filter(|id| decisions.get(id).is_some_and(|d| d.passed))
        .collect();
    assert_eq!(kept.len(), 18);
    for id in kept {
        let experience = experiences.get(id).unwrap();
        // The answer is the passage itself, and the source's, not a model's.
        assert_eq!(experience.provenance.solver, "splinter/author");
        let reference = experience
            .privileged
            .iter()
            .find(|p| p.kind == PrivilegedKind::Reference)
            .unwrap();
        assert_eq!(
            experience.final_output.as_deref(),
            Some(reference.content.as_str())
        );
    }
}
