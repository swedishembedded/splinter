// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements frozen held-out examinations of what a model
// learned from a person's writing, for its clients. If your team needs
// expertise in building an exam that cannot drift between the models it
// compares, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Spec: the exam is written from the reserved text alone, within its
//! budget and spread over the families, and frozen: it loads again as it was
//! written, and a manifest that was changed is refused.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use common::{scratch_context, Scripted};
use serde_json::json;
use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_core::model_ref::ModelRef;
use splinter_pipelines::exam_set::{create, ExamSet, ExamSetRequest, Role};
use splinter_pipelines::reserve::{reserve, ReserveRequest};
use splinter_pipelines::sources::{add, SourceTarget};

fn words(seed: &str, n: usize) -> String {
    (0..n)
        .map(|i| format!("{seed}w{i}"))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn the_exam_is_written_from_the_reserved_text_within_its_budget_and_frozen() {
    let counter = Arc::new(Mutex::new(0usize));
    let seen = counter.clone();
    // Two tasks from every text it is shown, each quoting what it shows.
    let generator = Scripted::new(move |prompt| {
        if !prompt.contains("You write training tasks") {
            return String::new();
        }
        let end = prompt.find("w0 l").expect("a letter is shown");
        let digits: String = prompt[..end]
            .chars()
            .rev()
            .take_while(char::is_ascii_digit)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        let key = format!("l{digits}w0");
        let text = words(key.trim_end_matches("w0"), 40);
        let tasks: Vec<_> = (0..2)
            .map(|_| {
                let n = {
                    let mut n = seen.lock().unwrap();
                    *n += 1;
                    *n
                };
                json!({
                    "instruction": format!("I wonder, question {n}: what does Letter {digits} say?"),
                    "subject": format!("Letter {digits}"),
                    "reference": text.split(' ').take(3).collect::<Vec<_>>().join(" "),
                    "evidence": [{ "section": 1, "quote": text }]
                })
            })
            .collect();
        json!({ "tasks": tasks }).to_string()
    });
    let (scratch, ctx) = scratch_context("exam-set", Scripted::new(|_| String::new()), false);
    let generator_ref: ModelRef = "local:test/generator".parse().unwrap();
    ctx.add_model(
        generator_ref.clone(),
        Model::new(Arc::new(generator), "scripted/generator"),
    );
    let dir = scratch.0.join("letters");
    std::fs::create_dir_all(&dir).unwrap();
    for n in 0..40 {
        std::fs::write(
            dir.join(format!("letter-{n:02}.md")),
            format!(
                "# Letter {n}\n\n## Body\n\n{}\n",
                words(&format!("l{n}"), 200)
            ),
        )
        .unwrap();
    }
    let id = add(&ctx, &SourceTarget::Path { path: dir })
        .unwrap()
        .source
        .id;
    let reservation = reserve(
        &ctx,
        &ReserveRequest {
            sources: &[id],
            families: 6,
            dev_families: 0,
            seed: 1,
            touched_by: &[],
        },
    )
    .unwrap();
    let exam = create(
        &ctx,
        &ExamSetRequest {
            sources: &reservation.exam,
            families: &reservation.families,
            role: Role::Final,
            kinds: &["recall".to_string()],
            generator: &generator_ref,
            goal: None,
            author: None,
            tasks_per_family: 1,
            cancel: CancelToken::new(),
        },
    )
    .unwrap();
    // Twelve tasks were admitted from six families; one a family is kept.
    assert_eq!(exam.tasks.len(), 6);
    let reserved: BTreeSet<_> = reservation.families.iter().map(|f| &f.family).collect();
    assert!(exam.tasks.iter().all(|t| reserved.contains(&t.family)));
    assert_eq!(exam.families_examined(), 6);
    // Every task is of reserved text alone.
    let store = ctx.tasks();
    let parts: BTreeSet<_> = reservation
        .exam
        .iter()
        .flat_map(|s| ctx.sources().get_source(s).unwrap().parts)
        .map(|p| p.content)
        .collect();
    for t in &exam.tasks {
        let task = store.get(&t.task).unwrap();
        assert!(task
            .evidence
            .iter()
            .all(|span| parts.contains(&span.source)));
    }

    // Frozen: it loads again as written, by id and by file, and a changed
    // manifest is refused.
    let loaded = ExamSet::load(&ctx, &exam.id).unwrap();
    assert_eq!(loaded, exam);
    let file = exam.file(&ctx);
    assert_eq!(ExamSet::load(&ctx, file.to_str().unwrap()).unwrap(), exam);
    // And it carries itself: loaded from its file in another state root, its
    // tasks and the text they were written from are installed there.
    let (_other_scratch, other) =
        scratch_context("exam-set-other", Scripted::new(|_| String::new()), false);
    let carried = ExamSet::load(&other, file.to_str().unwrap()).unwrap();
    assert_eq!(carried.tasks, exam.tasks);
    for t in &exam.tasks {
        assert_eq!(
            other.tasks().get(&t.task).unwrap(),
            ctx.tasks().get(&t.task).unwrap()
        );
    }
    for id in &reservation.exam {
        assert!(other.sources().contains(id).unwrap());
    }
    let text = std::fs::read_to_string(&file).unwrap();
    std::fs::write(
        &file,
        text.replace("\"tasks_per_family\": 1", "\"tasks_per_family\": 10"),
    )
    .unwrap();
    let refused = ExamSet::load(&ctx, &exam.id).unwrap_err().to_string();
    assert!(refused.contains("frozen"), "{refused}");
}
