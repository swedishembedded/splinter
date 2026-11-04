// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: a denoise task is a deterministic function of its source part and
//! seed, grounded in a span that resolves through the source store to the
//! original passage, shows the student only the corrupted passage, and
//! keeps the original as the teacher's reference.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use splinter_core::clock::FixedClock;
use splinter_core::experience::PrivilegedKind;
use splinter_core::source::{CapturedSource, Origin, PartContent};
use splinter_knowledge::denoise::{Denoise, DenoiseError};
use splinter_store::sources::SourceStore;
use splinter_store::StateRoot;

const PASSAGE: &str = "Splinter keeps every experience immutable and content addressed. \
    Graders append annotations beside an experience and never rewrite it. \
    Training sets are projected from experiences through views.";

fn source(text: &str) -> CapturedSource {
    CapturedSource::new(
        Origin::Document {
            path: "/notes/design.md".into(),
        },
        vec![PartContent {
            name: "design.md".into(),
            media_type: "text/markdown".into(),
            bytes: text.as_bytes().to_vec(),
        }],
        &FixedClock::new("2026-09-30T08:00:00.000Z"),
    )
    .unwrap()
}

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn the_task_is_deterministic_and_its_evidence_resolves_through_the_store() {
    let scratch =
        Scratch(std::env::temp_dir().join(format!("splinter-denoise-task-{}", std::process::id())));
    let store = SourceStore::new(&splinter_store::workspace::Workspace::at(&StateRoot::new(
        &scratch.0,
    )));
    let captured = source(PASSAGE);
    let id = store.put_source(&captured).unwrap();
    let stored = store.get_source(&id).unwrap();
    let bytes = store.read_part(&id, "design.md").unwrap();

    let task = Denoise::new(7)
        .generate(&stored, "design.md", &bytes)
        .unwrap();
    assert_eq!(
        task,
        Denoise::new(7)
            .generate(&stored, "design.md", &bytes)
            .unwrap()
    );
    assert_eq!(task.task.kind, splinter_core::kinds::DENOISE);

    let [span] = &task.evidence[..] else {
        panic!("one evidence span, got {:?}", task.evidence)
    };
    let part = span.part.as_ref().unwrap();
    assert_eq!((&part.source, part.name.as_str()), (&id, "design.md"));
    let original = String::from_utf8(store.read_span(span).unwrap()).unwrap();
    assert!(PASSAGE.contains(&original));
    let [reference] = &task.privileged[..] else {
        panic!("one privileged item")
    };
    assert_eq!(reference.kind, PrivilegedKind::Reference);
    assert_eq!(reference.content, original);
    assert_eq!(reference.span.as_ref(), Some(span));
    assert!(
        !task.instruction.contains(&original),
        "the student never sees the original: {}",
        task.instruction
    );

    let other_seeds: Vec<_> = (0..8)
        .map(|seed| {
            Denoise::new(seed)
                .generate(&stored, "design.md", &bytes)
                .unwrap()
                .task
                .id
        })
        .collect();
    assert!(
        other_seeds.iter().any(|id| *id != task.task.id),
        "the seed chooses the task"
    );
}

#[test]
fn mismatched_bytes_an_unknown_part_or_a_too_short_part_are_refused() {
    let captured = source(PASSAGE);
    let stored = captured.source();
    assert!(matches!(
        Denoise::new(1).generate(stored, "design.md", b"another text"),
        Err(DenoiseError::Digest { .. })
    ));
    assert!(matches!(
        Denoise::new(1).generate(stored, "other.md", PASSAGE.as_bytes()),
        Err(DenoiseError::UnknownPart { .. })
    ));
    let short = source("two words");
    assert!(matches!(
        Denoise::new(1).generate(short.source(), "design.md", b"two words"),
        Err(DenoiseError::TooShort { .. })
    ));
}

/// One seed generates the tasks of every part of a run, so the corruption
/// must follow the content too: were it the seed's alone, every task would
/// drop the word at the same position and swap the same pair, and a
/// student could learn the position instead of the text.
#[test]
fn parts_of_different_content_are_corrupted_at_different_positions() {
    let dropped: std::collections::BTreeSet<usize> = (0..16)
        .map(|n| {
            // Twelve distinct words: the window is the whole passage.
            let words: Vec<String> = (0..12).map(|i| format!("w{n}x{i}")).collect();
            let text = words.join(" ");
            let captured = source(&text);
            let task = Denoise::new(0)
                .generate(captured.source(), "design.md", text.as_bytes())
                .unwrap();
            let shown = task.instruction.rsplit("\n\n").next().unwrap().to_string();
            words
                .iter()
                .position(|w| !shown.split(' ').any(|s| s == w))
                .unwrap()
        })
        .collect();
    assert!(dropped.len() > 1, "every task dropped word {dropped:?}");
}
