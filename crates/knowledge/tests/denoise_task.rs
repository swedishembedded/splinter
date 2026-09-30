// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: a denoise task is a deterministic function of its source and seed,
//! grounded in a span of the source, shows the student only the corrupted
//! passage, and keeps the original as the teacher's reference.

use splinter_knowledge::denoise::{Denoise, DenoiseError};
use splinter_store::experience::{Digest, PrivilegedKind};

const SOURCE: &str = "Splinter keeps every experience immutable and content addressed. \
    Graders append annotations beside an experience and never rewrite it. \
    Training sets are projected from experiences through views.";

#[test]
fn the_task_is_deterministic_and_grounded_in_a_span_of_the_source() {
    let digest = Digest::of(SOURCE.as_bytes());
    let task = Denoise::new(7).generate(SOURCE, &digest).unwrap();
    assert_eq!(task, Denoise::new(7).generate(SOURCE, &digest).unwrap());
    assert_eq!(task.task.kind, splinter_lab::denoise::KIND);

    let [span] = &task.evidence[..] else {
        panic!("one evidence span, got {:?}", task.evidence)
    };
    assert_eq!(span.source, digest);
    let original = &SOURCE[span.start as usize..span.end as usize];
    let [reference] = &task.privileged[..] else {
        panic!("one privileged item")
    };
    assert_eq!(reference.kind, PrivilegedKind::Reference);
    assert_eq!(reference.content, original);
    assert_eq!(reference.span.as_ref(), Some(span));
    assert!(
        !task.instruction.contains(original),
        "the student never sees the original: {}",
        task.instruction
    );

    let other_seeds: Vec<_> = (0..8)
        .map(|seed| {
            Denoise::new(seed)
                .generate(SOURCE, &digest)
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
fn a_mismatched_digest_or_a_too_short_source_is_refused() {
    let wrong = Digest::of(b"another text");
    assert!(matches!(
        Denoise::new(1).generate(SOURCE, &wrong),
        Err(DenoiseError::Digest { .. })
    ));
    let short = "two words";
    assert!(matches!(
        Denoise::new(1).generate(short, &Digest::of(short.as_bytes())),
        Err(DenoiseError::TooShort { .. })
    ));
}
