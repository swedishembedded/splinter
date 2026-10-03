// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Spec: the views that read sources and tasks rather than solves.
//!
//! * retrieval: a verified experience's instruction paired with the source
//!   span it is grounded in, resolved through the source store; the other
//!   spans of that content the corpus knows are its negatives.
//! * denoise: a denoise task's corrupted passage and its original, with no
//!   solve needed, once per task.
//! * cpt: the raw text of source parts, once per content.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::*;
use splinter_core::annotation::{Outcome, Strength};
use splinter_core::clock::FixedClock;
use splinter_core::experience::{Digest, PrivilegedKind, Span};
use splinter_core::kinds::DENOISE;
use splinter_core::source::{CapturedSource, Origin, PartContent, PartRef, SourceId};
use splinter_record::sources::SourceStore;
use splinter_record::StateRoot;
use splinter_views::{Corpus, Cpt, DenoiseView, Exclusion, Objective, RecordBody, Retrieval, View};

/// A conversation as (role, content, supervised) triples.
type Turns = Vec<(String, String, bool)>;

const TEXT: &str = "Experiences are immutable. Annotations are appended. Views project them.";

fn capture(sources: &SourceStore, path: &str, parts: &[(&str, &[u8])]) -> SourceId {
    let captured = CapturedSource::new(
        Origin::Document { path: path.into() },
        parts
            .iter()
            .map(|(name, bytes)| PartContent {
                name: (*name).into(),
                media_type: "text/plain".into(),
                bytes: bytes.to_vec(),
            })
            .collect(),
        &FixedClock::new("2026-09-30T00:00:00.000Z"),
    )
    .unwrap();
    sources.put_source(&captured).unwrap()
}

fn span_of(source: &SourceId, needle: &str) -> Span {
    let start = TEXT.find(needle).unwrap() as u64;
    Span::in_part(
        PartRef {
            source: source.clone(),
            name: "notes.txt".into(),
        },
        Digest::of(TEXT.as_bytes()),
        start,
        start + needle.len() as u64,
    )
    .unwrap()
}

#[test]
fn retrieval_pairs_the_instruction_with_the_span_it_is_grounded_in() {
    let scratch = Scratch::new("retrieval");
    let sources = SourceStore::new(&splinter_record::workspace::Workspace::at(&StateRoot::new(
        &scratch.0,
    )));
    let source = capture(&sources, "/notes.txt", &[("notes.txt", TEXT.as_bytes())]);
    let used = "Annotations are appended.";
    let other = "Views project them.";
    let grounded = task_with_evidence(
        "recall",
        "What happens to annotations?",
        Vec::new(),
        vec![span_of(&source, used)],
    );
    let elsewhere = task_with_evidence(
        "recall",
        "What do views do?",
        Vec::new(),
        vec![span_of(&source, other)],
    );
    let pass = answered(&grounded, "They are appended.", "2026-09-30T01:00:00.000Z");
    let fail = answered(&elsewhere, "Nothing.", "2026-09-30T02:00:00.000Z");
    let mut corpus = Corpus::new();
    corpus
        .insert(
            pass.clone(),
            vec![verdict(&pass, Outcome::Pass, Strength::Formal)],
        )
        .unwrap();
    corpus
        .insert(
            fail.clone(),
            vec![verdict(&fail, Outcome::Fail, Strength::Formal)],
        )
        .unwrap();

    let view = Retrieval::new(Strength::Formal, &sources);
    assert_eq!(view.objective(), Objective::Contrastive);
    let projection = view.project(&corpus).unwrap();
    let [record] = &projection.records[..] else {
        panic!("one record, got {:?}", projection.records)
    };
    let RecordBody::Contrastive {
        query,
        positive,
        negatives,
    } = &record.body
    else {
        panic!("a contrastive record, got {record:?}")
    };
    assert_eq!(query, "What happens to annotations?");
    assert_eq!(positive, used);
    assert_eq!(negatives, &[other.to_string()]);
    assert_eq!(record.metadata.experiences, [id(&pass)]);
    assert_eq!(record.metadata.sources, [Digest::of(TEXT.as_bytes())]);
    assert_eq!(projection.count(Exclusion::Failed), 1);
}

#[test]
fn denoise_restores_the_passage_of_each_task_once_without_a_solve() {
    const ORIGINAL: &str = "the quick brown fox jumps";
    let instruction = "Restore the original passage: quick the brown jumps";
    let denoise = task(
        DENOISE,
        instruction,
        vec![privileged(PrivilegedKind::Reference, ORIGINAL)],
    );
    let unsolved = task(
        DENOISE,
        "Restore the original passage: dog lazy the",
        vec![privileged(PrivilegedKind::Reference, "the lazy dog")],
    );
    let other = task(
        "recall",
        "What jumps?",
        vec![privileged(PrivilegedKind::Reference, "fox")],
    );
    let solved = answered(&denoise, "wrong", "2026-09-30T01:00:00.000Z");
    let mut corpus = Corpus::new();
    corpus.add_task(denoise.clone());
    corpus.add_task(unsolved.clone());
    corpus.add_task(other);
    corpus
        .insert(
            solved.clone(),
            vec![verdict(&solved, Outcome::Fail, Strength::Formal)],
        )
        .unwrap();

    let view = DenoiseView::new();
    assert_eq!(view.objective(), Objective::Sft);
    let projection = view.project(&corpus).unwrap();
    let restored: Vec<(Option<Digest>, Turns)> = projection
        .records
        .iter()
        .map(|r| match &r.body {
            RecordBody::Chat { messages } => (
                r.metadata.task.clone(),
                after_system(messages)
                    .iter()
                    .map(|m| (m.role.clone(), m.content.clone(), m.train))
                    .collect(),
            ),
            other => panic!("a chat record, got {other:?}"),
        })
        .collect();
    assert_eq!(
        restored,
        [
            (
                Some(denoise.task.id.clone()),
                vec![
                    ("user".into(), instruction.into(), false),
                    ("assistant".into(), ORIGINAL.into(), true)
                ]
            ),
            (
                Some(unsolved.task.id.clone()),
                vec![
                    ("user".into(), unsolved.instruction.clone(), false),
                    ("assistant".into(), "the lazy dog".into(), true)
                ]
            ),
        ],
        "the failed solve does not matter; the task is its own supervision"
    );
    assert_eq!(projection.count(Exclusion::Duplicate), 1);
}

#[test]
fn cpt_yields_each_texts_content_once() {
    let scratch = Scratch::new("cpt");
    let sources = SourceStore::new(&splinter_record::workspace::Workspace::at(&StateRoot::new(
        &scratch.0,
    )));
    let first = capture(
        &sources,
        "/a",
        &[("a.txt", b"alpha text"), ("b.bin", &[0xff, 0xfe, 0x00])],
    );
    let second = capture(
        &sources,
        "/b",
        &[("c.txt", b"alpha text"), ("d.txt", b"beta text")],
    );
    let mut corpus = Corpus::new();
    corpus.add_source(first);
    corpus.add_source(second);

    let view = Cpt::new(&sources);
    assert_eq!(view.objective(), Objective::Cpt);
    let projection = view.project(&corpus).unwrap();
    let texts: Vec<&str> = projection
        .records
        .iter()
        .map(|r| match &r.body {
            RecordBody::Text { text } => text.as_str(),
            other => panic!("a text record, got {other:?}"),
        })
        .collect();
    assert_eq!(texts, ["alpha text", "beta text"]);
    assert_eq!(
        projection.records[0].metadata.sources,
        [Digest::of(b"alpha text")]
    );
    assert_eq!(projection.count(Exclusion::Duplicate), 1);
    assert_eq!(projection.count(Exclusion::NotText), 1);
    assert_eq!(projection.strip, None, "no student input to strip");
}
