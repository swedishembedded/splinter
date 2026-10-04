// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements question answering grounded in the passages
// of a body of writing that bear on the question, for its clients. If your
// team needs expertise in retrieval-grounded model answers, you can procure
// our services by sending an email to info@swedishembedded.com.

//! Spec: a question asked with retrieval is put to the model with the
//! passages of the named sources that bear on it - the few it needs, not the
//! whole text - each under the part it is from, and the answer records the
//! sources it was grounded in. A source with nothing to retrieve is refused.

// Helpers outside a #[test] fn unwrap too: a panic is the failure report.
#![allow(clippy::unwrap_used)]

mod common;

use common::{scratch_context, Scripted};
use splinter_core::model_ref::ModelRef;
use splinter_knowledge::retrieve::{EmbedError, Embedder};
use splinter_orchestrator::OrchestratorError;
use splinter_pipelines::ask::ask_retrieving;
use splinter_pipelines::lineage::{lineage, Direction, LineageRequest, Relation};
use splinter_pipelines::sources::{self, SourceTarget};

const LETTERS: &str = "# Letters\n\n## To Carr\n\nEducation of the people is the surest foundation of liberty, and a nation that wishes to be free must see that its youth are taught to reason.\n\n## To Jay\n\nThe tobacco shipped to Havre by the brig Eliza was sold at a poor price, and the merchants complain of the duties laid upon the hogsheads.\n";

/// An embedder over two concepts: teaching, and trade.
struct Concepts;

impl Embedder for Concepts {
    fn name(&self) -> String {
        "concepts".into()
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        Ok(texts
            .iter()
            .map(|t| {
                let has = |words: &[&str]| words.iter().filter(|w| t.contains(*w)).count() as f32;
                vec![
                    has(&["education", "taught", "reason", "teach", "learn"]),
                    has(&["tobacco", "merchants", "duties", "price", "sold"]),
                    0.1,
                ]
            })
            .collect())
    }
}

#[test]
fn the_model_is_shown_the_passage_that_bears_on_the_question_and_not_the_rest() {
    // The model answers with the prompt it was sent, so the answer shows it.
    let (scratch, ctx) = scratch_context("ask-retrieving", Scripted::new(str::to_string), false);
    let path = scratch.0.join("letters.md");
    std::fs::write(&path, LETTERS).unwrap();
    let source = sources::add(
        &ctx,
        &SourceTarget::from_learn_arg(&path.display().to_string()).unwrap(),
    )
    .unwrap()
    .source
    .id;
    let answer = ask_retrieving(
        &ctx,
        "How should a young person learn?",
        &[source.to_string()],
        1,
        &Concepts,
        &ModelRef::policy_default(),
    )
    .unwrap();
    assert!(answer.answer.contains("surest foundation of liberty"));
    assert!(
        !answer.answer.contains("brig Eliza"),
        "only what bears on the question: {}",
        answer.answer
    );
    assert!(answer.answer.contains("How should a young person learn?"));
    assert_eq!(answer.retrieved_from, std::slice::from_ref(&source));
    // What the model was shown is on the answer and in its record, so a bad
    // answer can be traced to what was retrieved.
    assert_eq!(answer.shown.len(), 1);
    assert_eq!(answer.shown[0].source, source);
    assert!(
        answer.shown[0].part.contains("letters"),
        "{:?}",
        answer.shown
    );
    assert!(answer.shown[0].excerpt.contains("Education of the people"));
    assert_eq!(ctx.answers().get(&answer.id).unwrap().shown, answer.shown);
    assert_eq!(answer.open_book, None);
    // The answer traces back to the source its passages came from.
    let up = lineage(
        &ctx,
        &LineageRequest {
            id: answer.id.to_string(),
            direction: Direction::Up,
            depth: None,
        },
    )
    .unwrap();
    assert!(up.edges.iter().any(|e| e.from == answer.id.as_str()
        && e.relation == Relation::Retrieved
        && e.to == source.as_str()));
}

#[test]
fn a_source_with_no_passage_to_retrieve_is_refused() {
    let (scratch, ctx) =
        scratch_context("ask-retrieving-none", Scripted::new(str::to_string), false);
    let path = scratch.0.join("short.md");
    std::fs::write(&path, "# Note\n\nToo short.\n").unwrap();
    let source = sources::add(
        &ctx,
        &SourceTarget::from_learn_arg(&path.display().to_string()).unwrap(),
    )
    .unwrap()
    .source
    .id;
    let refused = ask_retrieving(
        &ctx,
        "anything",
        &[source.to_string()],
        3,
        &Concepts,
        &ModelRef::policy_default(),
    );
    assert!(matches!(refused, Err(OrchestratorError::Refused(_))));
}

/// Concepts, counting the passages it is asked to embed (queries are free).
struct Counting(std::sync::atomic::AtomicUsize);

impl Embedder for Counting {
    fn name(&self) -> String {
        "counting-concepts".into()
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.0
            .fetch_add(texts.len(), std::sync::atomic::Ordering::SeqCst);
        Concepts.embed(texts)
    }

    fn embed_query(&self, query: &str) -> Result<Vec<f32>, EmbedError> {
        Concepts.embed(&[query]).map(|mut v| v.remove(0))
    }
}

#[test]
fn the_passages_of_a_source_are_embedded_once_and_the_index_is_kept() {
    let (scratch, ctx) = scratch_context("ask-index", Scripted::new(str::to_string), false);
    let path = scratch.0.join("letters.md");
    std::fs::write(&path, LETTERS).unwrap();
    let source = sources::add(
        &ctx,
        &SourceTarget::from_learn_arg(&path.display().to_string()).unwrap(),
    )
    .unwrap()
    .source
    .id
    .to_string();
    let embedder = Counting(0.into());
    let ask = |question: &str| {
        ask_retrieving(
            &ctx,
            question,
            std::slice::from_ref(&source),
            1,
            &embedder,
            &ModelRef::policy_default(),
        )
        .unwrap()
        .answer
    };
    let first = ask("How should a young person learn?");
    let embedded = embedder.0.load(std::sync::atomic::Ordering::SeqCst);
    assert_eq!(embedded, 2, "the two passages of the letters, once");
    let second = ask("How should a young person learn?");
    assert_eq!(
        embedder.0.load(std::sync::atomic::Ordering::SeqCst),
        embedded,
        "the kept index serves the second question"
    );
    assert!(second.contains("surest foundation of liberty") && first.contains("surest foundation"));
    // Another embedder makes other vectors: its index is its own.
    let other = Counting(0.into());
    ask_retrieving(
        &ctx,
        "anything",
        std::slice::from_ref(&source),
        1,
        &Renamed(&other),
        &ModelRef::policy_default(),
    )
    .unwrap();
    assert_eq!(other.0.load(std::sync::atomic::Ordering::SeqCst), 2);
}

/// An embedder under another name.
struct Renamed<'a>(&'a Counting);

impl Embedder for Renamed<'_> {
    fn name(&self) -> String {
        "renamed".into()
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.0.embed(texts)
    }

    fn embed_query(&self, query: &str) -> Result<Vec<f32>, EmbedError> {
        self.0.embed_query(query)
    }
}
