// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements retrieval over a person's writing that
// finds the passages a task or a principle bears on, for its clients. If
// your team needs expertise in hybrid lexical and semantic search over a
// large corpus, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Retrieval put to work on a task: the passages that bear on its
//! instruction, shown before it, and whether they include the passage the
//! task was written from.

use splinter_agent::solve::open_book_prompt;
use splinter_core::experience::Task;
use splinter_core::source::SourceId;
use splinter_knowledge::retrieve::{passages, Embedder, Library, Passage, Reranker};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

use crate::{index, sources};

/// The stored sources `ids` name and the library of their passages: read
/// from the kept index when there is one, else embedded now and kept
/// ([`index::library`]). Refused when they hold no passage.
pub fn library_of(
    ctx: &Context,
    ids: &[String],
    embedder: &dyn Embedder,
) -> Result<(Vec<SourceId>, Library), OrchestratorError> {
    let resolved = ids
        .iter()
        .map(|id| sources::resolve(ctx, id))
        .collect::<Result<Vec<_>, _>>()?;
    let found = passages(&ctx.sources(), &resolved)
        .map_err(|e| OrchestratorError::Refused(e.to_string()))?;
    if found.is_empty() {
        return Err(OrchestratorError::Refused(
            "the sources hold no passage to retrieve".into(),
        ));
    }
    Ok((resolved, index::library(ctx, found, embedder)?))
}

/// How many candidates search finds for a reader, for each passage shown:
/// enough that a passage meaning ranked low can be moved up.
const CANDIDATES_PER_PASSAGE: usize = 6;

/// How many candidates a reader of a search that shows `passages` reads.
#[must_use]
pub fn candidates_for(passages: usize) -> usize {
    passages.max(1) * CANDIDATES_PER_PASSAGE
}

/// A reader that reorders what search found.
#[derive(Clone, Copy)]
pub struct Rerank<'a> {
    /// Judges whether a passage bears on the question.
    pub reranker: &'a dyn Reranker,
    /// How many passages search finds for it to read.
    pub candidates: usize,
}

/// A library to retrieve from, and how.
pub struct Retrieval<'a> {
    /// The passages of the sources, searched.
    pub library: &'a Library,
    /// The embedder the library was made with.
    pub embedder: &'a dyn Embedder,
    /// How many passages to show with a question.
    pub passages: usize,
    /// A reader of the candidates, when there is one.
    pub rerank: Option<Rerank<'a>>,
}

/// What retrieval put in front of a task.
pub struct Retrieved<'a> {
    /// The passages found, nearest in meaning first.
    pub passages: Vec<&'a Passage>,
}

impl<'a> Retrieval<'a> {
    /// The passages that bear on `question`.
    pub fn find(&self, question: &str) -> Result<Retrieved<'a>, OrchestratorError> {
        let library: &'a Library = self.library;
        let shown = self.passages.max(1);
        let found = match &self.rerank {
            None => library
                .find(question, self.embedder, shown)
                .map_err(|e| e.to_string()),
            Some(rerank) => library
                .find_reranked(
                    question,
                    self.embedder,
                    rerank.reranker,
                    shown,
                    rerank.candidates,
                )
                .map_err(|e| e.to_string()),
        };
        let passages = found.map_err(|e| OrchestratorError::Refused(format!("retrieval: {e}")))?;
        Ok(Retrieved { passages })
    }
}

impl Retrieved<'_> {
    /// `question` with the passages shown before it, each under its part and
    /// section.
    #[must_use]
    pub fn prompt(&self, question: &str) -> String {
        let material: Vec<String> = self
            .passages
            .iter()
            .map(|p| format!("--- {}, section {} ---\n{}", p.part, p.section + 1, p.text))
            .collect();
        open_book_prompt(question, &material)
    }

    /// Whether one of the passages overlaps a span `task` is grounded in:
    /// the retriever found where the task was written from.
    #[must_use]
    pub fn finds_the_evidence_of(&self, task: &Task) -> bool {
        self.passages.iter().any(|p| {
            p.content.as_ref().is_some_and(|content| {
                task.evidence.iter().any(|span| {
                    &span.source == content
                        && (p.range.start as u64) < span.end
                        && span.start < p.range.end as u64
                })
            })
        })
    }
}
