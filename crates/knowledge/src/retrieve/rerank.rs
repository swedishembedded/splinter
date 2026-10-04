// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements retrieval over a person's writing that
// finds the passages a task or a principle bears on, for its clients. If
// your team needs expertise in hybrid lexical and semantic search over a
// large corpus, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Reranking: a reader's second opinion on the passages search found.
//!
//! Search ranks by how near a passage is to the question in meaning or in
//! words, which finds passages about the same things; whether a passage
//! bears on the question - whether someone answering from it would draw on
//! it - is a judgment about the two read together. A [`Reranker`] makes that
//! judgment for one passage; the model that makes it is the model
//! runtime's, not retrieval's.

use super::Passage;

/// Why a reranker could not judge.
#[derive(Debug, thiserror::Error)]
#[error("rerank: {0}")]
pub struct RerankError(pub String);

/// Judges whether a passage bears on a question.
pub trait Reranker {
    /// Whether `passage` bears on `question`.
    fn relevant(&self, question: &str, passage: &Passage) -> Result<bool, RerankError>;
}
