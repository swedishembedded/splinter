// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements retrieval over a person's writing that
// finds the passages a task or a principle bears on, for its clients. If
// your team needs expertise in hybrid lexical and semantic search over a
// large corpus, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The model runtime's embeddings as retrieval's [`Embedder`]: retrieval
//! knows no model, and the model runtime knows no retrieval.

use splinter_knowledge::retrieve::{EmbedError, Embedder};
use splinter_model::embed::{Embeddings, DEFAULT_MODEL};
use splinter_orchestrator::error::OrchestratorError;

/// A loaded embedding model, as an [`Embedder`].
pub struct ModelEmbedder(Embeddings);

impl ModelEmbedder {
    /// The default embedding model ([`DEFAULT_MODEL`]), loaded.
    pub fn load_default() -> Result<Self, OrchestratorError> {
        Embeddings::load(DEFAULT_MODEL)
            .map(Self)
            .map_err(|e| OrchestratorError::Refused(format!("the embedding model: {e}")))
    }
}

impl Embedder for ModelEmbedder {
    fn name(&self) -> String {
        DEFAULT_MODEL.to_string()
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.0
            .passages(texts)
            .map_err(|e| EmbedError::Failed(e.to_string()))
    }

    fn embed_query(&self, query: &str) -> Result<Vec<f32>, EmbedError> {
        self.0
            .query(query)
            .map_err(|e| EmbedError::Failed(e.to_string()))
    }
}
