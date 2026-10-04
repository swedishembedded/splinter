// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements retrieval over a person's writing that
// finds the passages a task or a principle bears on, for its clients. If
// your team needs expertise in hybrid lexical and semantic search over a
// large corpus, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Text embeddings from a local Qwen3-Embedding checkpoint, on brain.
//!
//! A passage is embedded as it is; a query carries an instruction in front of
//! it, which is how the model is meant to be used for asymmetric retrieval.
//! Both are truncated at [`MAX_TOKENS`] rather than refused, so one long
//! section does not stop an index being built.

use brain::{EmbeddingOptions, EmbeddingPipeline};

use crate::error::PolicyError;

/// The embedding model Splinter uses where none is named.
pub const DEFAULT_MODEL: &str = "Qwen/Qwen3-Embedding-0.6B";

/// The longest text embedded, in tokens; the rest is not read.
pub const MAX_TOKENS: u32 = 512;

/// What a query is asked to find.
const QUERY_INSTRUCTION: &str = "Given a situation or a question, find the passages of a \
person's letters that bear on it";

/// A loaded embedding model.
pub struct Embeddings {
    pipeline: EmbeddingPipeline,
}

impl Embeddings {
    /// The embedding model `model` names: a name in brain's model store
    /// (`Qwen/Qwen3-Embedding-0.6B`), or a brain-format `.safetensors` file.
    pub fn load(model: &str) -> Result<Self, PolicyError> {
        let pipeline = EmbeddingPipeline::builder(model)
            .capacity(MAX_TOKENS + 64)
            .load()
            .map_err(|e| PolicyError::Load {
                path: model.into(),
                reason: e.to_string(),
            })?;
        Ok(Self { pipeline })
    }

    /// One unit-length vector per passage.
    pub fn passages(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, PolicyError> {
        let options = EmbeddingOptions::new().max_tokens(MAX_TOKENS);
        self.vectors(texts, options)
    }

    /// The vector of `query`, instructed.
    pub fn query(&self, query: &str) -> Result<Vec<f32>, PolicyError> {
        let options = EmbeddingOptions::new()
            .instruction(QUERY_INSTRUCTION)
            .max_tokens(MAX_TOKENS);
        let mut vectors = self.vectors(&[query], options)?;
        vectors.pop().ok_or_else(|| PolicyError::Embedding {
            reason: "no vector came back for the query".into(),
        })
    }

    fn vectors(
        &self,
        texts: &[&str],
        options: EmbeddingOptions,
    ) -> Result<Vec<Vec<f32>>, PolicyError> {
        self.pipeline
            .embed_batch_with(texts, options)
            .map(|made| made.into_iter().map(|e| e.as_slice().to_vec()).collect())
            .map_err(|e| PolicyError::Embedding {
                reason: e.to_string(),
            })
    }
}
