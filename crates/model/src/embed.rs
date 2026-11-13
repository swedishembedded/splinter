// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements retrieval over a person's writing that
// finds the passages a task or a principle bears on, for its clients. If
// your team needs expertise in hybrid lexical and semantic search over a
// large corpus, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Text embeddings from a Qwen3-Embedding model a brain server serves
//! (`POST /v1/embeddings`).
//!
//! A passage is embedded as it is; a query carries an instruction in front of
//! it, which is how the model is meant to be used for asymmetric retrieval.
//! Both are truncated by the server at [`MAX_TOKENS`] rather than refused, so
//! one long section does not stop an index being built.

use std::path::Path;
use std::time::Duration;

use serde_json::{json, Value};

use crate::endpoint::{post_json, BrainEndpoints};
use crate::error::PolicyError;
use crate::local::LocalWeights;

/// The embedding model Splinter uses where none is named.
pub const DEFAULT_MODEL: &str = "Qwen/Qwen3-Embedding-0.6B";

/// The longest text embedded, in tokens; the server reads no more.
pub const MAX_TOKENS: u32 = 512;

/// What a query is asked to find.
const QUERY_INSTRUCTION: &str = "Given a situation or a question, find the passages of a \
person's letters that bear on it";

/// How long one embedding request may take, queueing included.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

/// An embedding model a brain server serves.
pub struct Embeddings {
    url: String,
    key: String,
    model: String,
}

impl Embeddings {
    /// The embedding model `model` names in brain's model store
    /// (`Qwen/Qwen3-Embedding-0.6B`), as the endpoints serve it.
    ///
    /// # Errors
    /// An endpoint does not serve it.
    pub fn request(endpoints: &BrainEndpoints, model: &str) -> Result<Self, PolicyError> {
        let weights = LocalWeights {
            base: Path::new(model).to_path_buf(),
            adapter: None,
            context_tokens: None,
        };
        let ids = endpoints.locate(&weights)?;
        Ok(Self {
            url: endpoints.urls()[0].clone(),
            key: endpoints.key().to_string(),
            model: ids[0].clone(),
        })
    }

    /// One unit-length vector per passage.
    pub fn passages(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, PolicyError> {
        self.vectors(json!({"model": self.model, "input": texts, "max_tokens": MAX_TOKENS}))
    }

    /// The vector of `query`, instructed.
    pub fn query(&self, query: &str) -> Result<Vec<f32>, PolicyError> {
        let body = json!({
            "model": self.model,
            "input": [query],
            "instruction": QUERY_INSTRUCTION,
            "max_tokens": MAX_TOKENS,
        });
        self.vectors(body)?
            .pop()
            .ok_or_else(|| PolicyError::Embedding {
                reason: "no vector came back for the query".into(),
            })
    }

    fn vectors(&self, body: Value) -> Result<Vec<Vec<f32>>, PolicyError> {
        let failed = |reason: String| PolicyError::Embedding { reason };
        let reply = post_json(
            &format!("{}/embeddings", self.url),
            &self.key,
            &body,
            REQUEST_TIMEOUT,
        )
        .map_err(failed)?;
        let data = reply["data"]
            .as_array()
            .ok_or_else(|| failed("the reply has no \"data\" array".into()))?;
        let expected = body["input"].as_array().map_or(0, Vec::len);
        if data.len() != expected {
            return Err(failed(format!(
                "{} vectors came back for {expected} texts",
                data.len()
            )));
        }
        let mut ordered: Vec<(u64, Vec<f32>)> = Vec::with_capacity(data.len());
        for item in data {
            let index = item["index"].as_u64().unwrap_or(ordered.len() as u64);
            let vector = item["embedding"]
                .as_array()
                .ok_or_else(|| failed("an item has no \"embedding\" array".into()))?
                .iter()
                .map(|v| {
                    // Embeddings are f32 on the wire; narrowing is the format.
                    #[allow(clippy::cast_possible_truncation)]
                    v.as_f64().map(|x| x as f32)
                })
                .collect::<Option<Vec<f32>>>()
                .ok_or_else(|| failed("an embedding holds a value that is not a number".into()))?;
            ordered.push((index, vector));
        }
        ordered.sort_by_key(|(index, _)| *index);
        Ok(ordered.into_iter().map(|(_, vector)| vector).collect())
    }
}
