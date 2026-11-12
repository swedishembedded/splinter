// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements retrieval that finds the material that
// means what a question asks when it does not use the question's words, for
// its clients. If your team needs expertise in semantic and lexical search
// over technical corpora you can procure our services by sending an email
// to info@swedishembedded.com.

//! An embedding model served by brain, asked over its OpenAI-compatible
//! `/v1/embeddings`, and a wrapper that remembers what failed.

use std::cell::RefCell;
use std::rc::Rc;

use splinter_sdk::knowledge::retrieve::{EmbedError, Embedder};
use splinter_sdk::Context;

/// The instruction the embedding model is given in front of a query.
const QUERY_INSTRUCTION: &str =
    "Instruct: Given a question about how something works or behaves, find the passages that answer it\nQuery: ";

/// Embeds through a served model.
pub struct Served {
    ctx: Context,
    client: reqwest::Client,
    url: String,
    key: String,
    model: String,
}

impl Served {
    /// The model `model` at `base_url` (ending in `/v1`), asked with `key`.
    pub fn new(ctx: Context, base_url: &str, key: &str, model: &str) -> Self {
        // The workspace builds reqwest without a TLS provider; one is installed
        // once for the process (a second install is ignored).
        let _ = rustls::crypto::ring::default_provider().install_default();
        Self {
            ctx,
            client: reqwest::Client::new(),
            url: format!("{}/embeddings", base_url.trim_end_matches('/')),
            key: key.to_string(),
            model: model.to_string(),
        }
    }

    fn call(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let body = serde_json::json!({"model": self.model, "input": texts}).to_string();
        let request = self
            .client
            .post(&self.url)
            .header("authorization", format!("Bearer {}", self.key))
            .header("content-type", "application/json")
            .body(body);
        let text = self
            .ctx
            .block_on(async {
                let response = request.send().await.map_err(|e| e.to_string())?;
                let status = response.status();
                let text = response.text().await.map_err(|e| e.to_string())?;
                if status.is_success() {
                    Ok(text)
                } else {
                    Err(format!(
                        "{status}: {}",
                        text.chars().take(200).collect::<String>()
                    ))
                }
            })
            .map_err(EmbedError::Failed)?;
        let parsed: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| EmbedError::Failed(e.to_string()))?;
        let vectors: Vec<Vec<f32>> = parsed["data"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|item| {
                item["embedding"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|x| x.as_f64().map(|x| x as f32))
                    .collect()
            })
            .collect();
        if vectors.len() != texts.len() {
            return Err(EmbedError::Malformed {
                wanted: texts.len(),
                got: vectors.len(),
            });
        }
        Ok(vectors)
    }
}

impl Embedder for Served {
    fn name(&self) -> String {
        format!("served:{}", self.model)
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.call(&texts.iter().map(|t| (*t).to_string()).collect::<Vec<_>>())
    }

    fn embed_query(&self, query: &str) -> Result<Vec<f32>, EmbedError> {
        let mut v = self.call(&[format!("{QUERY_INSTRUCTION}{query}")])?;
        v.pop().ok_or(EmbedError::Malformed { wanted: 1, got: 0 })
    }
}

/// Passes calls through and keeps every failure: a run with one is void.
pub struct Tracked {
    inner: Rc<dyn Embedder>,
    failures: Rc<RefCell<Vec<String>>>,
}

impl Tracked {
    /// Wraps `inner`, recording failures in `failures`.
    pub fn new(inner: Rc<dyn Embedder>, failures: Rc<RefCell<Vec<String>>>) -> Self {
        Self { inner, failures }
    }

    fn note<T>(&self, result: Result<T, EmbedError>) -> Result<T, EmbedError> {
        if let Err(e) = &result {
            self.failures.borrow_mut().push(e.to_string());
        }
        result
    }
}

impl Embedder for Tracked {
    fn name(&self) -> String {
        self.inner.name()
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        self.note(self.inner.embed(texts))
    }

    fn embed_query(&self, query: &str) -> Result<Vec<f32>, EmbedError> {
        self.note(self.inner.embed_query(query))
    }
}
