// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent solvers whose every run is replayable
// evidence, for its clients. If your team needs expertise in agent
// environments or learning from agent experience, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Telling what a busy server did to an attempt from what the agent did.
//!
//! On a server shared with other jobs a request waits in a queue before the
//! first token arrives. [`Timed`] wraps the model an attempt talks to and
//! counts its requests, how long each waited for its first stream item and
//! how long the streams took in all, so an attempt that ran long can be told
//! apart from one that did a lot. Only the agent's behaviour (calls, turns)
//! bounds an attempt; wall time never does.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use futures::StreamExt;
use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseStream};

/// What the requests of one attempt cost in waiting.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stats {
    /// Requests sent.
    pub requests: usize,
    /// Seconds from request to first stream item, summed.
    pub wait_seconds: f64,
    /// The longest such wait.
    pub max_wait_seconds: f64,
    /// Seconds from request to the last stream item, summed.
    pub model_seconds: f64,
}

/// `inner`, counting its requests into a shared [`Stats`].
pub struct Timed {
    inner: Arc<dyn ModelProvider>,
    stats: Arc<Mutex<Stats>>,
}

impl Timed {
    /// `inner` as a provider that records into `stats`.
    #[must_use]
    pub fn wrap(inner: Arc<dyn ModelProvider>, stats: Arc<Mutex<Stats>>) -> Arc<dyn ModelProvider> {
        Arc::new(Self { inner, stats })
    }
}

#[async_trait::async_trait]
impl ModelProvider for Timed {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let started = Instant::now();
        let stream = self.inner.complete(request).await?;
        let stats = Arc::clone(&self.stats);
        if let Ok(mut s) = stats.lock() {
            s.requests += 1;
        }
        let mut previous = 0.0_f64;
        let mut first = true;
        Ok(Box::pin(stream.inspect(move |_| {
            let at = started.elapsed().as_secs_f64();
            if let Ok(mut s) = stats.lock() {
                if first {
                    s.wait_seconds += at;
                    s.max_wait_seconds = s.max_wait_seconds.max(at);
                }
                s.model_seconds += at - previous;
            }
            first = false;
            previous = at;
        })))
    }

    fn catalog_max_output_tokens(&self) -> Option<u32> {
        self.inner.catalog_max_output_tokens()
    }

    fn catalog_context_window(&self) -> Option<u32> {
        self.inner.catalog_context_window()
    }

    fn config_context_window(&self) -> Option<u32> {
        self.inner.config_context_window()
    }

    fn config_max_output_tokens(&self) -> Option<u32> {
        self.inner.config_max_output_tokens()
    }

    async fn probe_context_window(&self) -> Option<u32> {
        self.inner.probe_context_window().await
    }

    fn supports_images(&self) -> bool {
        self.inner.supports_images()
    }

    fn supports_audio(&self) -> bool {
        self.inner.supports_audio()
    }
}
