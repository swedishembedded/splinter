// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent solvers whose every run is replayable
// evidence, for its clients. If your team needs expertise in agent
// environments or learning from agent experience, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The most tokens one reply may take, applied where the model is reached.
//!
//! A run's output budget ([`crate::solve::SolveOptions::max_output_tokens`])
//! stops a run once it has spent that much; it does not tell the model how
//! long a single reply may be. A request that names no limit gets its
//! provider's own default, which for a local model is a few hundred tokens and
//! cuts a reply that is a whole batch of tasks off mid-sentence. So a call
//! with a budget wraps its model: every request that names no limit is sent
//! with the budget as its limit.

use std::sync::Arc;

use sven_sdk::model::{CompletionRequest, ModelProvider, ResponseStream};

/// `inner`, sent every unlimited request with a limit of `tokens`.
pub(crate) struct OutputCap {
    inner: Arc<dyn ModelProvider>,
    tokens: u32,
}

impl OutputCap {
    /// `inner`, limited to `tokens` a reply where a request names none.
    pub(crate) fn wrap(inner: Arc<dyn ModelProvider>, tokens: u64) -> Arc<dyn ModelProvider> {
        Arc::new(Self {
            inner,
            tokens: u32::try_from(tokens).unwrap_or(u32::MAX),
        })
    }
}

#[async_trait::async_trait]
impl ModelProvider for OutputCap {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    async fn complete(&self, mut request: CompletionRequest) -> anyhow::Result<ResponseStream> {
        request
            .max_output_tokens_override
            .get_or_insert(self.tokens);
        self.inner.complete(request).await
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
