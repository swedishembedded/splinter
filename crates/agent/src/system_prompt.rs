// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent solvers whose prompts match the data
// their models are trained on, for its clients. If your team needs
// expertise in agent training pipelines, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The system prompt a solve runs under, applied where the model is
//! reached.
//!
//! sven seeds every session it runs with its own coding-agent system
//! prompt (its identity, guidelines and the working directory), and its
//! SDK applies an agent's role in place of that prompt to typed method
//! calls only, not to a plain `send`. A solve is a plain `send`, and what the
//! policy answers under must be what its training records show
//! ([`SYSTEM_PROMPT`]). So the solving model is wrapped: every request it
//! receives carries [`SYSTEM_PROMPT`] as its one system turn, first, with
//! whatever system turns sven put in the conversation replaced and no
//! dynamic suffix appended.

use std::sync::Arc;

use splinter_core::prompt::SYSTEM_PROMPT;
use sven_sdk::model::{CompletionRequest, Message, ModelProvider, ResponseStream, Role};

/// `inner`, sent every request under [`SYSTEM_PROMPT`].
pub(crate) struct UnderSystemPrompt(Arc<dyn ModelProvider>);

impl UnderSystemPrompt {
    /// `inner`, wrapped.
    pub(crate) fn wrap(inner: Arc<dyn ModelProvider>) -> Arc<dyn ModelProvider> {
        Arc::new(Self(inner))
    }
}

/// `request` with [`SYSTEM_PROMPT`] as its one system turn, first, and no
/// dynamic suffix.
fn under_system_prompt(mut request: CompletionRequest) -> CompletionRequest {
    request.messages.retain(|m| m.role != Role::System);
    request.messages.insert(0, Message::system(SYSTEM_PROMPT));
    request.system_dynamic_suffix = None;
    request
}

// The catalog lookups the trait derives from `name` and `model_name` reach
// the inner model's entries through the delegated names; the rest of the
// trait's per-model limits are delegated one by one.
#[async_trait::async_trait]
impl ModelProvider for UnderSystemPrompt {
    fn name(&self) -> &str {
        self.0.name()
    }

    fn model_name(&self) -> &str {
        self.0.model_name()
    }

    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<ResponseStream> {
        self.0.complete(under_system_prompt(request)).await
    }

    fn catalog_max_output_tokens(&self) -> Option<u32> {
        self.0.catalog_max_output_tokens()
    }

    fn catalog_context_window(&self) -> Option<u32> {
        self.0.catalog_context_window()
    }

    fn config_context_window(&self) -> Option<u32> {
        self.0.config_context_window()
    }

    fn config_max_output_tokens(&self) -> Option<u32> {
        self.0.config_max_output_tokens()
    }

    async fn probe_context_window(&self) -> Option<u32> {
        self.0.probe_context_window().await
    }

    fn supports_images(&self) -> bool {
        self.0.supports_images()
    }

    fn supports_audio(&self) -> bool {
        self.0.supports_audio()
    }
}
