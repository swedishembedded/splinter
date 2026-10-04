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
//! ([`SYSTEM_PROMPT`], or the person's it was trained to be). So the solving
//! model is wrapped: every request it receives carries that prompt as its one
//! system turn, first, with
//! whatever system turns sven put in the conversation replaced and no
//! dynamic suffix appended.

use std::sync::Arc;

use sven_sdk::model::{CompletionRequest, Message, ModelProvider, ResponseStream, Role};

/// `inner`, sent every request under [`SYSTEM_PROMPT`].
pub(crate) struct UnderSystemPrompt {
    inner: Arc<dyn ModelProvider>,
    system: String,
}

impl UnderSystemPrompt {
    /// `inner`, wrapped to send every request under `system`.
    pub(crate) fn wrap(inner: Arc<dyn ModelProvider>, system: &str) -> Arc<dyn ModelProvider> {
        Arc::new(Self {
            inner,
            system: system.to_string(),
        })
    }
}

/// `request` with `system` as its one system turn, first, and no dynamic
/// suffix.
fn under_system_prompt(system: &str, mut request: CompletionRequest) -> CompletionRequest {
    request.messages.retain(|m| m.role != Role::System);
    request.messages.insert(0, Message::system(system));
    request.system_dynamic_suffix = None;
    request
}

// The catalog lookups the trait derives from `name` and `model_name` reach
// the inner model's entries through the delegated names; the rest of the
// trait's per-model limits are delegated one by one.
#[async_trait::async_trait]
impl ModelProvider for UnderSystemPrompt {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<ResponseStream> {
        self.inner
            .complete(under_system_prompt(&self.system, request))
            .await
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

/// `inner` with `addendum` appended to the system turn of every request, a
/// blank line after what is there: the model asked under its system prompt
/// and one more thing it is told, as when a model is prompted with a goal.
/// Put inside a solve's own wrapping ([`UnderSystemPrompt`]), which sets the
/// system turn the addendum is appended to.
pub fn with_system_addendum(
    inner: Arc<dyn ModelProvider>,
    addendum: &str,
) -> Arc<dyn ModelProvider> {
    Arc::new(Addended {
        inner,
        addendum: addendum.to_string(),
    })
}

struct Addended {
    inner: Arc<dyn ModelProvider>,
    addendum: String,
}

#[async_trait::async_trait]
impl ModelProvider for Addended {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    async fn complete(&self, mut request: CompletionRequest) -> anyhow::Result<ResponseStream> {
        if let Some(system) = request.messages.iter_mut().find(|m| m.role == Role::System) {
            let text = format!(
                "{}\n\n{}",
                system.as_text().unwrap_or_default(),
                self.addendum
            );
            *system = Message::system(text);
        } else {
            request
                .messages
                .insert(0, Message::system(self.addendum.clone()));
        }
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
