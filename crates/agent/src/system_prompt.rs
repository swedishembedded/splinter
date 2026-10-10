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

use std::sync::{Arc, RwLock};

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
    with_live_addendum(inner, Arc::new(RwLock::new(addendum.to_string())))
}

/// [`with_system_addendum`] whose addendum is read at every request from
/// `addendum`, so what the model is told can change between turns. Nothing is
/// appended while it is empty.
pub fn with_live_addendum(
    inner: Arc<dyn ModelProvider>,
    addendum: Arc<RwLock<String>>,
) -> Arc<dyn ModelProvider> {
    Arc::new(Addended { inner, addendum })
}

struct Addended {
    inner: Arc<dyn ModelProvider>,
    addendum: Arc<RwLock<String>>,
}

/// `request` with `addendum` after its system turn, or as its system turn when
/// it has none; unchanged when `addendum` is empty.
fn append_addendum(mut request: CompletionRequest, addendum: &str) -> CompletionRequest {
    if addendum.is_empty() {
        return request;
    }
    if let Some(system) = request.messages.iter_mut().find(|m| m.role == Role::System) {
        let text = format!("{}\n\n{addendum}", system.as_text().unwrap_or_default());
        *system = Message::system(text);
    } else {
        request.messages.insert(0, Message::system(addendum));
    }
    request
}

#[async_trait::async_trait]
impl ModelProvider for Addended {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<ResponseStream> {
        let addendum = self
            .addendum
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        self.inner
            .complete(append_addendum(request, &addendum))
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

#[cfg(test)]
mod specs {
    use super::*;

    fn request(messages: Vec<Message>) -> CompletionRequest {
        CompletionRequest {
            messages,
            ..Default::default()
        }
    }

    #[test]
    fn an_addendum_follows_the_system_turn_and_an_empty_one_changes_nothing() {
        let with = append_addendum(
            request(vec![Message::system("be brief"), Message::user("hi")]),
            "taught: x",
        );
        assert_eq!(
            with.messages[0].as_text().unwrap_or_default(),
            "be brief\n\ntaught: x"
        );
        let without = append_addendum(request(vec![Message::system("be brief")]), "");
        assert_eq!(
            without.messages[0].as_text().unwrap_or_default(),
            "be brief"
        );
        let bare = append_addendum(request(vec![Message::user("hi")]), "taught: x");
        assert_eq!(bare.messages.len(), 2);
        assert_eq!(bare.messages[0].role, Role::System);
    }
}
