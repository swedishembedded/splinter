// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements repeatable evaluation of locally run
// language models for its clients. If your team needs expertise in measuring
// what a fine-tune changed, you can procure our services by sending an email
// to info@swedishembedded.com.

//! One local model answering one question at a time, greedily.
//!
//! An evaluation asks the same model twice: with no adapter, then with the one
//! training produced. Greedy decoding means a question has one answer for a
//! given set of weights, so a difference between the two is the weights'.

use std::path::Path;
use std::time::Instant;

use futures::StreamExt;
use sven_sdk::model::{
    CompletionRequest, Message, MessageContent, ModelProvider, ResponseEvent, Role,
};

use crate::error::PolicyError;
use crate::local::{LocalQwen, LocalWeights, GREEDY_SAMPLING};
use crate::residency::Residency;

/// What a model said to one question.
#[derive(Clone, Debug, Default)]
pub struct Reply {
    /// Its reasoning, when it reasoned.
    pub thinking: String,
    /// The visible reply after its reasoning.
    pub text: String,
    /// Tokens it generated.
    pub tokens: u32,
    /// Whether it ran out of its token budget.
    pub truncated: bool,
    /// Wall-clock seconds the answer took.
    pub seconds: f64,
}

/// A loaded model, with its adapter if it has one.
pub struct Answerer {
    model: LocalQwen,
    _residency: Residency,
}

impl Answerer {
    /// Loads the base at `base`, attaching `adapter` when given. `label` names
    /// the model in the provider seam's records.
    ///
    /// # Errors
    /// The checkpoint cannot be opened or the adapter does not fit it.
    pub fn load(
        base: &Path,
        adapter: Option<&Path>,
        context_tokens: u32,
        label: &str,
    ) -> Result<Self, PolicyError> {
        let residency = Residency::default();
        let weights = LocalWeights {
            base: base.to_path_buf(),
            adapter: adapter.map(Path::to_path_buf),
            context_tokens,
        };
        let model = LocalQwen::load(&residency, &weights, label)?.resampled(GREEDY_SAMPLING);
        Ok(Self {
            model,
            _residency: residency,
        })
    }

    /// The model's reply to `user` under the system message `system`, within
    /// `max_tokens` generated tokens.
    ///
    /// # Errors
    /// The generation failed on the device.
    pub async fn ask(&self, system: &str, user: &str, max_tokens: u32) -> anyhow::Result<Reply> {
        let message = |role, text: &str| Message {
            role,
            content: MessageContent::Text(text.to_string()),
        };
        let request = CompletionRequest {
            messages: vec![message(Role::System, system), message(Role::User, user)],
            max_output_tokens_override: Some(max_tokens),
            ..CompletionRequest::default()
        };
        let started = Instant::now();
        let mut stream = self.model.complete(request).await?;
        let mut reply = Reply::default();
        while let Some(event) = stream.next().await {
            match event? {
                ResponseEvent::TextDelta(delta) => reply.text.push_str(&delta),
                ResponseEvent::ThinkingDelta(delta) => reply.thinking.push_str(&delta),
                ResponseEvent::MaxTokens => reply.truncated = true,
                ResponseEvent::Usage { output_tokens, .. } => reply.tokens = output_tokens,
                ResponseEvent::Error(e) => anyhow::bail!("generation failed: {e}"),
                _ => {}
            }
        }
        reply.seconds = started.elapsed().as_secs_f64();
        Ok(reply)
    }
}
