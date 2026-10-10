// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that hold a conversation as a
// person from what that person wrote, for its clients. If your team needs
// expertise in measuring how a model behaves under a person's pressure, you
// can procure our services by sending an email to info@swedishembedded.com.

//! A model's next reply to a conversation it was not in.
//!
//! A solve is one instruction and the model's own answer; to ask what a model
//! says after a person pushes back on an answer it gave, the conversation
//! has to be put to it as it stands, with the assistant's turn written by
//! someone else. [`reply`] sends the turns given, in order, under the system
//! prompt the model answers under, and returns the text of what it says
//! next. No tools are offered and nothing is kept: the caller records the
//! reply where it belongs.

use std::time::Duration;

use futures::StreamExt;
use sven_sdk::model::{CompletionRequest, Message, ResponseEvent};
use sven_sdk::CancelToken;

use crate::solve::{Model, SYSTEM_PROMPT};

/// One turn of the conversation put to the model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Turn {
    /// Whether the assistant said it (`true`) or the person did.
    pub by_assistant: bool,
    /// What was said.
    pub text: String,
}

impl Turn {
    /// A turn the person said.
    #[must_use]
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            by_assistant: false,
            text: text.into(),
        }
    }

    /// A turn the assistant said.
    #[must_use]
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            by_assistant: true,
            text: text.into(),
        }
    }
}

/// The bounds of one reply.
#[derive(Clone, Debug)]
pub struct ChatOptions {
    /// The longest the reply may take.
    pub deadline: Duration,
    /// The most tokens it may run to; the model's own limit when `None`.
    pub max_output_tokens: Option<u32>,
    /// Stops the reply between two chunks of the stream.
    pub cancel: Option<CancelToken>,
}

impl ChatOptions {
    /// A reply bounded by `deadline` alone.
    #[must_use]
    pub fn new(deadline: Duration) -> Self {
        Self {
            deadline,
            max_output_tokens: None,
            cancel: None,
        }
    }
}

/// Why a reply was not had.
#[derive(Debug, thiserror::Error)]
pub enum ChatError {
    /// The model could not be reached or its stream failed.
    #[error("the model failed: {0}")]
    Model(String),
    /// The model said nothing.
    #[error("the model replied with nothing")]
    Empty,
    /// The deadline passed before the reply was whole.
    #[error("the reply took longer than {0:?}")]
    TimedOut(Duration),
    /// The cancel token fired.
    #[error("cancelled")]
    Cancelled,
}

/// The model's reply to `turns`; see the module documentation.
pub async fn reply(
    model: &Model,
    turns: &[Turn],
    options: &ChatOptions,
) -> Result<String, ChatError> {
    let system = model.system.as_deref().unwrap_or(SYSTEM_PROMPT);
    let messages = std::iter::once(Message::system(system))
        .chain(turns.iter().map(|turn| {
            if turn.by_assistant {
                Message::assistant(&turn.text)
            } else {
                Message::user(&turn.text)
            }
        }))
        .collect();
    let request = CompletionRequest {
        messages,
        stream: true,
        max_output_tokens_override: options.max_output_tokens,
        ..CompletionRequest::default()
    };
    let collected = tokio::time::timeout(options.deadline, collect(model, request, options));
    let text = collected
        .await
        .map_err(|_| ChatError::TimedOut(options.deadline))??;
    let text = text.trim();
    if text.is_empty() {
        return Err(ChatError::Empty);
    }
    Ok(text.to_string())
}

async fn collect(
    model: &Model,
    request: CompletionRequest,
    options: &ChatOptions,
) -> Result<String, ChatError> {
    let mut stream = model
        .provider
        .complete(request)
        .await
        .map_err(|e| ChatError::Model(e.to_string()))?;
    let mut text = String::new();
    while let Some(event) = stream.next().await {
        if options
            .cancel
            .as_ref()
            .is_some_and(CancelToken::is_cancelled)
        {
            return Err(ChatError::Cancelled);
        }
        match event.map_err(|e| ChatError::Model(e.to_string()))? {
            ResponseEvent::TextDelta(delta) => text.push_str(&delta),
            ResponseEvent::Done => break,
            _ => {}
        }
    }
    Ok(text)
}
