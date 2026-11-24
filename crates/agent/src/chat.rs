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

use std::sync::Arc;
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
    /// Ask the server for the policy trace of a bare reply: the prompt ids
    /// the model read, the generated ids and the policy's own
    /// log-probabilities of them (a brain server scores them over the whole
    /// vocabulary). A server that returns none leaves the trace empty.
    pub trace: bool,
}

impl ChatOptions {
    /// A reply bounded by `deadline` alone.
    #[must_use]
    pub fn new(deadline: Duration) -> Self {
        Self {
            deadline,
            max_output_tokens: None,
            cancel: None,
            trace: false,
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

/// A bare reply and what the server said about it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Reply {
    /// The text the model wrote, trimmed.
    pub text: String,
    /// The reasoning the model wrote before the text, trimmed; empty when it
    /// did not reason (or its server returns none).
    pub thinking: String,
    /// Tokens generated, reasoning included, when the server reported its
    /// usage; `None` otherwise.
    pub output_tokens: Option<u32>,
    /// The output-token limit ended the generation: the text is cut, so a
    /// verifier's verdict on it is not the model's answer.
    pub truncated: bool,
    /// The ids of the prompt as the model read it (asked with
    /// [`ChatOptions::trace`]); empty when the server returned none.
    pub prompt_token_ids: Vec<i64>,
    /// The generated ids, the stop token included when the server reports it.
    pub token_ids: Vec<i64>,
    /// The log-probability of each generated id, parallel to `token_ids`
    /// when the server reported both.
    pub logprobs: Vec<f64>,
    /// The served identity of the decode: who answered.
    pub served: Served,
}

/// Who served a reply, as the server reported it; a field the server did not
/// report is `None`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Served {
    /// The content digest of the adapter that answered; `None` for the base.
    pub adapter_digest: Option<String>,
    /// `sha256:<hex>` of the base checkpoint.
    pub base_digest: Option<String>,
    /// `sha256:<hex>` of the chat template's rendering of a fixed probe.
    pub chat_template_digest: Option<String>,
    /// The engine build.
    pub engine: Option<String>,
    /// The sampling seed the decode used.
    pub seed: Option<u64>,
}

/// The model's reply to `turns`; see the module documentation.
pub async fn reply(
    model: &Model,
    turns: &[Turn],
    options: &ChatOptions,
) -> Result<String, ChatError> {
    let system = model.system.as_deref().unwrap_or(SYSTEM_PROMPT);
    let messages = std::iter::once(Message::system(system))
        .chain(turns.iter().map(message))
        .collect();
    let text = ask(model, messages, options).await?.text;
    if text.is_empty() {
        return Err(ChatError::Empty);
    }
    Ok(text)
}

/// The model's reply to exactly `turns`, with no system prompt added: what a
/// measurement that states its whole request sends. A model that says
/// nothing has said the empty text, which is an answer a verifier grades, not
/// an error; a reply the output limit cut says so ([`Reply::truncated`]).
pub async fn reply_bare(
    model: &Model,
    turns: &[Turn],
    options: &ChatOptions,
) -> Result<Reply, ChatError> {
    ask(model, turns.iter().map(message).collect(), options).await
}

fn message(turn: &Turn) -> Message {
    if turn.by_assistant {
        Message::assistant(&turn.text)
    } else {
        Message::user(&turn.text)
    }
}

async fn ask(
    model: &Model,
    messages: Vec<Message>,
    options: &ChatOptions,
) -> Result<Reply, ChatError> {
    let request = CompletionRequest {
        messages,
        stream: true,
        max_output_tokens_override: options.max_output_tokens,
        want_logprobs: options.trace,
        ..CompletionRequest::default()
    };
    let collected = tokio::time::timeout(options.deadline, collect(model, request, options));
    let mut reply = collected
        .await
        .map_err(|_| ChatError::TimedOut(options.deadline))??;
    reply.text = reply.text.trim().to_string();
    reply.thinking = reply.thinking.trim().to_string();
    Ok(reply)
}

async fn collect(
    model: &Model,
    request: CompletionRequest,
    options: &ChatOptions,
) -> Result<Reply, ChatError> {
    // A bare reply is asked through the same bounded retry as every other
    // model call: one dropped connection must not fail a measurement. A
    // connection that drops while the reply streams is as transient as one
    // that drops before it, and the reply is only used whole, so it is asked
    // again from the start.
    let mut pause = crate::retry::FIRST_PAUSE;
    for attempt in 0..=crate::retry::RETRIES {
        match collect_once(model, request.clone(), options).await {
            Err(ChatError::Model(text))
                if attempt < crate::retry::RETRIES && crate::retry::is_transient_text(&text) =>
            {
                eprintln!(
                    "splinter: the model reply failed transiently ({text}); asking again in {pause:?} ({}/{})",
                    attempt + 1,
                    crate::retry::RETRIES
                );
                tokio::time::sleep(pause).await;
                pause = pause.saturating_mul(2);
            }
            other => return other,
        }
    }
    unreachable!("the last attempt returns")
}

async fn collect_once(
    model: &Model,
    request: CompletionRequest,
    options: &ChatOptions,
) -> Result<Reply, ChatError> {
    let mut stream = crate::retry::Retrying::wrap(Arc::clone(&model.provider))
        .complete(request)
        .await
        .map_err(|e| ChatError::Model(e.to_string()))?;
    let mut reply = Reply::default();
    while let Some(event) = stream.next().await {
        if options
            .cancel
            .as_ref()
            .is_some_and(CancelToken::is_cancelled)
        {
            return Err(ChatError::Cancelled);
        }
        match event.map_err(|e| ChatError::Model(e.to_string()))? {
            ResponseEvent::TextDelta(delta) => reply.text.push_str(&delta),
            ResponseEvent::ThinkingDelta(delta) => reply.thinking.push_str(&delta),
            ResponseEvent::Usage { output_tokens, .. } => reply.output_tokens = Some(output_tokens),
            ResponseEvent::MaxTokens => reply.truncated = true,
            ResponseEvent::PromptTokens { ids } => reply.prompt_token_ids = ids,
            ResponseEvent::Tokens { ids, logprobs } => {
                reply.token_ids.extend(ids);
                reply.logprobs.extend(logprobs);
            }
            ResponseEvent::Identity(identity) => {
                reply.served = Served {
                    adapter_digest: identity.adapter.map(|a| a.digest),
                    base_digest: identity.base_digest,
                    chat_template_digest: identity.chat_template_digest,
                    engine: identity.engine,
                    seed: identity.seed,
                };
            }
            ResponseEvent::Done => break,
            _ => {}
        }
    }
    Ok(reply)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use sven_sdk::model::{ModelProvider, ResponseStream};

    use super::*;

    /// Drops its first request, then says "ok".
    struct DropsOnce(AtomicU32);

    #[async_trait::async_trait]
    impl ModelProvider for DropsOnce {
        fn name(&self) -> &str {
            "scripted"
        }

        fn model_name(&self) -> &str {
            "scripted"
        }

        async fn complete(&self, _: CompletionRequest) -> anyhow::Result<ResponseStream> {
            if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
                anyhow::bail!("error sending request: connection closed before message completed");
            }
            Ok(Box::pin(futures::stream::iter([
                Ok(ResponseEvent::TextDelta("ok".into())),
                Ok(ResponseEvent::Done),
            ])))
        }
    }

    /// Starts a reply and loses the connection in the middle of it the first
    /// time, then says "ok".
    struct DropsMidReply(AtomicU32);

    #[async_trait::async_trait]
    impl ModelProvider for DropsMidReply {
        fn name(&self) -> &str {
            "scripted"
        }

        fn model_name(&self) -> &str {
            "scripted"
        }

        async fn complete(&self, _: CompletionRequest) -> anyhow::Result<ResponseStream> {
            let events: Vec<anyhow::Result<ResponseEvent>> =
                if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
                    vec![
                        Ok(ResponseEvent::TextDelta("o".into())),
                        Err(anyhow::anyhow!(
                        "error decoding response body: connection closed before message completed"
                    )),
                    ]
                } else {
                    vec![
                        Ok(ResponseEvent::TextDelta("ok".into())),
                        Ok(ResponseEvent::Done),
                    ]
                };
            Ok(Box::pin(futures::stream::iter(events)))
        }
    }

    /// Answers with a cut reply and its trace, recording whether a trace was
    /// asked for.
    struct CutWithTrace(std::sync::Mutex<Vec<bool>>);

    #[async_trait::async_trait]
    impl ModelProvider for CutWithTrace {
        fn name(&self) -> &str {
            "scripted"
        }

        fn model_name(&self) -> &str {
            "scripted"
        }

        async fn complete(&self, request: CompletionRequest) -> anyhow::Result<ResponseStream> {
            self.0.lock().unwrap().push(request.want_logprobs);
            Ok(Box::pin(futures::stream::iter([
                Ok(ResponseEvent::PromptTokens { ids: vec![1, 2, 3] }),
                Ok(ResponseEvent::TextDelta(" {\"a\": ".into())),
                Ok(ResponseEvent::Tokens {
                    ids: vec![7, 8],
                    logprobs: vec![-0.5, -0.25],
                }),
                Ok(ResponseEvent::MaxTokens),
                Ok(ResponseEvent::Done),
            ])))
        }
    }

    #[tokio::test]
    async fn a_bare_reply_says_when_the_output_limit_cut_it_and_carries_the_trace_asked_for() {
        let provider = Arc::new(CutWithTrace(std::sync::Mutex::new(Vec::new())));
        let model = Model::new(provider.clone(), "scripted");
        let mut options = ChatOptions::new(Duration::from_secs(30));
        options.trace = true;
        let said = reply_bare(&model, &[Turn::user("hi")], &options)
            .await
            .unwrap();
        assert_eq!(said.text, "{\"a\":");
        assert!(said.truncated, "a cut reply is not graded as the answer");
        assert_eq!(said.prompt_token_ids, [1, 2, 3]);
        assert_eq!(said.token_ids, [7, 8]);
        assert_eq!(said.logprobs, [-0.5, -0.25]);
        assert_eq!(*provider.0.lock().unwrap(), [true]);
    }

    /// Reasons, then answers, and reports its usage.
    struct Reasons;

    #[async_trait::async_trait]
    impl ModelProvider for Reasons {
        fn name(&self) -> &str {
            "scripted"
        }

        fn model_name(&self) -> &str {
            "scripted"
        }

        async fn complete(&self, _: CompletionRequest) -> anyhow::Result<ResponseStream> {
            Ok(Box::pin(futures::stream::iter([
                Ok(ResponseEvent::ThinkingDelta(" 2 + 2 is 4. ".into())),
                Ok(ResponseEvent::TextDelta(" 4 ".into())),
                Ok(ResponseEvent::Usage {
                    input_tokens: 9,
                    output_tokens: 12,
                    cache_read_tokens: 0,
                    cache_write_tokens: 0,
                    cost_usd: None,
                }),
                Ok(ResponseEvent::Done),
            ])))
        }
    }

    #[tokio::test]
    async fn a_bare_reply_keeps_the_reasoning_apart_from_the_answer() {
        let model = Model::new(Arc::new(Reasons), "scripted");
        let options = ChatOptions::new(Duration::from_secs(30));
        let said = reply_bare(&model, &[Turn::user("2+2?")], &options)
            .await
            .unwrap();
        assert_eq!(said.thinking, "2 + 2 is 4.");
        assert_eq!(said.text, "4");
        assert_eq!(said.output_tokens, Some(12));
    }

    #[tokio::test]
    async fn a_bare_reply_survives_one_dropped_connection() {
        let provider = Arc::new(DropsOnce(AtomicU32::new(0)));
        let model = Model::new(provider.clone(), "scripted");
        let options = ChatOptions::new(Duration::from_secs(30));
        let said = reply_bare(&model, &[Turn::user("hi")], &options)
            .await
            .unwrap();
        assert_eq!(said.text, "ok");
        assert_eq!(provider.0.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn a_reply_cut_in_the_middle_by_a_dropped_connection_is_asked_again() {
        let provider = Arc::new(DropsMidReply(AtomicU32::new(0)));
        let model = Model::new(provider.clone(), "scripted");
        let options = ChatOptions::new(Duration::from_secs(30));
        let said = reply_bare(&model, &[Turn::user("hi")], &options)
            .await
            .unwrap();
        assert_eq!(
            said.text, "ok",
            "the whole second reply, not a join of the two"
        );
        assert_eq!(provider.0.load(Ordering::SeqCst), 2);
    }
}
