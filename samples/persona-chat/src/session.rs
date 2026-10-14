// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that act through tools a caller
// defines, for its clients. If your team needs expertise in measuring and
// tracing what a model does with a tool it has never seen, you can procure
// our services by sending an email to info@swedishembedded.com.

//! A conversation with a model that may call the caller's tools.

use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_sdk::agent::sven::model::{
    FunctionCall, Message, MessageContent, ModelProvider, Role,
};
use splinter_sdk::agent::tool_turn::tool_turn;

use crate::tools::ToolDef;

/// The most tokens a model may spend on one turn unless the caller says.
pub const MAX_TOKENS: u32 = 1024;

/// The most turns one thing said may take, a call and its answer being one:
/// the model speaks, calls, is answered, and speaks again.
pub const MAX_ROUNDS: usize = 4;

/// One call a model made and how it was answered.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CallTrace {
    /// The call's id.
    pub id: String,
    /// The tool named.
    pub name: String,
    /// The arguments as JSON; null when the model wrote something else.
    pub arguments: Value,
    /// Whether the tool exists and the arguments fit it.
    pub valid: bool,
    /// What the model was told in answer.
    pub result: String,
}

/// One turn of the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Round {
    /// What it said outside any call.
    pub text: String,
    /// Its calls.
    pub calls: Vec<CallTrace>,
    /// Tokens of the request, as the provider reports them.
    pub input_tokens: u32,
    /// Tokens of the reply, as the provider reports them.
    pub output_tokens: u32,
    /// How long the turn took.
    pub millis: u64,
}

/// What came of one thing said to the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Exchange {
    /// What was said.
    pub prompt: String,
    /// The model's turns, until it had nothing more to call.
    pub rounds: Vec<Round>,
    /// Whether the rounds allowed ran out with the model still calling.
    pub stopped: bool,
}

/// A conversation: the system prompt, the tools on offer and what has been
/// said so far.
pub struct Session {
    provider: Arc<dyn ModelProvider>,
    system: String,
    tools: Vec<ToolDef>,
    history: Vec<Message>,
    max_tokens: u32,
    max_rounds: usize,
}

impl Session {
    /// A conversation with `provider` under `system`, offering `tools`.
    pub fn new(
        provider: Arc<dyn ModelProvider>,
        system: impl Into<String>,
        tools: Vec<ToolDef>,
    ) -> Self {
        Self {
            provider,
            system: system.into(),
            tools,
            history: Vec::new(),
            max_tokens: MAX_TOKENS,
            max_rounds: MAX_ROUNDS,
        }
    }

    /// The most turns one thing said may take.
    #[must_use]
    pub fn rounds(mut self, rounds: usize) -> Self {
        self.max_rounds = rounds.max(1);
        self
    }

    /// The most tokens one turn may spend.
    #[must_use]
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Starts a conversation again: what was said is forgotten.
    pub fn forget(&mut self) {
        self.history.clear();
    }

    /// Says `text` to the model and follows it until it has nothing more to
    /// call, answering every call by its tool.
    ///
    /// # Errors
    /// The model failed.
    pub async fn say(&mut self, text: &str) -> anyhow::Result<Exchange> {
        self.history.push(Message::user(text));
        let mut exchange = Exchange {
            prompt: text.to_string(),
            rounds: Vec::new(),
            stopped: false,
        };
        for _ in 0..self.max_rounds {
            let mut messages = vec![Message::system(&self.system)];
            messages.extend(self.history.iter().cloned());
            let started = Instant::now();
            let turn = tool_turn(
                &self.provider,
                messages,
                self.tools.iter().map(ToolDef::schema).collect(),
                self.max_tokens,
            )
            .await?;
            if !turn.text.is_empty() {
                self.history.push(Message::assistant(&turn.text));
            }
            let mut calls = Vec::new();
            let mut last_word = false;
            for made in turn.calls {
                let answered = self.answer(&made.name, &made.arguments);
                last_word |= answered.0 && self.is_final(&made.name);
                self.history.push(Message {
                    role: Role::Assistant,
                    content: MessageContent::ToolCall {
                        tool_call_id: made.id.clone(),
                        function: FunctionCall {
                            name: made.name.clone(),
                            arguments: made.raw_arguments.clone(),
                        },
                    },
                });
                self.history
                    .push(Message::tool_result(&made.id, &answered.1));
                calls.push(CallTrace {
                    id: made.id,
                    name: made.name,
                    arguments: made.arguments,
                    valid: answered.0,
                    result: answered.1,
                });
            }
            let finished = calls.is_empty() || last_word;
            exchange.rounds.push(Round {
                text: turn.text,
                calls,
                input_tokens: turn.input_tokens,
                output_tokens: turn.output_tokens,
                millis: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
            });
            if finished {
                return Ok(exchange);
            }
        }
        exchange.stopped = true;
        Ok(exchange)
    }

    /// Whether a correct call to the tool `name` ends the exchange.
    fn is_final(&self, name: &str) -> bool {
        self.tools.iter().any(|t| t.name == name && t.final_call)
    }

    /// Whether the call is a good one, and the words the model is answered in.
    fn answer(&self, name: &str, arguments: &Value) -> (bool, String) {
        match self.tools.iter().find(|t| t.name == name) {
            None => (
                false,
                format!(
                    "refused: there is no tool {name:?}; the tools are {}",
                    self.tools
                        .iter()
                        .map(|t| t.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ),
            Some(tool) => match tool.check(arguments) {
                Ok(result) => (true, result.to_string()),
                Err(reason) => (false, reason),
            },
        }
    }
}
