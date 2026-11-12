// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that decide which documents a
// person or an agent should study to reach a goal, for its clients. If your
// team needs expertise in goal-directed selection of learning material you
// can procure our services by sending an email to info@swedishembedded.com.

//! A language model as the reader of the material it is shown.

use std::cell::{Cell, RefCell};
use std::sync::Arc;

use splinter_sdk::agent::sven::model::{Message, ModelProvider};
use splinter_sdk::agent::tool_turn::tool_turn;
use splinter_sdk::goal::{Query, Reader};
use splinter_sdk::Context;

/// The most tokens one answer may take: a value, not an essay.
const MAX_ANSWER_TOKENS: u32 = 48;

const INSTRUCTION: &str = "Answer the question with the value only, in as few words as possible. \
If passages are given, answer from them when they bear on the question. \
If passages disagree, trust the one whose header shows the higher authority, then the later \
date, then the higher version; a passage without a header counts for nothing against one with. \
If you do not know, answer: unknown.";

/// Asks a model one query at a time and counts what that cost. A failed
/// call is counted and answered with nothing, never with a guess; the run
/// that sees a failure reports it and does not claim a result.
pub struct ModelReader {
    ctx: Context,
    provider: Arc<dyn ModelProvider>,
    calls: Cell<usize>,
    prompt_chars: Cell<usize>,
    tokens: Cell<u64>,
    failures: RefCell<Vec<String>>,
    transcript: RefCell<Vec<serde_json::Value>>,
}

impl ModelReader {
    /// A reader over `provider`, run in `ctx`.
    pub fn new(ctx: Context, provider: Arc<dyn ModelProvider>) -> Self {
        Self {
            ctx,
            provider,
            calls: Cell::new(0),
            prompt_chars: Cell::new(0),
            tokens: Cell::new(0),
            failures: RefCell::new(Vec::new()),
            transcript: RefCell::new(Vec::new()),
        }
    }

    /// Calls made, characters of prompt sent, tokens the model reported.
    pub fn usage(&self) -> (usize, usize, u64) {
        (self.calls.get(), self.prompt_chars.get(), self.tokens.get())
    }

    /// Every question asked, the passages shown with it and the answer.
    pub fn transcript(&self) -> Vec<serde_json::Value> {
        self.transcript.borrow().clone()
    }

    /// What failed, one line each.
    pub fn failures(&self) -> Vec<String> {
        self.failures.borrow().clone()
    }

    fn ask(&self, query: &Query) -> String {
        let mut prompt = String::new();
        for (i, passage) in query.material.iter().enumerate() {
            prompt.push_str(&format!("Passage {}:\n{passage}\n\n", i + 1));
        }
        prompt.push_str(&format!("Question: {}", query.question));
        self.calls.set(self.calls.get() + 1);
        self.prompt_chars
            .set(self.prompt_chars.get() + prompt.len());
        let messages = vec![Message::system(INSTRUCTION), Message::user(&prompt)];
        match self.ctx.block_on(tool_turn(
            &self.provider,
            messages,
            Vec::new(),
            MAX_ANSWER_TOKENS,
        )) {
            Ok(turn) => {
                self.tokens.set(
                    self.tokens.get()
                        + u64::from(turn.input_tokens)
                        + u64::from(turn.output_tokens),
                );
                self.transcript.borrow_mut().push(serde_json::json!({
                    "question": query.question,
                    "passages": query.material.len(),
                    "answer": turn.text,
                    "truncated": turn.truncated,
                }));
                turn.text
            }
            Err(e) => {
                self.failures
                    .borrow_mut()
                    .push(format!("{}: {e}", query.question));
                String::new()
            }
        }
    }
}

impl Reader for ModelReader {
    fn read_all(&self, queries: &[Query]) -> Vec<String> {
        queries.iter().map(|q| self.ask(q)).collect()
    }
}
