// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements delegated-task coding agents whose answers
// are machine-readable by construction. If your team needs expertise in
// strict-output model interfaces, you can procure our services by sending
// an email to info@swedishembedded.com.

//! One-shot fact question: the model's whole reply is one JSON object.
//!
//! `ask` exists so a delegating script never has to parse prose: the reply
//! is demanded as `{"answer": string}`, parsed strictly, and the parsed
//! object - not the raw reply - is what reaches stdout. A reply that
//! cannot be parsed exits 2, because a script reading a broken answer as
//! an answer is worse than reading nothing.

use anyhow::Context;
use splinter_policy::{complete_text, ModelSelection};

/// What one ask needs.
#[derive(Clone, Debug)]
pub struct AskOptions {
    pub question: String,
    /// The model asked - the same selection every stage makes.
    pub model: ModelSelection,
}

/// Asks one question and returns the parsed answer string. The caller
/// prints it wrapped as `{"answer": ...}` so stdout stays strictly JSON.
pub(crate) fn run(options: AskOptions) -> anyhow::Result<String> {
    let prompt = prompt(&options.question);
    let provider = options.model.provider()?;
    let rt = tokio::runtime::Runtime::new()?;
    let reply: String = rt
        .block_on(async { complete_text(provider.as_ref(), &prompt).await })
        .context("the model produced no reply")?;
    crate::explore::parse_answer_reply(&reply)
}

/// The exact prompt one question sees - shared with `eval-facts`, so an
/// evaluated reply is asked the same way an `ask`ed one is.
pub(crate) fn prompt(question: &str) -> String {
    let prompt = format!(
        "Answer the question below from your knowledge. Reply with EXACTLY one JSON object \
         and nothing else - no prose, no code fences:\n\
         {{\"answer\": string}}\n\n\
         QUESTION:\n{}",
        question
    );
    prompt
}
