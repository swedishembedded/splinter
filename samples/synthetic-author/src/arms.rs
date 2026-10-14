// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Putting the examination to a model and reading what it does.
//!
//! The model is offered one tool and a situation. What it does is the call it
//! makes: the option the call names is compared with the one the beliefs
//! determine, as a string. A reply without a call is a wrong answer.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::Path;
use std::sync::Arc;

use anyhow::Context as _;
use futures::StreamExt as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_sdk::agent::sven::model::{
    CompletionRequest, Message, ModelProvider, ResponseEvent, ToolSchema,
};

use crate::exam::{grade, Item};
use crate::prompts::render;

/// What a model did with one item.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Outcome {
    /// The item.
    pub id: String,
    /// Whether the call named the option the beliefs determine.
    pub correct: bool,
    /// Whether the item's topic is one the author wrote about.
    pub topic_seen: bool,
    /// Whether the facts combine as some article has them.
    pub combo_seen: bool,
    /// Whether the tool is one the model was taught with.
    pub tool_seen: bool,
    /// The tool called and its arguments, if a call was made.
    pub call: Option<Value>,
}

/// The tool call `provider` makes when it is given `item` under `system`,
/// or nothing if it makes none.
///
/// # Errors
/// The provider failed.
pub async fn ask(
    provider: &Arc<dyn ModelProvider>,
    system: &str,
    item: &Item,
    max_tokens: u32,
) -> anyhow::Result<Option<(String, Value)>> {
    let definition = item.schema.definition(&item.options);
    let request = CompletionRequest {
        messages: vec![
            Message::system(system),
            Message::user(render(&item.briefing, &item.options, &item.schema)),
        ],
        tools: vec![ToolSchema {
            name: item.schema.name.to_string(),
            description: item.schema.description.to_string(),
            parameters: definition["function"]["parameters"].clone(),
            ..Default::default()
        }],
        stream: true,
        max_output_tokens_override: Some(max_tokens),
        ..Default::default()
    };
    let mut stream = provider.complete(request).await?;
    let mut calls: BTreeMap<u32, (String, String)> = BTreeMap::new();
    while let Some(event) = stream.next().await {
        match event? {
            ResponseEvent::ToolCall {
                index,
                name,
                arguments,
                ..
            } => {
                let call = calls.entry(index).or_default();
                if !name.is_empty() {
                    call.0 = name;
                }
                call.1.push_str(&arguments);
            }
            ResponseEvent::Error(e) => anyhow::bail!("the model failed: {e}"),
            ResponseEvent::Done => break,
            _ => {}
        }
    }
    Ok(calls.into_values().next().map(|(name, arguments)| {
        let args = serde_json::from_str(&arguments).unwrap_or(Value::Null);
        (name, args)
    }))
}

/// Reads the outcomes in `path`, each item once (the first answer stands).
///
/// # Errors
/// The file cannot be read or a line is not an outcome.
pub fn read_outcomes(path: &Path) -> anyhow::Result<Vec<Outcome>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut seen = std::collections::HashSet::new();
    let mut outcomes = Vec::new();
    for line in std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?
        .lines()
    {
        let outcome: Outcome =
            serde_json::from_str(line).with_context(|| format!("a line of {}", path.display()))?;
        // Runs that overlap append the same item twice; the first answer stands.
        if seen.insert(outcome.id.clone()) {
            outcomes.push(outcome);
        }
    }
    Ok(outcomes)
}

/// Puts every item of `items` to `provider` under `system`, appending an outcome
/// to `out` as each is done; items already in `out` are not asked again.
/// Returns the outcome of every item that is in `out` when it ends, in order.
///
/// # Errors
/// The provider failed, or `out` cannot be written.
pub fn run_items(
    provider: &Arc<dyn ModelProvider>,
    system: &str,
    items: &[Item],
    out: &Path,
    max_tokens: u32,
) -> anyhow::Result<Vec<Outcome>> {
    let mut done = read_outcomes(out)?;
    let runtime = tokio::runtime::Runtime::new()?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out)?;
    for item in items {
        if done.iter().any(|o| o.id == item.id) {
            continue;
        }
        let call = runtime.block_on(ask(provider, system, item, max_tokens))?;
        let outcome = Outcome {
            id: item.id.clone(),
            correct: grade(item, call.as_ref().map(|(n, a)| (n.as_str(), a))),
            topic_seen: item.topic_seen,
            combo_seen: item.combo_seen,
            tool_seen: item.tool_seen,
            call: call.map(|(name, args)| serde_json::json!({"name": name, "arguments": args})),
        };
        let mut line = serde_json::to_string(&outcome)?;
        line.push('\n');
        file.write_all(line.as_bytes())?;
        done.push(outcome);
    }
    Ok(done)
}
