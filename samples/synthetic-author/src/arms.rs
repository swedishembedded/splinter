// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Putting the examination to a model and reading what it does.
//!
//! The model is offered one tool and a situation. What it does is the call it
//! makes: the option the call names is compared with the one the beliefs
//! determine, as a string. A reply without a call is a wrong answer.

use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use splinter_sdk::agent::sven::model::{Message, ModelProvider, ToolSchema};
use splinter_sdk::agent::tool_turn::tool_turn;
use splinter_sdk::store::jsonl;

use crate::binding::item_digest;
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
    /// Whether the way the facts and options combine was not withheld from the
    /// corpus's planner (not that the corpus holds it: see `combo_trained`).
    pub combo_seen: bool,
    /// Whether some decision kept from the corpus has the item's facts and
    /// options; absent for the first version of the examination.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub combo_trained: Option<bool>,
    /// Whether the tool is one the model was taught with.
    pub tool_seen: bool,
    /// The tool called and its arguments, if a call was made.
    pub call: Option<Value>,
    /// Digest of the item as it was asked. Empty in outcomes written before
    /// outcomes carried it.
    #[serde(default)]
    pub item: String,
    /// Digest of the model that answered. Empty in outcomes written before
    /// outcomes carried it.
    #[serde(default)]
    pub model: String,
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
    let turn = tool_turn(
        provider,
        vec![
            Message::system(system),
            Message::user(render(&item.briefing, &item.options, &item.schema)),
        ],
        vec![ToolSchema {
            name: item.schema.name.to_string(),
            description: item.schema.description.to_string(),
            parameters: definition["function"]["parameters"].clone(),
            ..Default::default()
        }],
        max_tokens,
    )
    .await?;
    Ok(turn
        .calls
        .into_iter()
        .next()
        .map(|call| (call.name, call.arguments)))
}

/// Reads the outcomes in `path`, each item once (the first answer stands).
///
/// # Errors
/// The file cannot be read or a line is not an outcome.
pub fn read_outcomes(path: &Path) -> anyhow::Result<Vec<Outcome>> {
    Ok(jsonl::read_unique(path, |o: &Outcome| o.id.clone())?)
}

/// Puts every item of `items` to `provider` under `system`, appending an outcome
/// to `out` as each is done, marked with `model`, the digest of the model that
/// answers; items already in `out` are not asked again.
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
    model: &str,
) -> anyhow::Result<Vec<Outcome>> {
    run_items_with(
        provider,
        |_| system.to_string(),
        items,
        out,
        max_tokens,
        model,
    )
}

/// As [`run_items`], with the system turn chosen for each item.
///
/// # Errors
/// The provider failed, or `out` cannot be written.
pub fn run_items_with(
    provider: &Arc<dyn ModelProvider>,
    system: impl Fn(&Item) -> String,
    items: &[Item],
    out: &Path,
    max_tokens: u32,
    model: &str,
) -> anyhow::Result<Vec<Outcome>> {
    let mut done = read_outcomes(out)?;
    let runtime = tokio::runtime::Runtime::new()?;
    let log = jsonl::Log::open(out)?;
    for item in items {
        if done.iter().any(|o| o.id == item.id) {
            continue;
        }
        let call = runtime.block_on(ask(provider, &system(item), item, max_tokens))?;
        let outcome = Outcome {
            id: item.id.clone(),
            correct: grade(item, call.as_ref().map(|(n, a)| (n.as_str(), a))),
            topic_seen: item.topic_seen,
            combo_seen: item.combo_seen,
            combo_trained: item.combo_trained,
            tool_seen: item.tool_seen,
            call: call.map(|(name, args)| serde_json::json!({"name": name, "arguments": args})),
            item: item_digest(item).to_string(),
            model: model.to_string(),
        };
        log.append(&outcome)?;
        done.push(outcome);
    }
    Ok(done)
}
