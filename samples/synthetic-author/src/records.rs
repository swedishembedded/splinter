// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Training records: a matter put to the model, and the tool call that decides
//! it as the author did. Only the call is supervised.

use serde_json::{json, Value};

use crate::prompts::{render, SYSTEM};
use crate::scenario::Archetype;
use crate::tools::Schema;

/// One chat record in brain's `generic-messages-v2` format. Records of one
/// `group` are held out together.
#[must_use]
pub fn chat_record(
    group: &str,
    briefing: &str,
    options: &[Archetype],
    schema: &Schema,
    chose: Archetype,
    motivation: &str,
) -> Value {
    let position = options.iter().position(|&o| o == chose).unwrap_or(0);
    let arguments = json!({
        schema.choice: schema.label(chose, position),
        schema.reason: motivation,
    });
    json!({
        "messages": [
            {"role": "system", "content": SYSTEM, "train": false},
            {"role": "user", "content": render(briefing, options, schema), "train": false},
            {
                "role": "assistant",
                "content": "",
                "train": true,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": schema.name, "arguments": arguments.to_string()}
                }]
            }
        ],
        "tools": [schema.definition(options)],
        "metadata": {"group": group}
    })
}
