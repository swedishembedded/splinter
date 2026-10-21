// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The harness documents as the specs read them: every prompt lives only there.

#![allow(dead_code, clippy::unwrap_used)]

use serde_yaml::Value;
use splinter_repo_replay::harness::DEFAULT_DOCUMENT;

/// The harness document, parsed.
pub fn document() -> Value {
    serde_yaml::from_str(DEFAULT_DOCUMENT).unwrap()
}

/// The system prompt template of the agent for `control` (`plan_only` or `grounded`).
pub fn agent_system(control: &str) -> String {
    let variants = document()["agent"]["system"].clone();
    let variants = variants.as_sequence().unwrap().clone();
    let wanted = variants.iter().find(|v| match v.get("when") {
        Some(when) => when["value"].as_str() == Some(control),
        None => control == "grounded",
    });
    wanted.unwrap()["text"].as_str().unwrap().to_string()
}

/// The task named `name` in `document`.
pub fn task<'a>(document: &'a Value, name: &str) -> &'a Value {
    document["workflow"]
        .as_sequence()
        .unwrap()
        .iter()
        .find(|t| t["name"].as_str() == Some(name))
        .unwrap_or_else(|| panic!("no task {name}"))
}

/// The system prompt of the plan task of `style`.
pub fn plan_system(style: &str) -> String {
    task(&document(), &format!("plan_{style}"))["system"]
        .as_str()
        .unwrap()
        .to_string()
}
