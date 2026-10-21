// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The harness documents as the specs read them: every prompt lives only there.

#![allow(dead_code, clippy::unwrap_used)]

use serde_yaml::Value;
use splinter_repo_replay::harness::{DEFAULT_DOCUMENT, DEFAULT_PLAN_ONLY_DOCUMENT};

/// The grounded document, parsed.
pub fn grounded() -> Value {
    serde_yaml::from_str(DEFAULT_DOCUMENT).unwrap()
}

/// The plan-only document, parsed.
pub fn plan_only() -> Value {
    serde_yaml::from_str(DEFAULT_PLAN_ONLY_DOCUMENT).unwrap()
}

/// The system prompt of the agent in `document`.
pub fn agent_system(document: &Value) -> String {
    document["agent"]["system"].as_str().unwrap().to_string()
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
    task(&grounded(), &format!("plan_{style}"))["system"]
        .as_str()
        .unwrap()
        .to_string()
}
