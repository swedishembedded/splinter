// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A typed sven call whose prompt describes the reply's shape.
//!
//! sven sends a [`Method`]'s JSON schema as the request's response format;
//! a provider that cannot constrain decoding (a local model) ignores it,
//! and sven's contract leaves describing the schema in the prompt to the
//! caller. Without it a model is asked for a JSON object whose field names
//! it is never told.

use serde::de::DeserializeOwned;
use sven_sdk::schemars::JsonSchema;
use sven_sdk::Method;

/// `method` with `task` as its instructions, followed by the JSON schema
/// its reply must match - the same schema the provider is given.
#[must_use]
pub fn with_schema<T>(method: Method<T>, task: &str) -> Method<T>
where
    T: DeserializeOwned + JsonSchema,
{
    let schema = method.schema();
    method.task(format!(
        "{task}\n\nReply with nothing but one JSON object that matches this JSON schema:\n{schema}"
    ))
}
