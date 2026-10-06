// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent runtimes that run a model under
// bounded, replayable conditions, for its clients. If your team needs
// expertise in agent execution or typed model calls, you can procure our
// services by sending an email to info@swedishembedded.com.

//! A model asked to write user requests, through a typed sven call: the
//! brief is the call's task, the input says the domain and how many, and the
//! reply is a list of requests, each a string. Which requests are kept is
//! the caller's business.

use std::time::Duration;

use schemars::JsonSchema;
use serde::Deserialize;
use sven_sdk::{CallError, CancelToken};

use crate::solve::Model;
use crate::typed::TypedCall;

/// The role the model is given.
const ROLE: &str = "You write the requests users put to an assistant.";

/// The call's name.
const METHOD: &str = "propose_prompts";

/// The reply: the requests written.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Prompts {
    /// The requests, each standing on its own.
    pub prompts: Vec<String>,
}

/// One request for prompts.
#[derive(Clone, Debug)]
pub struct PromptRequest<'a, I: serde::Serialize> {
    /// The brief: what the requests are to be like.
    pub brief: &'a str,
    /// What the model is shown with the brief: the domain and the count.
    pub input: &'a I,
    /// How long the call may take, every attempt included.
    pub deadline: Duration,
    /// Output tokens the call may generate; `None` leaves it to the
    /// model's default.
    pub max_output_tokens: Option<u64>,
    /// How many times a reply that is not the shape is sent back.
    pub repairs: u32,
    /// Stops the call from outside.
    pub cancel: Option<CancelToken>,
}

/// The requests `model` writes for `request`, in its order; an empty reply
/// is sent back for correction as a malformed one is.
pub async fn propose_prompts<I: serde::Serialize + Sync>(
    model: &Model,
    request: &PromptRequest<'_, I>,
) -> Result<Vec<String>, CallError> {
    let mut call = TypedCall::<Prompts>::new(METHOD, request.brief, ROLE, request.deadline)
        .repairs(request.repairs)
        .postcondition(|reply: &Prompts| {
            if reply.prompts.iter().all(|p| p.trim().is_empty()) {
                return Err("write at least one request".into());
            }
            Ok(())
        });
    call.max_output_tokens = request.max_output_tokens;
    call.cancel = request.cancel.clone();
    let reply = call.run(model, request.input).await?;
    Ok(reply
        .prompts
        .into_iter()
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect())
}
