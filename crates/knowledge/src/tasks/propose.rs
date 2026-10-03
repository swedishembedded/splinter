// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model-driven task generation that turns
// any source into verifiable training tasks, for its clients. If your team
// needs expertise in synthetic task generation or verifier design, you can
// procure our services by sending an email to info@swedishembedded.com.

//! What the generator needs from a model, and nothing about which model or
//! how it is run.
//!
//! A [`TaskProposer`] is shown a brief and some input and answers with a
//! [`Reply`] of the shape the brief describes. Running a model - a typed
//! call on an agent runtime, bounded by a deadline and an output budget - is
//! an adapter's business; the generator's is deciding which proposals are
//! admitted.

use std::time::Duration;

use async_trait::async_trait;

use super::reply::Reply;
use super::Rejection;

/// One request to a proposer.
#[derive(Clone, Debug)]
pub struct ProposalRequest<'a> {
    /// The name of the typed call.
    pub method: &'a str,
    /// The task the model is briefed with.
    pub brief: &'a str,
    /// The role the model is given.
    pub role: &'a str,
    /// What the model is shown: the call's input.
    pub input: serde_json::Value,
    /// How long the request may take, every attempt included.
    pub deadline: Duration,
    /// Output tokens the request may generate; `None` leaves it to the
    /// model's own default.
    pub max_output_tokens: Option<u64>,
    /// How many times a reply that is not the shape is sent back for
    /// correction, within the deadline and budget.
    pub repairs: u32,
}

/// What a proposer made of a request.
#[derive(Debug)]
pub enum Proposed {
    /// A reply of the shape asked for.
    Replied(Reply),
    /// No usable reply: why, in the generator's terms.
    Declined {
        /// The rejection the declined request is counted under.
        rejection: Rejection,
        /// What happened.
        detail: String,
    },
}

/// Why a proposer could not even attempt a request.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ProposeError(pub String);

/// A model asked for task proposals.
#[async_trait]
pub trait TaskProposer: Send + Sync {
    /// The string an experience's provenance records for the model.
    fn identity(&self) -> &str;

    /// The model's answer to `request`.
    async fn propose(&self, request: &ProposalRequest<'_>) -> Result<Proposed, ProposeError>;
}
