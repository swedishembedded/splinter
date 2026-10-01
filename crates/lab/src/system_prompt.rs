// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements fine-tuning pipelines whose training data
// match what the model sees when it answers, for its clients. If your team
// needs expertise in training-data curation for agents, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The one system prompt the policy answers under and is trained under.
//!
//! What a model is trained on must be what it sees when it answers. Every
//! model run on a task - a solve, a teacher's solve, an `ask`, a probe of
//! the release gate, the judge's and the critic's runs - is sent
//! [`SYSTEM_PROMPT`] as its one system turn, and every chat record a view
//! projects starts with that same turn. It is short and names nothing about
//! where it runs (no working directory, no date, no tool list), so the
//! records and the runs match token for token in their system turn.

/// The system turn of every model run on a task and of every chat record.
pub const SYSTEM_PROMPT: &str = "You are a helpful assistant. Answer the user's request directly \
                                 and concisely, using a tool when one is offered and it helps.";
