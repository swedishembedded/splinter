// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements verifier-rewarded training of language models
// for its clients. If your team needs expertise in training a model against
// checks that do not trust it, you can procure our services by sending an email
// to info@swedishembedded.com.

//! The reinforcement-learning surface of brain, for a caller that supplies a
//! task family and a verifier.
//!
//! brain runs one GRPO cycle over an [`Environment`]'s tasks, scored by a
//! [`Verifier`], gates the trained candidate against the untouched incumbent on
//! a separate draw, and publishes an adapter only on promote ([`Improve`]).
//! Prompts and completions are token ids, so a task family reads and writes
//! them through [`ChatTokenizer`], which renders a chat prompt exactly as
//! serving does. This module is the one place Splinter reaches those types:
//! no other crate touches brain.

pub use brain::{
    ChatMessage, ChatRequest, ChatTokenizer, Decision, Environment, Improve, ImproveOptions,
    ImproveOutcome, Reward, Step, StepOutcome, Task, Verifier,
};
