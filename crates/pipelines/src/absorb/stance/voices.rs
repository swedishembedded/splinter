// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data that teaches a model to hold
// what it knows and to say what it does not, for its clients. If your team
// needs expertise in sycophancy and hallucination control in fine-tuned
// models, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The models a night's pairs are made with, each asked in the voice the
//! policy is deployed in.

use std::time::Duration;

use splinter_agent::chat::{reply, ChatError, ChatOptions, Turn};
use splinter_agent::solve::Model;
use splinter_agent::CancelToken;
use splinter_core::model_ref::ModelRef;
use splinter_core::prompt::SYSTEM_PROMPT;
use splinter_model::local::GREEDY_SAMPLING;
use splinter_model::{Sampling, AGENT_SAMPLING};
use splinter_orchestrator::context::Context;
use splinter_orchestrator::error::OrchestratorError;

/// The temperature the supervised candidate is sampled at to find the reply
/// a pair rejects: warm enough that the replies it gives in to a push or
/// makes up are drawn, not only the one it would give greedily.
pub const REJECTION_TEMPERATURE: f32 = 1.0;

/// How long one reply may take.
const REPLY_DEADLINE: Duration = Duration::from_secs(600);

/// The model `reference` names, decoding greedily where its sampling can be
/// set here.
pub(super) fn greedy(ctx: &Context, reference: &ModelRef) -> Result<Model, OrchestratorError> {
    match ctx.resampled(reference, GREEDY_SAMPLING)? {
        Some(model) => Ok(model),
        None => ctx.model(reference),
    }
}

/// What `model` says to `turns`; `None` when it says nothing or takes too
/// long, which is a reply that was not had, not a failure of the night.
pub(super) fn ask(
    ctx: &Context,
    model: &Model,
    turns: &[Turn],
    cancel: &CancelToken,
) -> Result<Option<String>, OrchestratorError> {
    let options = ChatOptions {
        cancel: Some(cancel.clone()),
        ..ChatOptions::new(REPLY_DEADLINE)
    };
    match ctx.block_on(reply(model, turns, &options)) {
        Ok(said) => Ok(Some(said)),
        Err(ChatError::Empty | ChatError::TimedOut(_)) => Ok(None),
        Err(ChatError::Cancelled) => Err(OrchestratorError::Cancelled),
        Err(ChatError::Model(why)) => Err(OrchestratorError::Refused(format!(
            "{} failed on a stance conversation: {why}",
            model.identity
        ))),
    }
}

/// The models of a night's pairs.
pub(super) struct Voices<'a> {
    ctx: &'a Context,
    /// The policy as deployed, greedy: the replies the pairs are about.
    pub policy: Model,
    /// Writes the preferred reply, shown the true value.
    pub teacher: Model,
    /// The supervised candidate, sampled: where the rejected reply comes from.
    pub student: Model,
    /// Writes the facts and the questions.
    pub generator: Model,
    /// The system prompt the policy is deployed under.
    pub persona: String,
    /// Stops the stage.
    pub cancel: CancelToken,
    /// How many replies are drawn before a pair is given up.
    pub tries: u32,
}

/// The references the voices are made from.
pub(super) struct Roles<'a> {
    pub policy: &'a ModelRef,
    pub teacher: &'a ModelRef,
    pub generator: &'a ModelRef,
    /// The supervised candidate.
    pub student: &'a ModelRef,
    /// The policy's system prompt, `None` for the default.
    pub system: Option<&'a str>,
}

impl<'a> Voices<'a> {
    pub(super) fn new(
        ctx: &'a Context,
        roles: &Roles<'_>,
        cancel: CancelToken,
        tries: u32,
    ) -> Result<Self, OrchestratorError> {
        let persona = roles.system.unwrap_or(SYSTEM_PROMPT).to_string();
        let under_persona = |model: Model| match roles.system {
            Some(system) => model.with_system(system),
            None => model,
        };
        let warm = Sampling {
            temperature: REJECTION_TEMPERATURE,
            ..AGENT_SAMPLING
        };
        let student = match ctx.resampled(roles.student, warm)? {
            Some(model) => model,
            None => ctx.model(roles.student)?,
        };
        Ok(Self {
            ctx,
            policy: greedy(ctx, roles.policy)?,
            teacher: under_persona(ctx.model(roles.teacher)?),
            student: under_persona(student),
            generator: ctx.model(roles.generator)?,
            persona,
            cancel,
            tries,
        })
    }

    pub(super) fn ctx(&self) -> &'a Context {
        self.ctx
    }

    /// What `model` says to `turns`; see [`ask`].
    pub(super) fn reply(
        &self,
        model: &Model,
        turns: &[Turn],
    ) -> Result<Option<String>, OrchestratorError> {
        ask(self.ctx, model, turns, &self.cancel)
    }

    /// The first of up to `tries` replies of `model` to `turns` that `ok`
    /// accepts.
    pub(super) fn first(
        &self,
        model: &Model,
        turns: &[Turn],
        ok: impl Fn(&str) -> bool,
    ) -> Result<Option<String>, OrchestratorError> {
        for _ in 0..self.tries {
            if let Some(said) = self.reply(model, turns)? {
                if ok(&said) {
                    return Ok(Some(said));
                }
            }
        }
        Ok(None)
    }
}
