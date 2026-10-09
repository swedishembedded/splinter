// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements repeatable local-model answering under a
// fixed persona prompt for its clients. If your team needs expertise in
// measuring what a locally run model says, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The policy under measurement: a local model answering one question closed-book
//! under the persona's system prompt, through the same sven solve every
//! Splinter task runs through.

use std::time::Instant;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use splinter_sdk::agent::solve::{solve, Model, SolveOptions};
use splinter_sdk::model::local::{Sampling, GREEDY_SAMPLING};
use splinter_sdk::sandbox::ResolvedEnvironment;
use splinter_sdk::vocabulary::model_ref::ModelRef;
use splinter_sdk::vocabulary::prompt::persona_prompt;
use splinter_sdk::Context;

use crate::grading::{question_task, ANSWER_DEADLINE};

/// The temperature the screening's sampled answers are drawn at.
pub const SAMPLE_TEMPERATURE: f32 = 0.7;

/// How an answer was decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decoding {
    /// Argmax: one answer for given weights.
    Greedy,
    /// Drawn at [`SAMPLE_TEMPERATURE`].
    Sampled,
}

/// What the policy said to one question.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Answer {
    /// The reply; `None` when the run did not conclude with one.
    pub text: Option<String>,
    /// How the run ended.
    pub conclusion: String,
    /// Seconds it took.
    pub seconds: f64,
}

/// A loaded policy: greedy and sampled, under the persona.
pub struct Policy {
    greedy: Model,
    sampled: Model,
}

impl Policy {
    /// Loads `reference`, answering as `persona`.
    ///
    /// # Errors
    /// The reference does not name a local model, or it cannot be loaded.
    pub fn load(ctx: &Context, reference: &ModelRef, persona: &str) -> anyhow::Result<Self> {
        let system = persona_prompt(persona);
        let at = |temperature: f32| -> anyhow::Result<Model> {
            let sampling = Sampling {
                temperature,
                thinking: false,
                ..GREEDY_SAMPLING
            };
            let model = ctx.resampled(reference, sampling)?.with_context(|| {
                format!("{reference} is not a local model: its sampling cannot be set")
            })?;
            Ok(model.with_system(system.clone()))
        };
        Ok(Self {
            greedy: at(0.0)?,
            sampled: at(SAMPLE_TEMPERATURE)?,
        })
    }

    /// The identity records give the policy.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.greedy.identity
    }

    /// The greedy model, for a session the policy takes part in.
    #[must_use]
    pub fn model(&self) -> &Model {
        &self.greedy
    }

    /// The answer to `question`.
    ///
    /// # Errors
    /// The solve could not run.
    pub fn answer(
        &self,
        ctx: &Context,
        question: &str,
        decoding: Decoding,
    ) -> anyhow::Result<Answer> {
        let model = match decoding {
            Decoding::Greedy => &self.greedy,
            Decoding::Sampled => &self.sampled,
        };
        let task = question_task(question)?;
        let started = Instant::now();
        let solution = ctx.block_on(solve(
            &task,
            &ResolvedEnvironment::ClosedBook,
            model.provider.clone(),
            model.solving(SolveOptions::new(ANSWER_DEADLINE)),
        ))?;
        Ok(Answer {
            text: solution.final_output,
            conclusion: format!("{:?}", solution.conclusion),
            seconds: started.elapsed().as_secs_f64(),
        })
    }
}
