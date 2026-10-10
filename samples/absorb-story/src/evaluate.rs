// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements repeatable local-model answering for its
// clients. If your team needs expertise in measuring what a locally run model
// says after each update, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The fixed evaluation: every item asked in its wordings of the deployed
//! policy, closed book and greedy, through the same `ask` the command line
//! runs, one resident model for all of them.

use std::path::Path;
use std::str::FromStr;
use std::time::Instant;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use splinter_sdk::store::StateRoot;
use splinter_sdk::vocabulary::model_ref::ModelRef;
use splinter_sdk::{ask::ask, Config, Splinter};

use crate::item::Item;

/// The most context the policy is loaded with, as in the nightly runs.
pub const POLICY_CONTEXT_TOKENS: u32 = 8192;

/// One answer to one wording of one item.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Answer {
    /// The item id.
    pub item: String,
    /// Which wording, from 0.
    pub phrasing: usize,
    /// The question as asked.
    pub question: String,
    /// What the model said.
    pub text: String,
}

/// A whole evaluation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Evaluation {
    /// The model reference asked.
    pub policy: String,
    /// The release the policy alias resolved to, when there was one.
    pub release: Option<String>,
    /// Seconds the whole evaluation took.
    pub seconds: f64,
    /// Every answer, item by item and wording by wording.
    pub answers: Vec<Answer>,
}

/// Asks every wording of every item of `items` of `policy`, with the state
/// under `state`.
///
/// # Errors
/// The runtime cannot start, `policy` is not a model reference, or a question
/// cannot be answered.
pub fn evaluate(state: &Path, policy: &str, items: &[&Item]) -> anyhow::Result<Evaluation> {
    let mut config = Config::from_env();
    config.state_root = StateRoot::new(state);
    config.policy_context_tokens = Some(POLICY_CONTEXT_TOKENS);
    let splinter = Splinter::builder(config)
        .build()
        .context("starting the Splinter runtime")?;
    let ctx = splinter.context();
    let reference = ModelRef::from_str(policy).with_context(|| format!("model {policy:?}"))?;
    let started = Instant::now();
    let mut answers = Vec::new();
    let mut release = None;
    for item in items {
        for (phrasing, question) in item.phrasings.iter().enumerate() {
            let said = ask(&ctx, question, None, &reference)
                .with_context(|| format!("asking {} ({question:?})", item.id))?;
            release = said.release.map(|r| r.to_string());
            answers.push(Answer {
                item: item.id.clone(),
                phrasing,
                question: question.clone(),
                text: said.answer,
            });
        }
    }
    Ok(Evaluation {
        policy: policy.into(),
        release,
        seconds: started.elapsed().as_secs_f64(),
        answers,
    })
}
