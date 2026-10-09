// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that learn what a person teaches
// them in conversation, for its clients. If your team needs expertise in
// continual learning from user feedback, you can procure our services by
// sending an email to info@swedishembedded.com.

//! A model asked what a person taught in a session.
//!
//! The call is typed: the brief is the call's task, the reply's schema
//! ([`ClaimReply`]) its shape, and sven reads a reply as that shape and sends
//! one that is not, or one that breaks [`ClaimReply::check`] (too many
//! claims, an empty or over-long statement), back for correction up to the
//! policy's `repairs` times. It runs with no tools, bounded by a deadline and
//! an output-token budget that cover every attempt. A reply that is still not
//! usable after that is declined, with what was wrong: the session yields no
//! proposals, and the caller says so. What the model proposed is only
//! proposed; the gates rule on it.

use splinter_core::claim::ClaimProposal;
use splinter_knowledge::claims::extract::{ExtractInput, ExtractionPolicy, BRIEF, METHOD, ROLE};
use splinter_knowledge::claims::reply::ClaimReply;
use splinter_knowledge::session::SessionView;
use sven_sdk::{CallError, CancelToken};

use crate::solve::Model;
use crate::typed::TypedCall;

/// What the extractor made of a session.
#[derive(Debug)]
pub enum Extraction {
    /// The claims it proposed, in its order; possibly none.
    Proposals(Vec<ClaimProposal>),
    /// No usable reply: why.
    Declined(String),
}

/// Why a session could not even be put to the model.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ExtractError(pub String);

/// A model that proposes claims from sessions.
#[derive(Clone)]
pub struct ClaimExtractor {
    model: Model,
    policy: ExtractionPolicy,
    cancel: Option<CancelToken>,
}

impl ClaimExtractor {
    /// An extractor on `model`, bounded by the default policy.
    #[must_use]
    pub fn new(model: Model) -> Self {
        Self {
            model,
            policy: ExtractionPolicy::default(),
            cancel: None,
        }
    }

    /// Bounds each session's extraction by `policy`.
    #[must_use]
    pub fn with_policy(mut self, policy: ExtractionPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Stops the extraction in progress, and every later one, once `cancel`
    /// is cancelled.
    #[must_use]
    pub fn with_cancel(mut self, cancel: CancelToken) -> Self {
        self.cancel = Some(cancel);
        self
    }

    /// The string an extraction records for the model.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.model.identity
    }

    /// The claims the model proposes from `session` in its first pass.
    pub async fn extract(&self, session: &SessionView) -> Result<Extraction, ExtractError> {
        self.extract_pass(session, 0).await
    }

    /// The claims the model proposes from `session` in extraction pass `pass`
    /// (counted from 0): the passes read the session in different orders.
    pub async fn extract_pass(
        &self,
        session: &SessionView,
        pass: u32,
    ) -> Result<Extraction, ExtractError> {
        let mut call = TypedCall::<ClaimReply>::new(METHOD, BRIEF, ROLE, self.policy.deadline)
            .max_output_tokens(self.policy.max_output_tokens)
            .repairs(self.policy.repairs)
            .postcondition(ClaimReply::check);
        call.cancel = self.cancel.clone();
        match call
            .run(&self.model, &ExtractInput::of(session, pass))
            .await
        {
            Ok(reply) => Ok(Extraction::Proposals(reply.into_proposals())),
            Err(CallError::Invalid {
                attempts,
                detail,
                last,
                ..
            }) => Ok(Extraction::Declined(format!(
                "{detail} (after {attempts} attempt(s)); last reply: {last}"
            ))),
            Err(CallError::Stopped { conclusion }) => Ok(Extraction::Declined(format!(
                "the call ended without a reply: {conclusion:?}"
            ))),
            Err(e) => Err(ExtractError(e.to_string())),
        }
    }
}
