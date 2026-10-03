// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent runtimes that run a model under
// bounded, replayable conditions, for its clients. If your team needs
// expertise in agent execution or typed model calls, you can procure our
// services by sending an email to info@swedishembedded.com.

//! A model asked for task proposals through sven.
//!
//! The call is typed: the brief is the call's task, the reply's schema its
//! shape, and sven reads a reply as that shape and sends one that is not back
//! for correction. It runs with no tools, bounded by a deadline and an
//! optional output-token budget that cover every attempt.

use async_trait::async_trait;
use splinter_knowledge::tasks::propose::{ProposalRequest, ProposeError, Proposed, TaskProposer};
use splinter_knowledge::tasks::reply::Reply;
use splinter_knowledge::tasks::Rejection;
use sven_sdk::{CallError, CancelToken};

use crate::solve::Model;
use crate::typed::TypedCall;

/// A [`TaskProposer`] that asks `model` through a typed sven call.
#[derive(Clone)]
pub struct SvenProposer {
    model: Model,
    cancel: Option<CancelToken>,
}

impl SvenProposer {
    /// A proposer on `model`.
    #[must_use]
    pub fn new(model: Model) -> Self {
        Self {
            model,
            cancel: None,
        }
    }

    /// Stops the request in progress, and every later one, once `cancel` is
    /// cancelled: a stopped request has no reply, declined as
    /// [`Rejection::NoReply`].
    #[must_use]
    pub fn with_cancel(mut self, cancel: CancelToken) -> Self {
        self.cancel = Some(cancel);
        self
    }
}

#[async_trait]
impl TaskProposer for SvenProposer {
    fn identity(&self) -> &str {
        &self.model.identity
    }

    async fn propose(&self, request: &ProposalRequest<'_>) -> Result<Proposed, ProposeError> {
        let mut call = TypedCall::<Reply>::new(
            request.method,
            request.brief,
            request.role,
            request.deadline,
        )
        .repairs(request.repairs);
        call.max_output_tokens = request.max_output_tokens;
        call.cancel = self.cancel.clone();
        match call.run(&self.model, &request.input).await {
            Ok(reply) => Ok(Proposed::Replied(reply)),
            Err(CallError::Invalid {
                attempts,
                detail,
                last,
                ..
            }) => Ok(Proposed::Declined {
                rejection: Rejection::Malformed,
                detail: format!("{detail} (after {attempts} attempt(s)); last reply: {last}"),
            }),
            Err(CallError::Stopped { conclusion }) => Ok(Proposed::Declined {
                rejection: Rejection::NoReply,
                detail: format!("the call ended without a reply: {conclusion:?}"),
            }),
            Err(e) => Err(ProposeError(e.to_string())),
        }
    }
}
