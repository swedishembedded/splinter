// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that learn what a person teaches
// them in conversation, for its clients. If your team needs expertise in
// continual learning from user feedback, you can procure our services by
// sending an email to info@swedishembedded.com.

//! A model asked what a gate cannot decide by code about claims.
//!
//! The calls are typed, like the extractor's: the brief is the task, the
//! reply's schema the shape, a reply that is not that shape is sent back for
//! correction, and the call runs with no tools within a deadline. A reply
//! that stays unusable is an error, never a guess: the gate does not rule
//! on a pair it could not decide. The judge sees only the words it is asked
//! about, never the session they came from.

use std::time::Duration;

use splinter_core::claim::Claim;
use splinter_knowledge::claims::judge::{PairInput, PairReply, PAIR_BRIEF, PAIR_METHOD, PAIR_ROLE};
use splinter_knowledge::claims::{ClaimJudge, JudgeError, PairVerdict};
use sven_sdk::CancelToken;
use tokio::runtime::Handle;

use crate::solve::Model;
use crate::typed::TypedCall;

/// A [`ClaimJudge`] that asks `model` through typed sven calls.
pub struct SvenClaimJudge {
    model: Model,
    runtime: Handle,
    deadline: Duration,
    repairs: u32,
    cancel: Option<CancelToken>,
}

impl SvenClaimJudge {
    /// A judge on `model`, whose calls run on `runtime` within `deadline`
    /// each and are sent back for correction up to `repairs` times. The
    /// trait's methods block on `runtime`: call them outside its async tasks.
    #[must_use]
    pub fn new(model: Model, runtime: Handle, deadline: Duration, repairs: u32) -> Self {
        Self {
            model,
            runtime,
            deadline,
            repairs,
            cancel: None,
        }
    }

    /// Stops a call in progress, and every later one, once `cancel` is
    /// cancelled.
    #[must_use]
    pub fn with_cancel(mut self, cancel: CancelToken) -> Self {
        self.cancel = Some(cancel);
        self
    }

    /// The judge's model identity.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.model.identity
    }

    fn call<T>(
        &self,
        call: TypedCall<T>,
        input: &(impl serde::Serialize + Sync),
    ) -> Result<T, JudgeError>
    where
        T: serde::de::DeserializeOwned + sven_sdk::schemars::JsonSchema + Send + 'static,
    {
        // `block_on` panics on a runtime thread; refuse instead.
        if Handle::try_current().is_ok() {
            return Err(JudgeError(
                "the claim judge blocks on its runtime and cannot be asked from inside it".into(),
            ));
        }
        let mut call = call.repairs(self.repairs);
        call.cancel = self.cancel.clone();
        self.runtime
            .block_on(call.run(&self.model, input))
            .map_err(|e| JudgeError(format!("{}: {e}", self.model.identity)))
    }
}

impl ClaimJudge for SvenClaimJudge {
    fn pair(&self, earlier: &Claim, later: &Claim) -> Result<PairVerdict, JudgeError> {
        let call = TypedCall::<PairReply>::new(PAIR_METHOD, PAIR_BRIEF, PAIR_ROLE, self.deadline);
        Ok(self.call(call, &PairInput::of(earlier, later))?.verdict())
    }
}
