// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agents that learn what a person teaches
// them in conversation, for its clients. If your team needs expertise in
// continual learning from user feedback, you can procure our services by
// sending an email to info@swedishembedded.com.

//! A model asked to write a claim in the other forms it is learned in
//! ([`splinter_knowledge::claims::forms`]): a typed call with no tools,
//! bounded like the extraction of claims. A reply that stays unusable after
//! correction yields no forms, with why.

use splinter_core::claim::Claim;
use splinter_knowledge::claims::extract::ExtractionPolicy;
use splinter_knowledge::claims::forms::{
    admit, Forms, FormsInput, FormsReply, BRIEF, METHOD, ROLE,
};
use sven_sdk::{CallError, CancelToken};

use crate::solve::Model;
use crate::typed::TypedCall;

/// A model that writes the forms of claims.
#[derive(Clone)]
pub struct FormsWriter {
    model: Model,
    policy: ExtractionPolicy,
    cancel: Option<CancelToken>,
}

impl FormsWriter {
    /// A writer on `model`, bounded by the default policy.
    #[must_use]
    pub fn new(model: Model) -> Self {
        Self {
            model,
            policy: ExtractionPolicy::default(),
            cancel: None,
        }
    }

    /// Stops the call in progress, and every later one, once `cancel` fires.
    #[must_use]
    pub fn with_cancel(mut self, cancel: CancelToken) -> Self {
        self.cancel = Some(cancel);
        self
    }

    /// The string a record names the model by.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.model.identity
    }

    /// The forms of `claim` the model writes and code admits; `Err` names
    /// why the call gave nothing.
    pub async fn write(&self, claim: &Claim) -> Result<Forms, String> {
        let mut call = TypedCall::<FormsReply>::new(METHOD, BRIEF, ROLE, self.policy.deadline)
            .max_output_tokens(self.policy.max_output_tokens)
            .repairs(self.policy.repairs)
            .postcondition(FormsReply::check);
        call.cancel = self.cancel.clone();
        match call.run(&self.model, &FormsInput::of(claim)).await {
            Ok(reply) => Ok(admit(claim, reply)),
            Err(CallError::Invalid {
                attempts, detail, ..
            }) => Err(format!("{detail} (after {attempts} attempt(s))")),
            Err(e) => Err(e.to_string()),
        }
    }
}
