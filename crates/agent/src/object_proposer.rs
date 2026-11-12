// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements agent runtimes that run a model under
// bounded, replayable conditions, for its clients. If your team needs
// expertise in agent execution or typed model calls, you can procure our
// services by sending an email to info@swedishembedded.com.

//! A model asked for knowledge objects through sven.
//!
//! The call is typed ([`ObjectReply`]) and its postcondition is the whole admission check ([`check_reply`]): a reply with a refused proposal is
//! sent back to the model with the reasons, up to the call's repairs, within its deadline and token budget. What is still refused after the last
//! repair is handed on as it is and dropped by admission, so one stubborn proposal never costs a document its good ones.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use splinter_knowledge::objects::propose::{
    brief, check_reply, normalise, DocumentInput, ObjectProposer, ObjectReply, ProposeError,
    Vocabulary,
};
use splinter_knowledge::objects::ProposedObject;
use sven_sdk::{CallError, CancelToken};

use crate::solve::Model;
use crate::typed::TypedCall;

/// How long one document may take, every attempt included, by default.
pub const DEFAULT_DEADLINE: Duration = Duration::from_secs(900);
/// Output tokens one document's reply may use, by default.
pub const DEFAULT_MAX_OUTPUT_TOKENS: u64 = 6144;
/// How many times a reply with refused proposals is sent back, by default.
pub const DEFAULT_REPAIRS: u32 = 2;

/// An [`ObjectProposer`] that asks `model` through a typed sven call.
#[derive(Clone)]
pub struct SvenObjectProposer {
    model: Model,
    cancel: Option<CancelToken>,
    deadline: Duration,
    repairs: u32,
    max_output_tokens: u64,
}

impl SvenObjectProposer {
    /// A proposer on `model` with the default bounds.
    #[must_use]
    pub fn new(model: Model) -> Self {
        Self {
            model,
            cancel: None,
            deadline: DEFAULT_DEADLINE,
            repairs: DEFAULT_REPAIRS,
            max_output_tokens: DEFAULT_MAX_OUTPUT_TOKENS,
        }
    }

    /// Stops the request in progress, and every later one, once `cancel` is cancelled.
    #[must_use]
    pub fn with_cancel(mut self, cancel: CancelToken) -> Self {
        self.cancel = Some(cancel);
        self
    }

    /// How many times a reply with refused proposals is sent back.
    #[must_use]
    pub fn with_repairs(mut self, repairs: u32) -> Self {
        self.repairs = repairs;
        self
    }

    /// How long one document may take.
    #[must_use]
    pub fn with_deadline(mut self, deadline: Duration) -> Self {
        self.deadline = deadline;
        self
    }

    /// The model's identity, as a pack's provenance records it.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.model.identity
    }
}

#[async_trait]
impl ObjectProposer for SvenObjectProposer {
    async fn propose(
        &self,
        document: &DocumentInput,
        vocabulary: &Vocabulary,
    ) -> Result<Vec<ProposedObject>, ProposeError> {
        let (checked, vocab) = (Arc::new(document.clone()), Arc::new(vocabulary.clone()));
        let mut call = TypedCall::<ObjectReply>::new(
            "propose_objects",
            brief(vocabulary),
            "extractor",
            self.deadline,
        )
        .repairs(self.repairs)
        .max_output_tokens(self.max_output_tokens)
        .postcondition({
            let (checked, vocab) = (Arc::clone(&checked), Arc::clone(&vocab));
            move |reply: &ObjectReply| check_reply(&checked, reply, &vocab).map(|_| ())
        });
        call.cancel = self.cancel.clone();
        let reply = match call.run(&self.model, document).await {
            Ok(reply) => reply,
            // Out of repairs: keep what the last reply held; admission drops what is still wrong.
            Err(CallError::Invalid { last, .. } | CallError::Postcondition { last, .. }) => serde_json::from_str::<ObjectReply>(&last)
                .map_err(|e| ProposeError(format!("the reply is not objects: {e}")))?,
            Err(e) => return Err(ProposeError(e.to_string())),
        };
        let mut out = Vec::new();
        for mut proposal in reply.objects {
            proposal.document.clone_from(&document.document);
            // A proposal whose vocabulary cannot be put right is dropped here; the rest go on to admission.
            if normalise(&mut proposal, vocabulary).is_ok() {
                out.push(proposal);
            }
        }
        Ok(out)
    }
}
