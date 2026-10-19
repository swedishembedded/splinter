// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements automated improvement of agent harnesses
// that adopts a change only on verified, regression-free evidence, for its
// clients. If your team needs expertise in measuring whether a harness
// change really helped, you can procure our services by sending an email to
// info@swedishembedded.com.

//! A model as the proposer of candidate harnesses.
//!
//! The model is shown the parent document and how that document did when it
//! was last evaluated, and answers with edits ([`splinter_agent::harness_edit`]):
//! an edit only counts once sven's compiler accepts the document it makes. A
//! model that does not manage that in its repairs is replaced for that
//! proposal by a mechanical mutation of the space, when the run has one, and
//! the fallback is counted, so a proposer that mostly fails shows as one.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use serde::Serialize;
use splinter_agent::harness_edit::{propose_edit, EditRequest};
use splinter_agent::solve::Model;
use splinter_agent::sven::search::{Proposal, ProposalStep, Proposer, SearchError, SearchRng};
use splinter_agent::CancelToken;
use splinter_core::digest::Digest;

use super::space::MutationSpace;

/// What the proposer did over a run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ProposalStats {
    /// Proposals the model made that compiled.
    pub model_proposals: u64,
    /// Proposals the model failed to make compile, within its repairs.
    pub model_failures: u64,
    /// Failed proposals replaced by a mutation of the space.
    pub mutation_fallbacks: u64,
}

/// How each document did when last evaluated, for the model to read.
#[derive(Default)]
pub struct Feedback(Mutex<BTreeMap<Digest, String>>);

impl Feedback {
    /// Records how `document` did.
    pub fn record(&self, document: &str, summary: String) {
        if let Ok(mut notes) = self.0.lock() {
            notes.insert(Digest::of(document.as_bytes()), summary);
        }
    }

    fn of(&self, document: &str) -> String {
        self.0
            .lock()
            .ok()
            .and_then(|notes| notes.get(&Digest::of(document.as_bytes())).cloned())
            .unwrap_or_default()
    }
}

/// Asks `model` for edits to the parent document.
pub struct LlmProposer<'a> {
    /// The proposing model.
    pub model: Model,
    /// How the documents already evaluated did.
    pub feedback: &'a Feedback,
    /// What replaces a proposal the model could not make compile; without it
    /// such a proposal aborts the search.
    pub fallback: Option<MutationSpace>,
    /// How long one proposal may take, repairs included.
    pub deadline: Duration,
    /// Corrections a reply that does not compile may be sent back for.
    pub repairs: u32,
    /// Stops the run.
    pub cancel: CancelToken,
    stats: [AtomicU64; 3],
}

impl<'a> LlmProposer<'a> {
    /// A proposer on `model` that reads `feedback`.
    #[must_use]
    pub fn new(
        model: Model,
        feedback: &'a Feedback,
        fallback: Option<MutationSpace>,
        deadline: Duration,
        repairs: u32,
        cancel: CancelToken,
    ) -> Self {
        Self {
            model,
            feedback,
            fallback,
            deadline,
            repairs,
            cancel,
            stats: Default::default(),
        }
    }

    /// What the proposer has done so far.
    #[must_use]
    pub fn stats(&self) -> ProposalStats {
        let [ok, failed, fallback] = &self.stats;
        ProposalStats {
            model_proposals: ok.load(Ordering::SeqCst),
            model_failures: failed.load(Ordering::SeqCst),
            mutation_fallbacks: fallback.load(Ordering::SeqCst),
        }
    }
}

impl Proposer<String> for LlmProposer<'_> {
    async fn propose(
        &self,
        parent: &String,
        _step: &ProposalStep,
        rng: &mut SearchRng,
    ) -> Result<Proposal<String>, SearchError> {
        let feedback = self.feedback.of(parent);
        let edited = propose_edit(
            &self.model,
            EditRequest {
                document: parent,
                feedback: &feedback,
                deadline: self.deadline,
                repairs: self.repairs,
                cancel: Some(self.cancel.clone()),
            },
        )
        .await;
        match edited {
            Ok(edited) => {
                self.stats[0].fetch_add(1, Ordering::SeqCst);
                let spent = edited.spent;
                Ok(Proposal {
                    candidate: edited.document,
                    tokens: spent.input_tokens.unwrap_or(0) + spent.output_tokens.unwrap_or(0),
                    trajectories: Vec::new(),
                })
            }
            Err(error) => {
                self.stats[1].fetch_add(1, Ordering::SeqCst);
                let Some(space) = &self.fallback else {
                    return Err(SearchError::proposer(format!(
                        "the proposing model made no edit that compiles: {error}"
                    )));
                };
                self.stats[2].fetch_add(1, Ordering::SeqCst);
                space
                    .mutate_compiling(parent, rng)
                    .map(Proposal::free)
                    .map_err(SearchError::proposer)
            }
        }
    }
}
