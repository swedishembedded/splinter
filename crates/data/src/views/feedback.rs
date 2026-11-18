// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions that turn graded agent attempts
// into feedback training data for its clients. If your team needs expertise
// in learning from verified successes and failures of agents then you can
// procure our services by sending an email to info@swedishembedded.com.

//! The feedback view: each verified attempt's continuation of its prompt,
//! desirable when it passed and undesirable when it failed, for KTO (brain's
//! `feedback` objective). No counterpart is needed: KTO weighs each
//! completion against the policy's divergence from its reference.
//!
//! An attempt is verified as [`super::attempts`] says: an infrastructure
//! failure, an undecided attempt and one decided below the minimum strength
//! are left out and counted, never an undesirable sample. Its conversation is
//! rendered as the model was shown it ([`Rendering`]); the messages before
//! its first own action are the prompt (for a continuation of another
//! trajectory, the copied prefix too: never trained on), and the rest to its
//! last action is the completion, whose assistant turns are trained on and
//! whose tool results are context. A completion that reads the repository's
//! history is left out unless the curation keeps such reads
//! ([`Exclusion::RepositoryHistory`], the sft-step view's policy).
//!
//! The classes are weighed so that each holds the same weight in all: the
//! smaller class's examples are weighed up by the ratio of the counts
//! ([`FeedbackBalance`], recorded in the manifest and taken by `train` as the
//! run's `desirable_weight` and `undesirable_weight`), inside the band
//! `[1, 4/3]` KTO's authors give for the ratio of the weighted classes. Each
//! record carries its attempt ([`crate::RolloutIdentity`]), and a projection
//! whose records name two tokenizers is refused
//! ([`ViewError::MixedTokenizerDigest`]).

use serde::{Deserialize, Serialize};
use splinter_core::annotation::Strength;
use splinter_core::harness::HarnessConversation;

use super::attempts::{
    context, continuation, last_action_from, one_tool_set, own_start, reads_history,
    tokenizer_digest, verified,
};
use super::rendering::Rendering;
use crate::curation::HistoryCuration;
use crate::{
    Corpus, Entry, Exclusion, Objective, Projection, Provenance, RecordBody, RolloutIdentity,
    Strip, View, ViewError,
};

/// The name every feedback record carries.
const NAME: &str = "feedback";

/// The two classes of a feedback dataset and the weight of each, as its
/// manifest records them.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeedbackBalance {
    /// Desirable examples: verified passes.
    pub desirable: usize,
    /// Undesirable examples: verified fails.
    pub undesirable: usize,
    /// The weight of a desirable example's term.
    pub desirable_weight: f64,
    /// The weight of an undesirable example's term.
    pub undesirable_weight: f64,
}

impl FeedbackBalance {
    /// `desirable` and `undesirable` examples, the smaller class weighed up
    /// so both weigh the same in all; one when a class is empty.
    #[must_use]
    pub fn of(desirable: usize, undesirable: usize) -> Self {
        let ratio = |more: usize, fewer: usize| {
            if fewer == 0 || more <= fewer {
                1.0
            } else {
                more as f64 / fewer as f64
            }
        };
        Self {
            desirable,
            undesirable,
            desirable_weight: ratio(undesirable, desirable),
            undesirable_weight: ratio(desirable, undesirable),
        }
    }
}

/// Judged completions of verified attempts.
#[derive(Clone, Debug)]
pub struct Feedback {
    min_strength: Strength,
    rendering: Rendering,
}

impl Feedback {
    /// The view, accepting decisions at `min_strength` or stronger, showing
    /// the student only the instruction, leaving out a completion that reads
    /// the repository's history.
    #[must_use]
    pub fn new(min_strength: Strength) -> Self {
        Self {
            min_strength,
            rendering: Rendering::default(),
        }
    }

    /// The same view under `strip`.
    #[must_use]
    pub fn with_strip(mut self, strip: Strip) -> Self {
        self.rendering.strip = strip;
        self
    }

    /// The same view doing `history` with a completion that reads the
    /// repository's history: [`HistoryCuration::Keep`] keeps it, either other
    /// mode leaves it out.
    #[must_use]
    pub fn history(mut self, history: HistoryCuration) -> Self {
        self.rendering.history = history;
        self
    }

    /// The same view judging only the conversations answered by one of the
    /// model `roles` ([`SftStep::only_roles`](super::SftStep::only_roles)).
    #[must_use]
    pub fn only_roles(mut self, roles: Vec<String>) -> Self {
        self.rendering.only_roles = (!roles.is_empty()).then_some(roles);
        self
    }

    /// The same view reading a run exported as one agent step as one call per
    /// turn ([`SftStep::serialise_calls`](super::SftStep::serialise_calls)).
    #[must_use]
    pub fn serialise_calls(mut self, serialise_calls: bool) -> Self {
        self.rendering.serialise_calls = serialise_calls;
        self
    }

    /// The judged completion `entry` is, and the tokenizer that encoded it.
    fn example(&self, entry: &Entry) -> Result<(RecordBody, Option<String>), Exclusion> {
        let harness = HarnessConversation::of(&entry.experience.trajectory);
        if !self.rendering.wants_role(harness.as_ref()) {
            return Err(Exclusion::RoleNotWanted);
        }
        let attempt = verified(entry, self.min_strength)?;
        let conversation = self.rendering.conversation_of(entry, harness.as_ref())?;
        let start = own_start(&conversation).ok_or(Exclusion::NoAction)?;
        let end = last_action_from(&conversation, start).ok_or(Exclusion::NoAction)?;
        let tools = one_tool_set(&conversation)?;
        let tokenizer = tokenizer_digest(entry)?;
        let completion = continuation(&conversation, start, end);
        if self.rendering.history != HistoryCuration::Keep && reads_history(&completion) {
            return Err(Exclusion::RepositoryHistory);
        }
        Ok((
            RecordBody::Feedback {
                prompt: context(&conversation, start),
                completion,
                desirable: attempt.passed,
                tools,
                identity: RolloutIdentity {
                    group: attempt.group,
                    rollouts: vec![attempt.name],
                    tokenizer_digest: tokenizer.clone(),
                    source: None,
                },
            },
            tokenizer,
        ))
    }
}

impl View for Feedback {
    fn name(&self) -> &str {
        NAME
    }

    fn objective(&self) -> Objective {
        Objective::Feedback
    }

    fn project(&self, corpus: &Corpus) -> Result<Projection, ViewError> {
        let mut projection = Projection::new(
            NAME,
            Objective::Feedback,
            Some(self.rendering.strip.clone()),
            Some(self.min_strength),
        );
        projection.curation = self.rendering.history.recorded();
        let mut tokenizer: Option<String> = None;
        let (mut desirable, mut undesirable) = (0, 0);
        for entry in corpus.entries() {
            let (body, named) = match self.example(entry) {
                Ok(example) => example,
                Err(reason) => {
                    projection.exclude(reason);
                    continue;
                }
            };
            match (&tokenizer, named) {
                (Some(first), Some(second)) if *first != second => {
                    return Err(ViewError::MixedTokenizerDigest {
                        first: first.clone(),
                        second,
                    })
                }
                (None, Some(named)) => tokenizer = Some(named),
                _ => {}
            }
            if matches!(
                body,
                RecordBody::Feedback {
                    desirable: true,
                    ..
                }
            ) {
                desirable += 1;
            } else {
                undesirable += 1;
            }
            projection.push(body, Provenance::of(vec![entry.id.clone()]));
        }
        projection.feedback = Some(FeedbackBalance::of(desirable, undesirable));
        Ok(projection)
    }
}
