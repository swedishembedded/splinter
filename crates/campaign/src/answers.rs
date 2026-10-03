// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements learning systems whose every answer traces
// back to the model release that gave it, for its clients. If your team
// needs expertise in model provenance or auditable AI, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The answers `ask` gave, recorded so an answer traces back to the model
//! and the release that gave it.
//!
//! An answer is a document in the experience database. An [`AnswerId`] is the
//! digest of the record's canonical form (see `splinter_core::digest`), the
//! time it was asked included, so asking the same question twice records two
//! answers. A record is stored once and checked against its address on every
//! read. Every record carries [`ANSWER_FORMAT`]; a later format adds fields
//! beside it rather than changing what these mean.

use serde::{Deserialize, Serialize};
use splinter_core::digest::Digest;
use splinter_core::source::SourceId;
use splinter_store::workspace::Workspace;

use crate::error::CampaignError;
use crate::release::ReleaseId;

const ANSWER: &str = "answer";

/// The `format` every answer record carries.
pub const ANSWER_FORMAT: &str = "splinter-answer-v1";

/// An answer's id: the digest of its record's canonical form.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AnswerId(pub Digest);

impl AnswerId {
    /// `blake3:<hex>`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl std::fmt::Display for AnswerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// One answer, and what gave it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnswerRecord {
    /// Always [`ANSWER_FORMAT`].
    pub format: String,
    /// The question, as asked.
    pub question: String,
    /// The model's answer.
    pub answer: String,
    /// The identity of the model that answered.
    pub model: String,
    /// The model reference it was asked through (`policy:default`, ...).
    pub policy: String,
    /// The release a `policy:` reference resolved to; `None` for any other
    /// reference, and for the base before any release.
    pub release: Option<ReleaseId>,
    /// The source shown with the question, if any.
    pub open_book: Option<SourceId>,
    /// When it was asked, from the injected clock.
    pub asked_at: String,
}

/// The recorded answers.
#[derive(Clone, Debug)]
pub struct AnswerStore {
    workspace: Workspace,
}

impl AnswerStore {
    /// The store over `workspace`.
    #[must_use]
    pub fn new(workspace: &Workspace) -> Self {
        Self {
            workspace: workspace.clone(),
        }
    }

    /// Records `record` and returns its id; recording the same record again is
    /// a no-op.
    pub fn put(&self, record: &AnswerRecord) -> Result<AnswerId, CampaignError> {
        Ok(AnswerId(self.workspace.put_document(ANSWER, record)?))
    }

    /// The answer `id`, verified against its address.
    pub fn get(&self, id: &AnswerId) -> Result<AnswerRecord, CampaignError> {
        self.workspace
            .get_document(ANSWER, &id.0)?
            .ok_or_else(|| CampaignError::NotFound {
                what: "answer",
                id: id.to_string(),
            })
    }

    /// Every recorded answer's id, in id order.
    pub fn list(&self) -> Result<Vec<AnswerId>, CampaignError> {
        Ok(self
            .workspace
            .document_ids(ANSWER)?
            .into_iter()
            .map(AnswerId)
            .collect())
    }
}
